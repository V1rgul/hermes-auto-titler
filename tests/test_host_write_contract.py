"""Host write-path contract on Hermes 0.23.3's storage interface.

``_write`` has two paths:

* untitled / derived → ``set_auto_title``, one host transaction that writes title and llm provenance together and refuses a higher provenance;
* llm → llm → the storage interface has no compare-and-set for an equal provenance, so the plugin checks the title it read before the model call, then writes strictly in two steps (``set_session_title`` → readback → ``set_session_title_source``) and fails closed.

``RealSqliteHost`` backs those calls with a real ``sqlite3`` database with the host's ``sessions`` columns, not a mock.

``RecordingHost`` records every host attribute the plugin resolves while evaluating. ``test_host_stub_covers_every_host_api_the_plugin_calls`` goes red when the plugin gains or renames a store call that the stub or the real ``SessionStore`` adapter does not provide, instead of the plugin breaking against the host.

# 纪律：新增 self.db.* 调用点必须同批带一条执行得到它的测试。
# 契约按调用点录制，未被测试走到的新调用点会静默退回盲区。
"""

import sqlite3
from types import SimpleNamespace

from hermes_auto_titler.config import DEFAULTS
from hermes_auto_titler.host_store import SessionStore
from hermes_auto_titler.titler import AutoTitler

MSGS = [
    {"role": "user", "content": "帮我排查后台任务重复触发"},
    {"role": "assistant", "content": "已定位到重复事件"},
]


class RecordingHost:
    """Proxy recording every host attribute the plugin resolves on ``db``."""

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "accessed", set())

    def __getattr__(self, name):
        self.accessed.add(name)
        return getattr(self._inner, name)


class RealSqliteHost:
    """The ``SessionStore`` surface backed by a real autocommit ``sqlite3`` connection; each write is its own host transaction."""

    def __init__(self):
        self.conn = sqlite3.connect(":memory:", isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(
            "CREATE TABLE sessions ("
            " id TEXT PRIMARY KEY, title TEXT, title_source TEXT)"
        )
        self.writes = 0

    def add_session(self, sid, title, source):
        self.conn.execute(
            "INSERT INTO sessions (id, title, title_source) VALUES (?, ?, ?)",
            (sid, title, source),
        )

    # -- host API the plugin calls -------------------------------------
    def get_messages_as_conversation(self, session_id, include_ancestors=False, include_compacted=False):
        return MSGS

    def get_session_title(self, sid):
        row = self.conn.execute(
            "SELECT title FROM sessions WHERE id = ?", (sid,)
        ).fetchone()
        return row["title"] if row else None

    def get_session_title_source(self, sid):
        row = self.conn.execute(
            "SELECT title_source FROM sessions WHERE id = ?", (sid,)
        ).fetchone()
        return row["title_source"] if row else None

    def set_auto_title(self, sid, title, *, source):
        # The host writes over an untitled row or a strictly lower provenance only.
        self.writes += 1
        cur = self.conn.execute(
            "UPDATE sessions SET title = ?, title_source = ? "
            "WHERE id = ? AND (title IS NULL OR title_source = 'derived')",
            (title, source, sid),
        )
        return cur.rowcount > 0

    def set_session_title(self, sid, title):
        self.writes += 1
        cur = self.conn.execute(
            "UPDATE sessions SET title = ?, title_source = 'user' WHERE id = ?",
            (title, sid),
        )
        return cur.rowcount > 0

    def set_session_title_source(self, sid, source):
        self.writes += 1
        cur = self.conn.execute(
            "UPDATE sessions SET title_source = ? WHERE id = ? AND title IS NOT NULL",
            (source, sid),
        )
        return cur.rowcount > 0

    def list_sessions_rich(self, limit=20, min_message_count=0, include_children=False, order_by_last_active=False):
        cur = self.conn.execute(
            "SELECT id, title, title_source FROM sessions ORDER BY id LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]

    def record_auxiliary_usage(self, session_id, task, **kwargs):
        # Best-effort usage accounting; recorded so the plugin can be verified
        # to call it through the real host surface.
        self.usage = getattr(self, "usage", [])
        self.usage.append({"session_id": session_id, "task": task, **kwargs})


class FakeLlm:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text=self.text, model="m", provider="p", usage=None)


def _dec(action, title=""):
    import json

    return json.dumps({"action": action, "title": title})


def _titler(db, text, cfg=None):
    ctx = SimpleNamespace(llm=FakeLlm(text))
    # rename_confirmations=0: a rename candidate applies in the same evaluate()
    # call, so these tests exercise the write path directly (the two-turn
    # review protocol has its own coverage elsewhere).
    return AutoTitler(
        ctx, {**DEFAULTS, "rename_confirmations": 0, **(cfg or {})}, db=db
    )


def test_llm_rename_lands_through_two_step_fail_closed_write():
    # llm → llm: set_auto_title would refuse an equal provenance, so the plugin writes the title (user provenance) and then restores llm provenance.
    host = RealSqliteHost()
    host.add_session("s1", "旧标题", "llm")
    t = _titler(host, _dec("rename", "新标题"))

    r = t.evaluate("s1", force=True)

    assert r["action"] == "renamed"
    assert host.get_session_title("s1") == "新标题"
    assert host.get_session_title_source("s1") == "llm"
    # write (set_session_title) + source recovery; readback is a plain read
    assert host.writes == 2


def test_llm_rename_aborts_when_title_changed_during_model_call():
    # The title the plugin read before the model call no longer matches: another writer renamed the session meanwhile, so the plugin must not overwrite it.
    host = RealSqliteHost()
    host.add_session("s1", "旧标题", "llm")
    t = _titler(host, _dec("rename", "新标题"))
    host.conn.execute(
        "UPDATE sessions SET title = '外部写入' WHERE id = 's1'"
    )

    r = t._write(host, "s1", "新标题", expected_title="旧标题")

    assert r is None
    assert host.get_session_title("s1") == "外部写入"
    assert host.get_session_title_source("s1") == "llm"
    assert host.writes == 0


def test_user_title_set_before_write_is_not_overwritten():
    # user authority wins: a user title present when the write starts stops it before any host write.
    host = RealSqliteHost()
    host.add_session("s1", "旧标题", "llm")
    t = _titler(host, _dec("rename", "新标题"))
    t._pending["s1"] = {"title": "新标题"}

    host.conn.execute(
        "UPDATE sessions SET title = '用户手改', title_source = 'user' WHERE id = 's1'"
    )

    r = t._write(host, "s1", "新标题", expected_title="旧标题")

    assert r is None
    assert host.get_session_title("s1") == "用户手改"
    assert host.get_session_title_source("s1") == "user"
    assert host.writes == 0


def test_two_step_write_aborts_when_readback_disagrees():
    host = RealSqliteHost()
    host.add_session("s1", "旧标题", "llm")
    t = _titler(host, _dec("rename", "新标题"))

    # Another writer lands between set_session_title and the readback.
    real_set = host.set_session_title

    def racing_set(sid, title):
        ok = real_set(sid, title)
        host.conn.execute(
            "UPDATE sessions SET title = '外部写入' WHERE id = ?", (sid,)
        )
        return ok

    host.set_session_title = racing_set
    r = t._write(host, "s1", "新标题")

    assert r is None
    assert host.get_session_title("s1") == "外部写入"


def test_untitled_session_uses_set_auto_title_only():
    # None/derived source must take the host's single-transaction set_auto_title
    # (title + source written atomically), never the two-step write.
    host = RealSqliteHost()
    host.add_session("s1", None, None)
    t = _titler(host, _dec("rename", "新标题"))

    r = t._write(host, "s1", "新标题")

    assert r == "新标题"
    assert host.get_session_title("s1") == "新标题"
    assert host.get_session_title_source("s1") == "llm"
    assert host.writes == 1  # set_auto_title only


def test_host_stub_covers_every_host_api_the_plugin_calls():
    # Tripwire: every host attribute the plugin resolves must exist on the real SessionStore adapter and be modelled by the stub. Recorded at runtime (not read from source), so it reddens on plugin-side drift — a new or renamed store call — instead of the plugin failing against the host.
    expected_host_api = {
        "get_messages_as_conversation",
        "get_session_title",
        "get_session_title_source",
        "set_auto_title",
        "set_session_title",
        "set_session_title_source",
        "list_sessions_rich",
        "record_auxiliary_usage",
    }

    host = RealSqliteHost()
    host.add_session("s1", "旧标题", "llm")
    rec = RecordingHost(host)
    t = _titler(rec, _dec("rename", "新标题"))
    t.evaluate("s1", force=True)          # llm → llm rename (two-step path)
    t._write(rec, "s1", "另一个标题")
    t.retitle_all(dry_run=True)            # list_sessions_rich path
    # None/derived source → set_auto_title path
    host.add_session("s2", None, None)
    t._write(rec, "s2", "第二标题")

    host_api_used = {a for a in rec.accessed if not a.startswith("__")}
    missing_on_adapter = {a for a in host_api_used if not callable(getattr(SessionStore, a, None))}
    assert not missing_on_adapter, (
        "plugin resolved store attributes the real SessionStore does not provide: "
        f"{sorted(missing_on_adapter)}"
    )
    uncovered = host_api_used - expected_host_api
    assert not uncovered, (
        "plugin resolved host attributes the stub does not model: "
        f"{sorted(uncovered)} — add them to the host stub"
    )
    # Every modelled attribute must actually have been resolved while driving
    # the paths above — a declared-but-unused entry is dead weight in the stub.
    never_used = expected_host_api - host_api_used
    assert not never_used, (
        "stub models host attributes the plugin never resolves on these paths: "
        f"{sorted(never_used)} — drop them or drive the path that uses them"
    )
