"""真实宿主会话存储集成验证：通过插件的 open_session_store 检查标题来源、更新、唯一性、对话读取与列表行为。

Run with the Hermes interpreter; it works on hosts with the storage interface (Hermes 0.23.3+) and on older hosts with hermes_state.SessionDB. The store lives in a temporary HERMES_HOME, never in a real profile.
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _seed(sessions):
    """Create the sessions and transcripts through the host's own write API."""
    try:
        from persistence_assembly import open_persistence, profile_scope
    except ImportError:
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            for sid, messages in sessions.items():
                db.create_session(sid, source="cli")
                for role, content in messages:
                    db.append_message(sid, role=role, content=content)
        finally:
            db.close()
        return
    from persistence.conversation.messages import NewMessage
    from persistence.conversation.sessions import NewSession

    with open_persistence(profile_scope(), eager=True) as store:
        for sid, messages in sessions.items():
            store.sessions.create(NewSession(session_id=sid, source="cli"))
            for role, content in messages:
                store.messages.append(sid, NewMessage(role=role, content=content))


def main() -> None:
    checks = []
    with tempfile.TemporaryDirectory(prefix="autotitler-int-") as tmp_dir:
        # Set before any host import resolves the profile, so every handle opens the temporary store.
        os.environ["HERMES_HOME"] = tmp_dir
        from hermes_auto_titler.host_store import open_session_store
        from hermes_auto_titler.titler import TITLE_SOURCE_LLM, TITLE_SOURCE_USER, _is_title_conflict

        s1 = "sess-0001"
        s2 = "sess-0002"
        _seed({
            s1: [("user", "帮我排查后台任务重复触发"), ("assistant", "已定位到重复事件")],
            s2: [("user", "请保留我的手改标题")],
        })
        db = open_session_store()
        try:
            # 无标题 → 原生自动标题写入成功，并记录 llm 来源。
            ok = db.set_auto_title(s1, "后台任务重复触发", source=TITLE_SOURCE_LLM)
            assert ok is True
            assert db.get_session_title(s1) == "后台任务重复触发"
            assert db.get_session_title_source(s1) == TITLE_SOURCE_LLM
            checks.append(("untitled→llm write", "OK"))

            # Hermes 的 set_auto_title 不会直接覆盖已有 llm 标题。
            ok2 = db.set_auto_title(s1, "后台任务去重", source=TITLE_SOURCE_LLM)
            assert ok2 is False
            assert db.get_session_title(s1) == "后台任务重复触发"
            checks.append(("llm→llm auto-title no-op", "OK"))

            # 插件采用权威标题写入后恢复 llm provenance。
            ok3 = db.set_session_title(s1, "后台任务去重")
            assert ok3 is True
            assert db.set_session_title_source(s1, TITLE_SOURCE_LLM) is True
            assert db.get_session_title(s1) == "后台任务去重"
            assert db.get_session_title_source(s1) == TITLE_SOURCE_LLM
            checks.append(("plugin update path", "OK"))

            # 手改标题标记为 user 来源，供插件入口拒绝自动覆盖。
            assert db.set_session_title(s2, "用户手改标题") is True
            assert db.get_session_title_source(s2) == TITLE_SOURCE_USER
            checks.append(("user provenance", "OK"))

            # 标题唯一性冲突必须抛出插件能识别的错误，插件再负责生成安全后缀重试。
            try:
                db.set_session_title(s1, "用户手改标题")
            except Exception as exc:
                assert _is_title_conflict(exc, "用户手改标题"), repr(exc)
                checks.append(("unique-title conflict", "OK"))
            else:
                raise AssertionError("duplicate title did not raise")

            conv = db.get_messages_as_conversation(s1, include_ancestors=True, include_compacted=True)
            assert [(m.get("role"), m.get("content")) for m in conv] == [
                ("user", "帮我排查后台任务重复触发"),
                ("assistant", "已定位到重复事件"),
            ]
            checks.append(("conversation read", "OK"))

            rows = db.list_sessions_rich(limit=10, include_children=False, order_by_last_active=True)
            assert {row.get("id") for row in rows} == {s1, s2}
            rows = db.list_sessions_rich(limit=10, min_message_count=2, include_children=False)
            assert [row.get("id") for row in rows] == [s1]
            checks.append(("rich session listing", "OK"))

            assert db.resolve_session_id("sess-0001") == s1
            checks.append(("session id resolution", "OK"))

            db.record_auxiliary_usage(s1, "hermes_auto_titler", model="probe-model", input_tokens=12, output_tokens=3)
            checks.append(("auxiliary usage recording", "OK"))
        finally:
            db.close()

    for check in checks:
        print(" | ".join(check))
    print("temporary database cleanup | OK")


if __name__ == "__main__":
    main()
