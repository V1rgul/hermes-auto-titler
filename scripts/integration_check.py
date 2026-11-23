"""真实宿主会话存储集成验证：通过插件的 open_session_store 检查标题清洗、来源、更新、占用、对话读取、列表、ID 解析与用量记账。

Run with the Hermes interpreter (Hermes 0.23.3+). The store lives in a temporary HERMES_HOME, never in a real profile.
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _seed():
    """Create sessions through the host's own write API: a compacted transcript, a rewound one, a compression lineage, a user-titled session and an ended empty session holding a user title."""
    from persistence_assembly import open_persistence, profile_scope
    from persistence.conversation.messages import NewMessage, ReplaceMode
    from persistence.conversation.sessions import END_REASON_COMPRESSION, NewSession

    with open_persistence(profile_scope(), eager=True) as store:
        def session(sid, messages, **kwargs):
            store.sessions.create(NewSession(session_id=sid, source="cli", **kwargs))
            for role, content in messages:
                store.messages.append(sid, NewMessage(role=role, content=content))

        session("sess-0001", [("user", "帮我排查后台任务重复触发"), ("assistant", "已定位到重复事件")])
        store.messages.compact("sess-0001", [NewMessage(role="user", content="[摘要] 已排查重复触发"), NewMessage(role="assistant", content="ok")])
        session("sess-0002", [("user", "请保留我的手改标题"), ("assistant", "好的"), ("user", "这一轮会被撤回"), ("assistant", "x")])
        store.messages.replace_transcript("sess-0002", [NewMessage(role="user", content="请保留我的手改标题"), NewMessage(role="assistant", content="好的")], ReplaceMode.ARCHIVE_DROPPED)
        store.sessions.set_title("sess-0002", "用户手改标题")
        session("lineage-root", [("user", "压缩前的请求")])
        store.sessions.end("lineage-root", END_REASON_COMPRESSION)
        session("lineage-tip", [("user", "压缩后的请求")], parent_session_id="lineage-root")
        session("ghost", [])
        store.sessions.set_title("ghost", "被占用的标题")
        store.sessions.end("ghost", "user_exit")
        # Activity order differs from creation order: sess-0001 is the oldest session but the most recently active one.
        store.messages.append("sess-0001", NewMessage(role="user", content="压缩后的继续消息"))


def main() -> None:
    checks = []
    with tempfile.TemporaryDirectory(prefix="autotitler-int-") as tmp_dir:
        # Set before any host import resolves the profile, so every handle opens the temporary store.
        os.environ["HERMES_HOME"] = tmp_dir
        from hermes_auto_titler.host_store import open_session_store
        from hermes_auto_titler.titler import TITLE_SOURCE_DERIVED, TITLE_SOURCE_LLM, TITLE_SOURCE_USER, _is_title_conflict

        _seed()
        db = open_session_store()
        s1, s2 = "sess-0001", "sess-0002"
        try:
            # 宿主按清洗后的形式存储标题；插件以此形式写入与比较。
            assert db.sanitize_title("Ship\u200b  plan ") == "Ship plan"
            assert db.sanitize_title("\u200b") is None
            checks.append(("title sanitization", "OK"))

            # 无标题 → 原生自动标题写入成功，并记录 llm 来源。
            assert db.set_auto_title(s1, "后台任务重复触发", source=TITLE_SOURCE_LLM) is True
            assert db.get_session_title(s1) == "后台任务重复触发"
            assert db.get_session_title_source(s1) == TITLE_SOURCE_LLM
            checks.append(("untitled→llm write", "OK"))

            # Hermes 的 set_auto_title 不会直接覆盖已有 llm 标题。
            assert db.set_auto_title(s1, "后台任务去重", source=TITLE_SOURCE_LLM) is False
            assert db.get_session_title(s1) == "后台任务重复触发"
            checks.append(("llm→llm auto-title no-op", "OK"))

            # 插件的 llm → llm 替换：先降为 derived，再由同一个宿主 CAS 写入。
            assert db.set_session_title_source(s1, TITLE_SOURCE_DERIVED) is True
            assert db.get_session_title_source(s1) == TITLE_SOURCE_DERIVED
            assert db.set_auto_title(s1, "后台任务去重", source=TITLE_SOURCE_LLM) is True
            assert (db.get_session_title(s1), db.get_session_title_source(s1)) == ("后台任务去重", TITLE_SOURCE_LLM)
            checks.append(("plugin replacement path", "OK"))

            # 手改标题为 user 来源；自动写入不会覆盖它。
            assert db.get_session_title_source(s2) == TITLE_SOURCE_USER
            assert db.set_auto_title(s2, "自动标题", source=TITLE_SOURCE_LLM) is False
            assert db.get_session_title(s2) == "用户手改标题"
            checks.append(("user provenance", "OK"))

            # 被其他会话占用的标题：活跃会话持有时宿主抛出插件能识别的冲突；已结束的空会话持有时宿主会让出标题，插件靠 title_holder 先行发现并改用后缀。
            assert db.title_holder("用户手改标题", excluding=s1) == s2
            try:
                db.set_auto_title("lineage-tip", "用户手改标题", source=TITLE_SOURCE_LLM)
            except Exception as exc:
                assert _is_title_conflict(exc, "用户手改标题"), repr(exc)
            else:
                raise AssertionError("duplicate title did not raise")
            assert db.title_holder("被占用的标题", excluding=s1) == "ghost"
            assert db.title_holder("后台任务去重", excluding=s1) is None
            checks.append(("title holder and conflict", "OK"))

            # 对话读取：DISPLAY 视图取回压缩前的原始轮次，撤回轮次永不出现，include_ancestors 读取压缩谱系。
            def contents(sid, **kwargs):
                return [m.get("content") for m in db.get_messages_as_conversation(sid, **kwargs)]

            assert contents(s1) == ["[摘要] 已排查重复触发", "ok", "压缩后的继续消息"]
            assert contents(s1, include_ancestors=True, include_compacted=True) == [
                "帮我排查后台任务重复触发", "已定位到重复事件", "[摘要] 已排查重复触发", "ok", "压缩后的继续消息",
            ]
            assert contents(s2, include_ancestors=True, include_compacted=True) == ["请保留我的手改标题", "好的"]
            assert contents("lineage-tip") == ["压缩后的请求"]
            assert contents("lineage-tip", include_ancestors=True, include_compacted=True) == ["压缩前的请求", "压缩后的请求"]
            checks.append(("conversation read", "OK"))

            by_start = [row["id"] for row in db.list_sessions_rich(limit=10)]
            by_activity = [row["id"] for row in db.list_sessions_rich(limit=10, order_by_last_active=True)]
            assert set(by_start) == {s1, s2, "lineage-tip", "ghost"}, by_start
            assert by_start[-1] == s1 and by_activity[0] == s1, (by_start, by_activity)
            assert [row["id"] for row in db.list_sessions_rich(limit=10, min_message_count=3)] == [s1]
            assert len(db.list_sessions_rich(limit=-1)) == len(by_start)
            checks.append(("rich session listing", "OK"))

            assert db.resolve_session_id(s1) == s1
            assert db.resolve_session_id("lineage-t") == "lineage-tip"
            assert db.resolve_session_id("sess-000") is None
            checks.append(("session id resolution", "OK"))

            db.record_auxiliary_usage(s1, "hermes_auto_titler", model="probe-model", input_tokens=12, output_tokens=3, estimated_cost_usd=0.5)
            from persistence_assembly import open_persistence, profile_scope

            with open_persistence(profile_scope(), eager=True) as store:
                usage = store.usage.auxiliary_usage_by_task(s1)["hermes_auto_titler"]
            assert (usage.api_call_count, usage.tokens.input_tokens, usage.tokens.output_tokens, usage.estimated_cost_usd) == (1, 12, 3, 0.5)
            checks.append(("auxiliary usage recording", "OK"))
        finally:
            db.close()

    for check in checks:
        print(" | ".join(check))
    print("temporary database cleanup | OK")


if __name__ == "__main__":
    main()
