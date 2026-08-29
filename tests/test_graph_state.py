"""Graph state must not leak between messages.

LangGraph state is cumulative per thread, and thread_id here is the
conversation, so anything a node returns survives into the next message
unless something clears it. That produced a real bug in live use: after
confirming a calendar event, the next request -- for an email -- had the
finished event re-proposed, and saying yes created a duplicate.
"""

import app.graph as graph_module
from app.graph import _clear_pending, _needs_approval, agent_node


async def test_agent_node_clears_pending_after_a_plain_reply(monkeypatch):
    """The actual bug, at the level it happened.

    Message 1 proposes a calendar event and it gets confirmed. Message 2
    asks for something unrelated, and the model just replies -- so
    agent_node must wipe pending_tool. Testing _clear_pending alone does
    not catch this: the first version of this file passed happily with
    agent_node broken.
    """
    async def only_talks(*args, **kwargs):
        return {"reply": "sure, I'll write that email"}

    monkeypatch.setattr(graph_module, "run_agent", only_talks)

    leftover = {
        "conversation_id": "c",
        "message": "write an email to sam@example.com",
        "pending_tool": "create_calendar_event",          # from message 1
        "pending_input": {"summary": "Meeting with HH Staff"},
    }
    result = await agent_node(leftover)

    merged = {**leftover, **result}
    assert _needs_approval(merged) == "done", (
        "a finished action was re-proposed on the next message"
    )


async def test_agent_node_still_proposes_a_real_write(monkeypatch):
    """The clear must not swallow genuine writes."""
    async def wants_to_send(*args, **kwargs):
        return {"pending": {"name": "send_email", "input": {"to": "a@b.c"}}}

    monkeypatch.setattr(graph_module, "run_agent", wants_to_send)

    result = await agent_node({"conversation_id": "c", "message": "email a@b.c"})

    assert _needs_approval(result) == "approve"


def test_reply_clears_the_pending_write():
    """The fix. A plain reply must reset pending_tool, not just omit it --
    omitting merges nothing and the stale value survives."""
    result = _clear_pending("here's your answer")

    assert result["pending_tool"] is None
    assert result["pending_input"] is None
    assert "pending_tool" in result, "omitting the key leaves the old value in place"


def test_stale_pending_would_reroute_to_approval():
    """Why it mattered: the branch reads pending_tool off the merged
    state, so a leftover value sends a finished action back to the gate."""
    leftover = {
        "reply": "sure, I'll email them",
        "pending_tool": "create_calendar_event",   # from a previous message
        "pending_input": {"summary": "Meeting with HH Staff"},
    }
    assert _needs_approval(leftover) == "approve", "this is the bug"

    cleaned = {**leftover, **_clear_pending("sure, I'll email them")}
    assert _needs_approval(cleaned) == "done", "cleared state must not re-propose"


def test_a_real_pending_write_still_reaches_the_gate():
    """The clear must not go so far that genuine writes skip approval."""
    proposing = {"pending_tool": "send_email", "pending_input": {"to": "a@b.c"}}
    assert _needs_approval(proposing) == "approve"


def test_absent_pending_is_treated_as_done():
    assert _needs_approval({"reply": "hello"}) == "done"


async def test_a_text_message_does_not_erase_the_forwarded_file():
    """The save_to_drive bug, at the level it happened.

    Cumulative state means every key a turn supplies overwrites what was
    there. run_graph used to pass `document or {}` on every message, so
    the moment the user typed anything after forwarding a file, the media
    id was replaced with an empty dict. Approval then failed with
    KeyError: 'media_id', reported to the user as a Google rejection for
    an upload that was never attempted.

    A turn carrying no file must leave the key alone.
    """
    from app.graph import _turn

    with_file = _turn(
        "998900000000",
        "I just sent you a file called report.pdf",
        [],
        {"media_id": "123", "filename": "report.pdf", "mime_type": "application/pdf"},
    )
    assert with_file["document"]["media_id"] == "123"

    # The follow-up: "save that to my drive", typed as a separate message.
    without_file = _turn("998900000000", "save that to my drive", [], None)
    assert "document" not in without_file, (
        "a text-only turn must not supply `document`, or it overwrites the "
        "reference from the turn the file arrived on"
    )
