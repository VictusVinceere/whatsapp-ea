"""The approval prompt must never re-enter the model's history.

The prompt is the system asking, not the assistant talking. Stored as an
assistant turn it comes back on the next call as a worked example, and
the model starts writing the prompt as plain text instead of calling the
tool. Plain text raises no interrupt, so the graph stays idle -- and the
user's "yes" is then read as a fresh request rather than an answer.

Observed live on 2026-08-29: forwarding a PDF produced the Drive prompt
twice. The logs show the first one had no tool_write_proposed and no
graph_interrupted behind it. It was imitation, copied from three real
prompts sitting in the replayed history.
"""

import app.graph as graph_module
import app.webhook as webhook
from app.graph import GraphTurn

PROMPT = 'Save "report.pdf" to your Google Drive?\n\nReply YES or NO.'


class _IdleState:
    """What aget_state returns when nothing is in flight."""

    next = ()
    tasks = ()


class _FakeGraph:
    def __init__(self, result):
        self._result = result

    async def aget_state(self, config):
        return _IdleState()

    async def ainvoke(self, payload, config):
        return self._result


class _Interrupt:
    value = PROMPT


async def test_a_parked_run_tells_the_caller_not_to_store_it(monkeypatch):
    monkeypatch.setattr(
        graph_module, "_graph", _FakeGraph({"__interrupt__": [_Interrupt()]})
    )

    turn = await graph_module.run_graph("998900000000", "save that to my drive")

    assert turn.awaiting_approval is True
    assert turn.reply == PROMPT


async def test_an_ordinary_reply_is_flagged_storable(monkeypatch):
    monkeypatch.setattr(
        graph_module, "_graph", _FakeGraph({"reply": "Saved it to your Drive."})
    )

    turn = await graph_module.run_graph("998900000000", "yes")

    assert turn.awaiting_approval is False
    assert turn.reply == "Saved it to your Drive."


def _stub_webhook(monkeypatch, turn):
    """Everything process_message touches, minus the network and the db."""
    stored: list[tuple[str, str]] = []
    sent: list[str] = []

    async def recent_messages(tenant, conversation):
        return []

    async def save_message(tenant, conversation, role, content, wa_message_id=None):
        stored.append((role, content))
        return True

    async def run_graph(*args, **kwargs):
        return turn

    async def send_text_message(to, body):
        sent.append(body)

    monkeypatch.setattr(webhook, "recent_messages", recent_messages)
    monkeypatch.setattr(webhook, "save_message", save_message)
    monkeypatch.setattr(webhook, "run_graph", run_graph)
    monkeypatch.setattr(webhook, "send_text_message", send_text_message)
    return stored, sent


MESSAGE = {
    "type": "text",
    "text": "save that to my drive",
    "message_id": "wamid.test-approval",
    "from": "998900000000",
}


async def test_the_prompt_reaches_the_user_but_not_the_history(monkeypatch):
    """The bug, at the level it happened."""
    stored, sent = _stub_webhook(monkeypatch, GraphTurn(PROMPT, awaiting_approval=True))

    await webhook.process_message(dict(MESSAGE))

    assert sent == [PROMPT], "the user still has to see the question"
    assert [role for role, _ in stored] == ["user"], (
        "the approval prompt was stored as an assistant turn -- replayed as "
        "history it teaches the model to write the prompt instead of calling "
        "the tool, and a text prompt parks no interrupt"
    )


async def test_a_real_answer_is_still_stored(monkeypatch):
    """The skip must not go so far that ordinary replies stop persisting;
    without them the assistant loses its memory of the conversation."""
    stored, sent = _stub_webhook(
        monkeypatch, GraphTurn("Saved it to your Drive.", awaiting_approval=False)
    )

    await webhook.process_message(dict(MESSAGE))

    assert sent == ["Saved it to your Drive."]
    assert ("assistant", "Saved it to your Drive.") in stored
