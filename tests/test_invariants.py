"""The rules the app breaks silently when they break.

Each test here corresponds to a bug that produced no error message: a
duplicate row, a wiped refresh token, a leaked document, a lost write.
Those are the ones worth a test, because nothing else will tell you.
"""

import asyncio

import pytest
from sqlalchemy import text

from app.db import (
    CONFIRM_ACTION_SQL,
    get_google_tokens,
    recent_messages,
    save_google_tokens,
    save_message,
)
from app.oauth import _make_state, _read_state
from tests.conftest import insert_pending_action


# --------------------------------------------------------------------------
# Meta redelivers every webhook. Without the UNIQUE constraint on
# wa_message_id the bot answered every message twice.
# --------------------------------------------------------------------------

async def test_duplicate_message_is_rejected(tenant_id, conversation_id, wa_message_id):
    first = await save_message(tenant_id, conversation_id, "user", "hi", wa_message_id)
    second = await save_message(tenant_id, conversation_id, "user", "hi", wa_message_id)

    assert first is True, "first insert should be stored"
    assert second is False, "redelivery must not create a second row"


async def test_assistant_replies_never_collide(tenant_id, conversation_id):
    """Our own replies carry no wa_message_id. NULLs must not conflict, or
    the bot could only ever reply once per conversation."""
    a = await save_message(tenant_id, conversation_id, "assistant", "one")
    b = await save_message(tenant_id, conversation_id, "assistant", "two")

    assert (a, b) == (True, True)


# --------------------------------------------------------------------------
# Conversation history is replayed to Claude, so order and isolation matter.
# --------------------------------------------------------------------------

async def test_history_is_oldest_first(tenant_id, conversation_id):
    """Selected newest-first for the LIMIT, then reversed. Handing Claude
    newest-first tells it the conversation ran backwards."""
    for i in range(3):
        await save_message(tenant_id, conversation_id, "user", f"message {i}")

    history = await recent_messages(tenant_id, conversation_id)

    assert [h["content"] for h in history] == ["message 0", "message 1", "message 2"]


async def test_history_does_not_leak_across_tenants(conversation_id):
    """Same conversation id, different tenants. A missing tenant filter
    would return the other customer's messages."""
    import uuid

    tenant_a, tenant_b = str(uuid.uuid4()), str(uuid.uuid4())
    await save_message(tenant_a, conversation_id, "user", "tenant A secret")
    await save_message(tenant_b, conversation_id, "user", "tenant B secret")

    history_a = await recent_messages(tenant_a, conversation_id)

    assert [h["content"] for h in history_a] == ["tenant A secret"]


# --------------------------------------------------------------------------
# Google returns a refresh token only on the FIRST authorization. Losing it
# means the user's calendar silently disconnects an hour later.
# --------------------------------------------------------------------------

async def test_reauth_preserves_refresh_token(tenant_id, conversation_id):
    await save_google_tokens(tenant_id, conversation_id, "access-1", "refresh-1", 3600, "s")
    # Second authorization: Google omits refresh_token, so we pass None.
    await save_google_tokens(tenant_id, conversation_id, "access-2", None, 3600, "s")

    tokens = await get_google_tokens(tenant_id, conversation_id)

    assert tokens["access_token"] == "access-2", "new access token should win"
    assert tokens["refresh_token"] == "refresh-1", "COALESCE must keep the stored one"


async def test_unconnected_conversation_returns_none(tenant_id, conversation_id):
    assert await get_google_tokens(tenant_id, conversation_id) is None


# --------------------------------------------------------------------------
# The approval gate. Two "yes" replies arriving together must produce one
# calendar event, not two.
# --------------------------------------------------------------------------

async def test_concurrent_confirms_produce_one_winner(conversation_id):
    action_id = await insert_pending_action(conversation_id)

    async def confirm() -> bool:
        from app.db import async_session

        async with async_session() as session:
            result = await session.execute(
                text(CONFIRM_ACTION_SQL), {"action_id": action_id}
            )
            won = result.first() is not None
            await session.commit()
        return won

    results = await asyncio.gather(confirm(), confirm())

    assert sum(results) == 1, f"exactly one confirm must win, got {results}"


async def test_confirming_twice_in_sequence_is_a_noop(conversation_id):
    """Not just a race -- a user tapping "yes" twice minutes apart must not
    fire the action again."""
    from app.db import async_session

    action_id = await insert_pending_action(conversation_id)

    async def confirm() -> bool:
        async with async_session() as session:
            result = await session.execute(
                text(CONFIRM_ACTION_SQL), {"action_id": action_id}
            )
            won = result.first() is not None
            await session.commit()
        return won

    assert await confirm() is True
    assert await confirm() is False


# --------------------------------------------------------------------------
# The OAuth `state` parameter is the only thing tying a Google callback back
# to a conversation, and /oauth/callback is a public URL.
# --------------------------------------------------------------------------

def test_state_round_trips():
    assert _read_state(_make_state("15550001234")) == "15550001234"


@pytest.mark.parametrize(
    "forged",
    [
        "15550001234",                    # unsigned
        "15550001234.deadbeef",           # wrong signature
        "",                                # empty
        ".abc",                            # no conversation id
    ],
)
def test_forged_state_is_rejected(forged):
    assert _read_state(forged) is None


def test_signature_cannot_be_reused_for_another_number():
    """The signature covers the number itself, so lifting a valid signature
    onto someone else's id must fail -- otherwise an attacker attaches
    their Google account to another person's conversation."""
    valid = _make_state("15550001234")
    signature = valid.split(".", 1)[1]

    assert _read_state(f"111111111111.{signature}") is None
