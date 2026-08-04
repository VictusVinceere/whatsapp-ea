"""Shared fixtures.

These tests hit a real Postgres rather than mocking it, because every
invariant worth protecting here is enforced *by the database* -- a UNIQUE
constraint, a conditional UPDATE, a COALESCE. Mocking the database would
mock away the thing under test.

Requires the dev container running:
    docker start whatsapp-ea-db
    uv run python -m app.db
"""

import uuid

import pytest
from sqlalchemy import text

from app.db import async_session, init_schema


@pytest.fixture(scope="session", autouse=True)
async def schema():
    """Make sure the tables exist before anything runs."""
    await init_schema()


@pytest.fixture
def tenant_id() -> str:
    """A fresh tenant per test.

    Tests that share a tenant see each other's rows and fail in confusing
    orders. A random uuid per test is cheaper than cleaning up, and it
    doubles as a check that nothing is hardcoding DEFAULT_TENANT_ID.
    """
    return str(uuid.uuid4())


@pytest.fixture
def conversation_id() -> str:
    return f"test-{uuid.uuid4().hex[:12]}"


@pytest.fixture
def wa_message_id() -> str:
    """A fresh WhatsApp message id per test.

    wa_message_id is UNIQUE across the whole table, not per tenant, so a
    hardcoded value passes on a clean database and fails on every run
    after -- the row from the first run is still there. Anything a test
    writes into a globally-unique column has to be generated per run.
    """
    return f"wamid.{uuid.uuid4().hex}"


@pytest.fixture
async def db():
    """A session for tests that need to poke at rows directly."""
    async with async_session() as session:
        yield session


async def insert_pending_action(conversation_id: str) -> str:
    """A pending_actions row, returned by id. Used by the race tests."""
    async with async_session() as session:
        result = await session.execute(
            text(
                "INSERT INTO pending_actions "
                "(conversation_id, agent, action_type, payload) "
                "VALUES (:c, 'calendar', 'create_event', '{}'::jsonb) "
                "RETURNING id"
            ),
            {"c": conversation_id},
        )
        action_id = result.scalar_one()
        await session.commit()
    return str(action_id)
