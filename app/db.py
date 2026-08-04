"""
Async Postgres access. Uses asyncpg -- never swap this for a blocking
driver (psycopg2) or you'll freeze the event loop for every in-flight
request, not just the one making the call.

Build order step 6: get pending_actions working with a single fake
action before wiring up real agents.
"""

from datetime import datetime, timedelta, timezone

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings

engine = create_async_engine(settings.database_url, echo=False)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

CREATE_PENDING_ACTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS pending_actions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id TEXT NOT NULL,
    agent TEXT NOT NULL,              -- 'email' | 'calendar' | 'drive'
    action_type TEXT NOT NULL,        -- e.g. 'send_email', 'create_event'
    payload JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',  -- 'pending' | 'confirmed' | 'rejected'
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

# Conversation history, replayed to Claude so the assistant remembers.
#
# id is an identity column rather than a UUID because this table needs a
# reliable sort order: now() is transaction time, so two rows written in
# one transaction share a created_at and ORDER BY created_at is undefined
# between them. An identity column is strictly increasing.
#
# tenant_id  = which customer's assistant (one row per customer of yours)
# conversation_id = the end-user talking to it, i.e. their WhatsApp number
CREATE_MESSAGES_TABLE = """
CREATE TABLE IF NOT EXISTS messages (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tenant_id       UUID NOT NULL,
    conversation_id TEXT NOT NULL,
    -- Sent straight back to the Anthropic API, so these are its values,
    -- not ours. CHECK over ENUM: widening a CHECK is a one-line change.
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT NOT NULL,
    -- WhatsApp's wamid; NULL for our own replies. UNIQUE makes this the
    -- durable replacement for the in-process dedup set in webhook.py.
    wa_message_id   TEXT UNIQUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

# Matches the only read this table serves: newest N for one conversation.
CREATE_MESSAGES_INDEX = """
CREATE INDEX IF NOT EXISTS messages_lookup_idx
    ON messages (tenant_id, conversation_id, id DESC);
"""


CREATE_GOOGLE_TOKENS_TABLE = """
CREATE TABLE IF NOT EXISTS google_tokens (
    tenant_id UUID NOT NULL,
    conversation_id TEXT NOT NULL,
    access_token TEXT NOT NULL,
    refresh_token TEXT,
    expires_at TIMESTAMPTZ NOT NULL,
    scope TEXT NOT NULL,
    PRIMARY KEY (tenant_id, conversation_id)
);
"""


# pgvector is an extension, not a separate database -- the same Postgres
# holds your messages and your embeddings, so a similarity search can join
# against normal tables. Must run before any table declares a vector column.
CREATE_VECTOR_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector;"

# One row per chunk of a document, not one per document. A whole file is
# too big to hand to Claude and too coarse to match against -- a question
# about one paragraph shouldn't drag in forty unrelated pages.
#
# vector(384) matches BAAI/bge-small-en-v1.5 exactly. The dimension is
# fixed at table-creation time, so switching embedding models later means
# a migration and re-embedding everything.
CREATE_DOCUMENT_CHUNKS_TABLE = """
CREATE TABLE IF NOT EXISTS document_chunks (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tenant_id    UUID NOT NULL,
    -- Where this came from, so a chunk can be cited and a changed file
    -- can have its old chunks deleted before re-ingesting.
    source_id    TEXT NOT NULL,
    source_name  TEXT NOT NULL,
    chunk_index  INT  NOT NULL,
    content      TEXT NOT NULL,
    embedding    vector(384) NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, source_id, chunk_index)
);
"""

# HNSW: approximate nearest neighbour. Exact search reads every row, which
# is fine at 100 chunks and hopeless at 100k. vector_cosine_ops must match
# the operator used at query time (<=>), or the index is silently ignored
# and you get a sequential scan with no error to tell you.
CREATE_DOCUMENT_CHUNKS_INDEX = """
CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
    ON document_chunks USING hnsw (embedding vector_cosine_ops);
"""


# Applied in order by init_schema(). Append new statements here -- this
# tuple is the whole schema, so nothing exists that isn't in it.
#
# Every statement must be idempotent (IF NOT EXISTS), because this runs
# again on every fresh checkout and after every schema addition. Note the
# ceiling: this creates, it never alters. The first time you need to
# change an existing column, switch to Alembic rather than editing these.
SCHEMA_STATEMENTS = (
    # Extension first -- document_chunks can't declare vector(384) until
    # the type exists.
    CREATE_VECTOR_EXTENSION,
    CREATE_PENDING_ACTIONS_TABLE,
    CREATE_MESSAGES_TABLE,
    CREATE_MESSAGES_INDEX,
    CREATE_GOOGLE_TOKENS_TABLE,
    CREATE_DOCUMENT_CHUNKS_TABLE,
    CREATE_DOCUMENT_CHUNKS_INDEX,
)


DEFAULT_TENANT_ID = "00000000-0000-0000-0000-000000000001"


async def init_schema() -> None:
    """Create every table and index. Safe to re-run."""
    async with engine.begin() as conn:
        for statement in SCHEMA_STATEMENTS:
            await conn.execute(text(statement))


async def save_message(
    tenant_id: str,
    conversation_id: str,
    role: str,
    content: str,
    wa_message_id: str | None = None,
) -> bool:
    async with async_session() as session:
        result = await session.execute(
            text(
                "INSERT INTO messages(tenant_id, conversation_id, role, content, wa_message_id) VALUES(:tenant_id, :conversation_id, :role, :content, :wa_message_id) ON CONFLICT (wa_message_id) DO NOTHING"
            ),
            {
                "tenant_id": tenant_id,
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
                "wa_message_id": wa_message_id,
            },
        )
        await session.commit()

        return result.rowcount > 0


async def get_google_tokens(
    tenant_id: str,
    conversation_id: str,
) -> dict | None:
    async with async_session() as session:
        result = await session.execute(
            text(
                "SELECT access_token, refresh_token, expires_at, scope FROM google_tokens WHERE tenant_id = :tenant_id AND conversation_id = :conversation_id"
            ),
            {
                "tenant_id": tenant_id,
                "conversation_id": conversation_id,
            },
        )
        row = result.mappings().first()

    if row is None:
        return None

    return {
        "access_token": row["access_token"],
        "refresh_token": row["refresh_token"],
        "expires_at": row["expires_at"],
        "scope": row["scope"],
    }


async def save_google_tokens(
    tenant_id: str,
    conversation_id: str,
    access_token: str,
    # None on re-authorization -- Google only sends a refresh token the
    # first time an account grants access.
    refresh_token: str | None,
    # Seconds, exactly as Google returns it. Converted below.
    expires_in: int,
    scope: str,
) -> None:
    # The column is TIMESTAMPTZ, an absolute moment; Google gives a
    # duration. Convert here so callers pass the API response through
    # unchanged, and "is this still valid?" is a plain > comparison.
    # timezone-aware: a naive datetime into TIMESTAMPTZ is an hour-off
    # bug that only shows up later.
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)

    async with async_session() as session:
        await session.execute(
            text(
                "INSERT INTO google_tokens(tenant_id, conversation_id, access_token, refresh_token, expires_at, scope) VALUES(:tenant_id, :conversation_id, :access_token, :refresh_token, :expires_at, :scope) ON CONFLICT (tenant_id, conversation_id) DO UPDATE SET access_token = EXCLUDED.access_token, refresh_token = COALESCE(EXCLUDED.refresh_token, google_tokens.refresh_token), expires_at = EXCLUDED.expires_at, scope = EXCLUDED.scope"
            ),
            {
                "tenant_id": tenant_id,
                "conversation_id": conversation_id,
                "access_token": access_token,
                "refresh_token": refresh_token,
                "expires_at": expires_at,
                "scope": scope,
            },
        )
        await session.commit()


async def recent_messages(
    tenant_id: str,
    conversation_id: str,
    limit: int = 10,
) -> list[dict]:
    """The last `limit` turns of one conversation, oldest first.

    Selected newest-first so LIMIT keeps the *most recent* turns, then
    reversed on the way out -- the Anthropic API reads `messages` in
    chronological order, so handing it newest-first would tell Claude the
    conversation ran backwards.

    Both ids are in the WHERE clause, not just conversation_id. Filtering
    by tenant is what stops one customer's assistant reading another's
    history; the index is ordered to match.
    """
    async with async_session() as session:
        result = await session.execute(
            text(
                "SELECT role, content FROM messages "
                "WHERE tenant_id = :tenant_id "
                "  AND conversation_id = :conversation_id "
                "ORDER BY id DESC "
                "LIMIT :limit"
            ),
            {
                "tenant_id": tenant_id,
                "conversation_id": conversation_id,
                "limit": limit,
            },
        )
        rows = result.mappings().all()

    # No commit -- nothing was written.
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


# The conditional update that prevents a double-confirm race condition --
# same pattern as the DynamoDB attribute_not_exists(slot_id) fix, applied
# to Postgres. Zero rows update if status isn't still 'pending'.
CONFIRM_ACTION_SQL = """
UPDATE pending_actions
SET status = 'confirmed'
WHERE id = :action_id AND status = 'pending'
RETURNING id;
"""


if __name__ == "__main__":
    # uv run python -m app.db
    import asyncio

    asyncio.run(init_schema())
    print(f"schema applied ({len(SCHEMA_STATEMENTS)} statements)")
