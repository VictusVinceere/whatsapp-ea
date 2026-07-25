"""
Async Postgres access. Uses asyncpg -- never swap this for a blocking
driver (psycopg2) or you'll freeze the event loop for every in-flight
request, not just the one making the call.

Build order step 6: get pending_actions working with a single fake
action before wiring up real agents.
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.config import settings

engine = create_async_engine(settings.database_url, echo=False)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Run this once against your local Postgres to create the table:
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

# The conditional update that prevents a double-confirm race condition --
# same pattern as the DynamoDB attribute_not_exists(slot_id) fix, applied
# to Postgres. Zero rows update if status isn't still 'pending'.
CONFIRM_ACTION_SQL = """
UPDATE pending_actions
SET status = 'confirmed'
WHERE id = :action_id AND status = 'pending'
RETURNING id;
"""
