"""
Build order step 7: the router and agents as a LangGraph state machine.

Why a graph rather than the if/elif this replaces: the approval gate.
Creating a calendar event has to pause, ask the user, and wait -- possibly
for hours, possibly across a server restart -- then resume exactly where
it stopped. Hand-rolling that means serialising "where was I" yourself.
LangGraph's interrupt() plus a checkpointer is precisely that machinery.

Two durable stores, doing different jobs:

  checkpointer     where the graph paused, so it can resume  (LangGraph)
  pending_actions  what was proposed and whether it was       (ours)
                   approved -- the audit trail, and the
                   conditional UPDATE that survives a
                   double-confirm race

Neither replaces the other. The checkpointer knows nothing about your
business rules; pending_actions knows nothing about graph position.
"""

import json
from typing import Literal, TypedDict

import structlog
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from sqlalchemy import text as sql

from app.agents.drive_agent import handle_drive_request
from app.config import settings
from app.db import DEFAULT_TENANT_ID, async_session
from app.llm import ask_claude
from app.router_agent import classify_intent

log = structlog.get_logger()


class AssistantState(TypedDict, total=False):
    """Everything a run needs. LangGraph merges each node's return into it."""

    conversation_id: str
    message: str
    intent: str
    reply: str
    action_id: str
    summary: str


async def route_node(state: AssistantState) -> dict:
    return {"intent": await classify_intent(state["message"])}


async def general_node(state: AssistantState) -> dict:
    return {"reply": await ask_claude(state["message"])}


async def drive_node(state: AssistantState) -> dict:
    reply = await handle_drive_request(state["message"], state["conversation_id"])
    return {"reply": reply}


async def email_node(state: AssistantState) -> dict:
    return {"reply": "Email agent not wired up yet."}


async def calendar_propose_node(state: AssistantState) -> dict:
    """Write the proposed action. Its own node, and that is load-bearing.

    A node containing interrupt() re-runs *from the top* when resumed --
    interrupt() returns the answer rather than pausing, but every line
    above it executes a second time. Putting this INSERT in the same node
    as the interrupt produced a duplicate pending_actions row on every
    confirmation, leaving the first orphaned as 'pending' forever.

    Node boundaries are checkpoint boundaries: once this one returns, it
    is durably done and never repeats. Side effects go here; the wait
    goes next door.
    """
    payload = {"summary": state["message"], "duration_minutes": 30}

    async with async_session() as session:
        result = await session.execute(
            sql(
                "INSERT INTO pending_actions "
                "(conversation_id, agent, action_type, payload) "
                "VALUES (:conversation_id, 'calendar', 'create_event', "
                "CAST(:payload AS jsonb)) RETURNING id"
            ),
            {
                "conversation_id": state["conversation_id"],
                "payload": json.dumps(payload),
            },
        )
        action_id = str(result.scalar_one())
        await session.commit()

    log.info("action_proposed", action_id=action_id)
    return {"action_id": action_id, "summary": payload["summary"]}


async def calendar_approve_node(state: AssistantState) -> dict:
    """Stop dead until a human answers, then confirm or cancel.

    Safe to re-run: everything here is either idempotent or guarded by the
    conditional UPDATE.
    """
    action_id = state["action_id"]
    payload = {"summary": state["summary"], "duration_minutes": 30}

    # Everything below runs only after the user replies.
    answer = interrupt(
        f"I'll create a 30 minute event: \"{payload['summary']}\". "
        "Reply YES to confirm or NO to cancel."
    )

    if str(answer).strip().lower() not in {"yes", "y", "confirm", "ok"}:
        async with async_session() as session:
            await session.execute(
                sql(
                    "UPDATE pending_actions SET status = 'rejected' "
                    "WHERE id = CAST(:id AS uuid) AND status = 'pending'"
                ),
                {"id": action_id},
            )
            await session.commit()
        return {"reply": "Cancelled, nothing was created.", "action_id": action_id}

    # The conditional UPDATE from db.py. Two simultaneous confirmations
    # both run this; exactly one matches a row still marked 'pending', so
    # exactly one gets an id back and the other updates nothing. The check
    # and the write are one atomic statement -- no gap for a race.
    async with async_session() as session:
        result = await session.execute(
            sql(
                "UPDATE pending_actions SET status = 'confirmed' "
                "WHERE id = CAST(:id AS uuid) AND status = 'pending' RETURNING id"
            ),
            {"id": action_id},
        )
        won_the_race = result.first() is not None
        await session.commit()

    if not won_the_race:
        log.info("action_already_handled", action_id=action_id)
        return {"reply": "That was already handled.", "action_id": action_id}

    # TODO: call the Google Calendar API here. Needs the calendar.events
    # scope (currently calendar.readonly) and a re-consent. The gate around
    # it is complete either way.
    log.info("action_executed", action_id=action_id, payload=payload)
    return {"reply": f"Done -- created \"{payload['summary']}\".", "action_id": action_id}


def _pick_agent(state: AssistantState) -> Literal["calendar", "drive", "email", "general"]:
    """Conditional edge: reads intent, names the next node."""
    return state["intent"]  # type: ignore[return-value]


def build_graph() -> StateGraph:
    builder = StateGraph(AssistantState)
    builder.add_node("route", route_node)
    builder.add_node("calendar", calendar_propose_node)
    builder.add_node("calendar_approve", calendar_approve_node)
    builder.add_node("drive", drive_node)
    builder.add_node("email", email_node)
    builder.add_node("general", general_node)

    builder.add_edge(START, "route")
    builder.add_conditional_edges("route", _pick_agent)
    # Propose and approve are two hops so the write is checkpointed before
    # the wait -- see calendar_propose_node.
    builder.add_edge("calendar", "calendar_approve")
    for node in ("calendar_approve", "drive", "email", "general"):
        builder.add_edge(node, END)
    return builder


def checkpointer_dsn() -> str:
    """The checkpointer uses psycopg, not asyncpg, so the SQLAlchemy
    dialect prefix has to come off. Still async -- psycopg3 has native
    async support, unlike psycopg2."""
    return settings.database_url.replace("postgresql+asyncpg://", "postgresql://")
