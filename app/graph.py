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
from datetime import datetime, timezone
from typing import Literal, TypedDict

import structlog
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from sqlalchemy import text as sql

from app.gmail_api import send_message
from app.calendar_api import calendar_timezone, insert_event
from app.config import settings
from app.db import DEFAULT_TENANT_ID, async_session
from app.google import valid_access_token
from app.llm import run_agent

log = structlog.get_logger()


class AssistantState(TypedDict, total=False):
    """Everything a run needs. LangGraph merges each node's return into it."""

    conversation_id: str
    message: str
    intent: str
    reply: str
    action_id: str
    summary: str
    history: list[dict]
    # "read" answers straight away; "write" goes through the approval gate.
    pending_tool: str
    pending_input: dict


async def agent_node(state: AssistantState) -> dict:
    """Claude with tools. Answers directly, or proposes a write.

    Replaces the router plus every triage and read node. Those existed to
    pick ONE destination, which is why "pull my last events and last
    email" -- a request for two things -- matched nothing and fell through
    to general chat. The model now calls whichever tools it needs.
    """
    outcome = await run_agent(
        state["message"],
        state["conversation_id"],
        history=state.get("history"),
    )

    if "pending" in outcome:
        pending = outcome["pending"]
        return {
            "pending_tool": pending["name"],
            "pending_input": pending["input"],
        }
    return {"reply": outcome["reply"]}


async def propose_node(state: AssistantState) -> dict:
    """Record the proposed write. Its own node because a node containing
    interrupt() re-runs from the top on resume -- see the docstring on
    approve_node."""
    agent = "calendar" if state["pending_tool"] == "create_calendar_event" else "email"

    async with async_session() as session:
        result = await session.execute(
            sql(
                "INSERT INTO pending_actions "
                "(conversation_id, agent, action_type, payload) "
                "VALUES (:conversation_id, :agent, :action_type, "
                "CAST(:payload AS jsonb)) RETURNING id"
            ),
            {
                "conversation_id": state["conversation_id"],
                "agent": agent,
                "action_type": state["pending_tool"],
                "payload": json.dumps(state["pending_input"]),
            },
        )
        action_id = str(result.scalar_one())
        await session.commit()

    log.info("action_proposed", action_id=action_id, tool=state["pending_tool"])
    return {"action_id": action_id}


def _describe_pending(tool: str, args: dict) -> str:
    """What the user is agreeing to.

    An email is quoted in full -- it cannot be recalled, so the user needs
    to approve the actual words. An event only needs its parsed time
    restated, which is enough to catch a misread date.
    """
    if tool == "send_email":
        return (
            f"Send this?\n\nTo: {args.get('to')}\n"
            f"Subject: {args.get('subject')}\n\n{args.get('body')}"
        )
    when = args.get("start", "?")
    try:
        when = datetime.fromisoformat(when).strftime("%a %d %b, %H:%M")
    except ValueError:
        pass
    return (
        f"Create \"{args.get('summary')}\" on {when} "
        f"for {args.get('duration_minutes')} minutes?"
    )


async def approve_node(state: AssistantState) -> dict:
    """Wait for a human, then execute exactly once.

    Safe to re-run: everything here is either idempotent or guarded by the
    conditional UPDATE, which is what makes two simultaneous confirmations
    produce one action rather than two.
    """
    action_id = state["action_id"]
    tool = state["pending_tool"]
    args = state["pending_input"]

    answer = interrupt(f"{_describe_pending(tool, args)}\n\nReply YES or NO.")

    if str(answer).strip().lower() not in {"yes", "y", "confirm", "ok", "send", "do it"}:
        async with async_session() as session:
            await session.execute(
                sql(
                    "UPDATE pending_actions SET status = 'rejected' "
                    "WHERE id = CAST(:id AS uuid) AND status = 'pending'"
                ),
                {"id": action_id},
            )
            await session.commit()
        return {"reply": "Cancelled, nothing was changed."}

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
        return {"reply": "That was already done."}

    token = await valid_access_token(DEFAULT_TENANT_ID, state["conversation_id"])
    if token is None:
        return {"reply": NOT_CONNECTED}

    try:
        if tool == "create_calendar_event":
            tz = await calendar_timezone(token)
            await insert_event(
                token,
                summary=args["summary"],
                start_iso=args["start"],
                duration_minutes=int(args["duration_minutes"]),
                tz=tz,
            )
            done = f"Done -- \"{args['summary']}\" is on your calendar."
        else:
            await send_message(
                token, to=args["to"], subject=args["subject"], body=args["body"]
            )
            done = f"Sent to {args['to']}."
    except Exception:
        # The row stays 'confirmed'. Resetting it would put the action back
        # up for a second approval, and exactly-one-winner is the whole
        # point of the conditional UPDATE.
        log.exception("action_failed", action_id=action_id, tool=tool)
        return {"reply": "I couldn't do that -- Google rejected it. Nothing changed."}

    log.info("action_executed", action_id=action_id, tool=tool)
    return {"reply": done}


def _needs_approval(state: AssistantState) -> Literal["approve", "done"]:
    return "approve" if state.get("pending_tool") else "done"


def build_graph() -> StateGraph:
    builder = StateGraph(AssistantState)
    builder.add_node("agent", agent_node)
    builder.add_node("propose", propose_node)
    builder.add_node("approve", approve_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent", _needs_approval, {"approve": "propose", "done": END}
    )
    # Two hops so the row is checkpointed before the wait.
    builder.add_edge("propose", "approve")
    builder.add_edge("approve", END)
    return builder


def checkpointer_dsn() -> str:
    """The checkpointer uses psycopg, not asyncpg, so the SQLAlchemy
    dialect prefix has to come off. Still async -- psycopg3 has native
    async support, unlike psycopg2."""
    return settings.database_url.replace("postgresql+asyncpg://", "postgresql://")


# Compiled once at startup by main.py's lifespan, because the checkpointer
# owns a connection pool that must outlive any single request.
_graph = None


def set_graph(compiled) -> None:
    global _graph
    _graph = compiled


async def run_graph(
    conversation_id: str,
    message: str,
    history: list[dict] | None = None,
) -> str:
    """One WhatsApp message through the graph. Returns what to reply.

    A conversation can be mid-approval, so the first job is deciding
    whether this message *starts* a run or *answers* one. `state.next`
    being non-empty means the graph is parked on an interrupt, and the
    message is the user's answer to whatever it asked.
    """
    if _graph is None:
        raise RuntimeError("graph not compiled -- is the lifespan handler running?")

    config = {"configurable": {"thread_id": conversation_id}}
    state = await _graph.aget_state(config)

    if state.next:
        log.info("graph_resuming", conversation_id=conversation_id, at=state.next)
        result = await _graph.ainvoke(Command(resume=message), config)
    else:
        result = await _graph.ainvoke(
            {
                "conversation_id": conversation_id,
                "message": message,
                "history": history or [],
            },
            config,
        )

    # An interrupted run has no "reply" -- what the user should see is the
    # question the interrupt raised ("Reply YES to confirm...").
    pending = result.get("__interrupt__")
    if pending:
        log.info("graph_interrupted", conversation_id=conversation_id)
        return pending[0].value

    return result["reply"]
