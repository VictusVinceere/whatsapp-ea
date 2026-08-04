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
from zoneinfo import ZoneInfo

import structlog
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from sqlalchemy import text as sql

from app.agents.drive_agent import handle_drive_request
from app.gmail_api import describe_messages, list_recent, send_message
from app.calendar_api import (
    calendar_timezone,
    describe_events,
    insert_event,
    list_events,
)
from app.config import settings
from app.db import DEFAULT_TENANT_ID, async_session
from app.google import valid_access_token
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
    history: list[dict]
    # "read" answers straight away; "write" goes through the approval gate.
    calendar_op: str
    event_start: str
    event_duration: int
    event_tz: str
    email_op: str
    email_to: str
    email_subject: str
    email_body: str


async def route_node(state: AssistantState) -> dict:
    return {"intent": await classify_intent(state["message"])}


async def general_node(state: AssistantState) -> dict:
    # Only this branch replays conversation history. The specialists answer
    # from their own source of truth, so chat history is noise and tokens.
    return {
        "reply": await ask_claude(state["message"], history=state.get("history"))
    }


async def drive_node(state: AssistantState) -> dict:
    reply = await handle_drive_request(state["message"], state["conversation_id"])
    return {"reply": reply}


DRAFT_PROMPT = """The user wants to send an email. Extract the details.

Reply with ONLY a JSON object, no prose, no code fence:
{"to": "<email address>", "subject": "<short subject>", "body": "<the message>"}

Write the body as a complete, polite email in the user's voice. If no
recipient address is given, use "" for `to` -- do not invent one."""


async def email_triage_node(state: AssistantState) -> dict:
    """Reading mail and sending it are different risks -- same split as
    calendar. Drafting happens here so a failed parse never leaves a
    half-written pending_actions row."""
    token = await valid_access_token(DEFAULT_TENANT_ID, state["conversation_id"])
    if token is None:
        return {"email_op": "unconnected"}

    verdict = await ask_claude(
        state["message"],
        system=(
            "Does this message ask to SEND an email, or only to READ or "
            "search existing mail? Reply with one word: send or read."
        ),
        max_tokens=5,
    )
    if "send" not in verdict.strip().lower():
        return {"email_op": "read"}

    raw = await ask_claude(state["message"], system=DRAFT_PROMPT, max_tokens=600)
    try:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        draft = json.loads(cleaned)
    except Exception:
        log.warning("email_parse_failed", raw=raw[:120])
        return {"email_op": "unparsed"}

    # A missing recipient is the common case -- "email Sarah about the
    # launch" has no address in it. Better to ask than to guess, and far
    # better than sending to an invented address.
    if not draft.get("to"):
        return {"email_op": "no_recipient"}

    return {
        "email_op": "send",
        "email_to": draft["to"],
        "email_subject": draft.get("subject") or "(no subject)",
        "email_body": draft.get("body") or "",
    }


async def email_read_node(state: AssistantState) -> dict:
    token = await valid_access_token(DEFAULT_TENANT_ID, state["conversation_id"])
    messages = await list_recent(token, max_results=6)

    reply = await ask_claude(
        f"Inbox:\n{describe_messages(messages)}\n\nQuestion: {state['message']}",
        system=(
            "Answer from the inbox listing. Brief and conversational -- "
            "this is WhatsApp, not a report."
        ),
    )
    return {"reply": reply}


async def email_unavailable_node(state: AssistantState) -> dict:
    reasons = {
        "unconnected": NOT_CONNECTED,
        "no_recipient": "Who should I send it to? I need an email address.",
        "unparsed": "I couldn't work out what to send -- try \"email sam@x.com about the launch\".",
    }
    return {"reply": reasons.get(state.get("email_op"), NOT_CONNECTED)}


async def email_propose_node(state: AssistantState) -> dict:
    """Store the draft. Separate node from the wait, for the same reason
    calendar_propose is -- a node holding interrupt() re-runs on resume."""
    payload = {
        "to": state["email_to"],
        "subject": state["email_subject"],
        "body": state["email_body"],
    }

    async with async_session() as session:
        result = await session.execute(
            sql(
                "INSERT INTO pending_actions "
                "(conversation_id, agent, action_type, payload) "
                "VALUES (:conversation_id, 'email', 'send_email', "
                "CAST(:payload AS jsonb)) RETURNING id"
            ),
            {
                "conversation_id": state["conversation_id"],
                "payload": json.dumps(payload),
            },
        )
        action_id = str(result.scalar_one())
        await session.commit()

    log.info("action_proposed", action_id=action_id, agent="email")
    return {"action_id": action_id}


async def email_approve_node(state: AssistantState) -> dict:
    """Show the whole draft before sending. An email is unrecallable, so
    the confirmation quotes it in full rather than summarising -- the
    user should approve the actual text, not a description of it."""
    action_id = state["action_id"]

    answer = interrupt(
        f"Send this?\n\nTo: {state['email_to']}\n"
        f"Subject: {state['email_subject']}\n\n{state['email_body']}\n\n"
        "Reply YES or NO."
    )

    if str(answer).strip().lower() not in {"yes", "y", "confirm", "ok", "send"}:
        async with async_session() as session:
            await session.execute(
                sql(
                    "UPDATE pending_actions SET status = 'rejected' "
                    "WHERE id = CAST(:id AS uuid) AND status = 'pending'"
                ),
                {"id": action_id},
            )
            await session.commit()
        return {"reply": "Cancelled, nothing was sent."}

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
        return {"reply": "That was already sent."}

    token = await valid_access_token(DEFAULT_TENANT_ID, state["conversation_id"])
    if token is None:
        return {"reply": NOT_CONNECTED}

    try:
        await send_message(
            token,
            to=state["email_to"],
            subject=state["email_subject"],
            body=state["email_body"],
        )
    except Exception:
        log.exception("email_send_failed", action_id=action_id)
        return {"reply": "I couldn't send that -- Gmail rejected it. Nothing was sent."}

    log.info("action_executed", action_id=action_id, agent="email")
    return {"reply": f"Sent to {state['email_to']}."}


NOT_CONNECTED = (
    "I can't reach your calendar yet -- connect your Google account first."
)

PARSE_PROMPT = """Extract calendar event details from the user's message.

Right now it is {now} in timezone {tz}.

Reply with ONLY a JSON object, no prose, no code fence:
{{"summary": "<short title>", "start": "<YYYY-MM-DDTHH:MM:SS>", "duration_minutes": <int>}}

Resolve relative times ("tomorrow", "next Tuesday at 3") against the
current time above. Default to 30 minutes when no duration is given, and
to 09:00 when a date is given with no time."""


async def calendar_triage_node(state: AssistantState) -> dict:
    """Reading a calendar and changing one are different risks.

    "What's on Thursday?" should answer immediately. "Book a meeting"
    must not happen until a human confirms. This node decides which, and
    for writes also parses the request into structured fields -- doing
    that here keeps the propose node free of anything that could fail
    after the row is written.
    """
    token = await valid_access_token(DEFAULT_TENANT_ID, state["conversation_id"])
    if token is None:
        return {"calendar_op": "unconnected"}

    verdict = await ask_claude(
        state["message"],
        system=(
            "Does this message ask to CREATE or CHANGE a calendar event, or "
            "only to READ the calendar? Reply with one word: write or read."
        ),
        max_tokens=5,
    )
    operation = "write" if "write" in verdict.strip().lower() else "read"

    if operation == "read":
        return {"calendar_op": "read"}

    tz = await calendar_timezone(token)
    now = datetime.now(ZoneInfo(tz)).strftime("%Y-%m-%d %H:%M (%A)")
    raw = await ask_claude(
        state["message"],
        system=PARSE_PROMPT.format(now=now, tz=tz),
        max_tokens=200,
    )

    try:
        # Models wrap JSON in fences however firmly you ask them not to.
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        parsed = json.loads(cleaned)
        start = parsed["start"]
        # Fail now if it isn't a real datetime, not later inside insert_event
        # where the pending_actions row already exists.
        datetime.fromisoformat(start)
    except Exception:
        log.warning("calendar_parse_failed", raw=raw[:120])
        return {"calendar_op": "unparsed"}

    return {
        "calendar_op": "write",
        "summary": parsed.get("summary") or state["message"],
        "event_start": start,
        "event_duration": int(parsed.get("duration_minutes") or 30),
        "event_tz": tz,
    }


async def calendar_read_node(state: AssistantState) -> dict:
    """Read-only, so no gate -- nothing here changes the world."""
    token = await valid_access_token(DEFAULT_TENANT_ID, state["conversation_id"])
    events = await list_events(token, max_results=8)

    reply = await ask_claude(
        f"Calendar:\n{describe_events(events)}\n\nQuestion: {state['message']}",
        system=(
            "Answer the question from the calendar listing. Be brief and "
            "conversational -- this is WhatsApp. Today's date is "
            f"{datetime.now(timezone.utc).date()}."
        ),
    )
    return {"reply": reply}


async def calendar_unavailable_node(state: AssistantState) -> dict:
    if state.get("calendar_op") == "unparsed":
        return {"reply": "I couldn't work out the date and time -- try e.g. \"book a review tomorrow at 3pm\"."}
    return {"reply": NOT_CONNECTED}


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
    payload = {
        "summary": state["summary"],
        "start": state["event_start"],
        "duration_minutes": state["event_duration"],
        "timezone": state["event_tz"],
    }

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

    log.info("action_proposed", action_id=action_id, start=payload["start"])
    return {"action_id": action_id}


async def calendar_approve_node(state: AssistantState) -> dict:
    """Stop dead until a human answers, then confirm or cancel.

    Safe to re-run: everything here is either idempotent or guarded by the
    conditional UPDATE.
    """
    action_id = state["action_id"]
    when = datetime.fromisoformat(state["event_start"]).strftime("%a %d %b, %H:%M")

    # Everything below runs only after the user replies. The question
    # states the parsed time back, so a misread date is caught by the
    # human rather than landing in the calendar.
    answer = interrupt(
        f"Create \"{state['summary']}\" on {when} "
        f"for {state['event_duration']} minutes? Reply YES or NO."
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

    # Past the gate: a human said yes and this is the only confirmation
    # that will ever win. Everything from here has real-world effect.
    token = await valid_access_token(DEFAULT_TENANT_ID, state["conversation_id"])
    if token is None:
        return {"reply": NOT_CONNECTED}

    try:
        event = await insert_event(
            token,
            summary=state["summary"],
            start_iso=state["event_start"],
            duration_minutes=state["event_duration"],
            tz=state["event_tz"],
        )
    except Exception:
        # The row says 'confirmed' but nothing was created. Left as-is
        # deliberately: it is the record that the user approved, and a
        # retry must not slip past the gate a second time.
        log.exception("calendar_insert_failed", action_id=action_id)
        return {"reply": "I couldn't create that -- Google rejected it. Nothing was changed."}

    log.info("action_executed", action_id=action_id, event_id=event.get("id"))
    return {"reply": f"Done -- \"{state['summary']}\" is on your calendar for {when}."}


def _pick_agent(state: AssistantState) -> Literal["calendar", "drive", "email", "general"]:
    """Conditional edge: reads intent, names the next node."""
    return state["intent"]  # type: ignore[return-value]


def _calendar_branch(state: AssistantState) -> Literal["read", "write", "unavailable"]:
    operation = state.get("calendar_op", "read")
    return operation if operation in {"read", "write"} else "unavailable"


def _email_branch(state: AssistantState) -> Literal["read", "send", "unavailable"]:
    operation = state.get("email_op", "read")
    return operation if operation in {"read", "send"} else "unavailable"


def build_graph() -> StateGraph:
    builder = StateGraph(AssistantState)
    builder.add_node("route", route_node)
    builder.add_node("calendar", calendar_triage_node)
    builder.add_node("calendar_read", calendar_read_node)
    builder.add_node("calendar_propose", calendar_propose_node)
    builder.add_node("calendar_approve", calendar_approve_node)
    builder.add_node("calendar_unavailable", calendar_unavailable_node)
    builder.add_node("drive", drive_node)
    builder.add_node("email", email_triage_node)
    builder.add_node("email_read", email_read_node)
    builder.add_node("email_propose", email_propose_node)
    builder.add_node("email_approve", email_approve_node)
    builder.add_node("email_unavailable", email_unavailable_node)
    builder.add_node("general", general_node)

    builder.add_edge(START, "route")
    builder.add_conditional_edges("route", _pick_agent)

    # Reads answer straight away; only writes reach the gate.
    builder.add_conditional_edges(
        "calendar",
        _calendar_branch,
        {
            "read": "calendar_read",
            "write": "calendar_propose",
            "unavailable": "calendar_unavailable",
        },
    )
    # Propose and approve stay two hops so the row is checkpointed before
    # the wait -- see calendar_propose_node.
    builder.add_edge("calendar_propose", "calendar_approve")

    builder.add_conditional_edges(
        "email",
        _email_branch,
        {
            "read": "email_read",
            "send": "email_propose",
            "unavailable": "email_unavailable",
        },
    )
    builder.add_edge("email_propose", "email_approve")

    for node in (
        "calendar_read",
        "calendar_approve",
        "calendar_unavailable",
        "email_read",
        "email_approve",
        "email_unavailable",
        "drive",
        "general",
    ):
        builder.add_edge(node, END)
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
