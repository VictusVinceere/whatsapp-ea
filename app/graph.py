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
from app.drive_api import upload_file
from app.rag import delete_document
from app.whatsapp import download_media, get_media_url
from app.config import settings
from app.db import DEFAULT_TENANT_ID, async_session
from app.google import valid_access_token
from app.llm import ask_claude, run_agent

log = structlog.get_logger()

NOT_CONNECTED = (
    "I can't reach your Google account -- connect it first with /oauth/start."
)


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
    # {media_id, filename, mime_type} of a file forwarded with
    # this message. Supplied by the webhook, not the model.
    document: dict


def _clear_pending(reply: str) -> dict:
    """A reply, plus an explicit reset of the pending write.

    LangGraph state is cumulative per thread, and thread_id here is the
    conversation, so a field set on one message is still set on the next.
    Leaving pending_tool behind meant _needs_approval saw a finished
    action and re-proposed it: asking for an email got the previous
    calendar event offered again, and confirming it created a duplicate.

    Returning None is not the same as omitting the key -- omitting merges
    nothing and the stale value survives.
    """
    return {"reply": reply, "pending_tool": None, "pending_input": None}


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
        args = dict(pending["input"])
        if pending["name"] == "save_to_drive":
            # The model knows a file arrived but not how to fetch it, and
            # it should not: a media id in tool input is something a
            # prompt injection could rewrite. Taken from state instead.
            args.update(state.get("document") or {})
        return {"pending_tool": pending["name"], "pending_input": args}
    return _clear_pending(outcome["reply"])


async def propose_node(state: AssistantState) -> dict:
    """Record the proposed write. Its own node because a node containing
    interrupt() re-runs from the top on resume -- see the docstring on
    approve_node."""
    agent = {
        "create_calendar_event": "calendar",
        "send_email": "email",
        "save_to_drive": "drive",
        "delete_document": "drive",
    }.get(state["pending_tool"], "unknown")

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
    if tool == "delete_document":
        return (
            f"Remove \"{args.get('source_name')}\" from the search index?\n"
            "The original file in Drive is not affected."
        )

    if tool == "save_to_drive":
        return f"Save \"{args.get('filename')}\" to your Google Drive?"

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
        return _clear_pending("Cancelled, nothing was changed.")

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
        return _clear_pending("That was already done.")

    token = await valid_access_token(DEFAULT_TENANT_ID, state["conversation_id"])
    if token is None:
        return _clear_pending(NOT_CONNECTED)

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
        elif tool == "delete_document":
            removed = await delete_document(
                DEFAULT_TENANT_ID, args["source_name"]
            )
            done = (
                f"Removed \"{args['source_name']}\" ({removed} sections)."
                if removed
                else f"Nothing indexed under \"{args['source_name']}\"."
            )
        elif tool == "save_to_drive":
            # Re-downloaded rather than carried through the approval: the
            # bytes could be megabytes, and pending_actions.payload is a
            # JSONB audit record, not a blob store.
            media_id = args.get("media_id")
            if not media_id:
                # No file reference on this conversation. Nothing was
                # uploaded, so say that plainly rather than reporting a
                # Google failure that never happened.
                return _clear_pending(
                    "I've lost track of which file you meant. Send it again "
                    "and I'll save it."
                )
            media_url = await get_media_url(media_id)
            data = await download_media(media_url)
            uploaded = await upload_file(
                token,
                name=args["filename"],
                data=data,
                mime_type=args.get("mime_type") or "application/octet-stream",
            )
            done = f"Saved \"{uploaded.get('name')}\" to your Drive."
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
        return _clear_pending("I couldn't do that -- Google rejected it. Nothing changed.")

    log.info("action_executed", action_id=action_id, tool=tool)
    return _clear_pending(done)


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

# Unambiguous replies, matched before spending a model call. Anything
# outside these two sets goes to Claude, because "sure, go ahead" and
# "actually don't" are answers too, and treating them as new requests
# would silently cancel what the user just approved.
YES_WORDS = {"yes", "y", "yeah", "yep", "yup", "ok", "okay", "sure",
             "confirm", "confirmed", "send", "do it", "go ahead", "please do"}
NO_WORDS = {"no", "n", "nope", "nah", "cancel", "stop", "don't", "dont",
            "no thanks", "never mind", "nevermind"}


async def _classify_reply(message: str, question: str) -> Literal["yes", "no", "new"]:
    """Is this an approval, a refusal, or a change of subject?

    Without this every message resumed the interrupt, so an unrelated
    question while an approval was pending got consumed as a rejection --
    observed live: "did you send the email to my friend windsor" was eaten
    as an answer and the conversation sat parked for hours.

    Normalising here rather than inside approve_node is deliberate: two
    places matching yes/no with different word lists meant "sure, go
    ahead" passed this check and was then rejected downstream. One
    matcher, one vocabulary.
    """
    normalised = message.strip().lower().rstrip("!.")
    if normalised in YES_WORDS:
        return "yes"
    if normalised in NO_WORDS:
        return "no"

    verdict = await ask_claude(
        f"Pending question: {question}\n\nUser said: {message}",
        system=(
            "The user was asked to confirm an action. Did they agree, "
            "decline, or say something unrelated to it? Reply with exactly "
            "one word: yes, no, or new."
        ),
        max_tokens=5,
    )
    answer = verdict.strip().lower().rstrip(".!")
    return answer if answer in {"yes", "no", "new"} else "new"  # type: ignore[return-value]


async def _abandon_pending(conversation_id: str) -> None:
    """Mark the outstanding proposal rejected when the user moves on.

    Conditional on status so it can never overwrite one that was already
    confirmed -- same reasoning as CONFIRM_ACTION_SQL.
    """
    async with async_session() as session:
        await session.execute(
            sql(
                "UPDATE pending_actions SET status = 'rejected' "
                "WHERE conversation_id = :c AND status = 'pending'"
            ),
            {"c": conversation_id},
        )
        await session.commit()


def set_graph(compiled) -> None:
    global _graph
    _graph = compiled


def _turn(
    conversation_id: str,
    message: str,
    history: list[dict] | None,
    document: dict | None,
) -> dict:
    """The state update for one incoming message.

    `document` is omitted entirely when this message carries no file.
    Passing {} instead would overwrite the reference from the turn the
    file actually arrived on, because state is cumulative per thread and
    this key has no reducer. That broke the ordinary way people use it:
    send a file, then ask to save it in a separate message. By approval
    time the media id was gone.
    """
    turn = {
        "conversation_id": conversation_id,
        "message": message,
        "history": history or [],
    }
    if document:
        turn["document"] = document
    return turn


async def run_graph(
    conversation_id: str,
    message: str,
    history: list[dict] | None = None,
    document: dict | None = None,
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
        question = ""
        if state.tasks and state.tasks[0].interrupts:
            question = str(state.tasks[0].interrupts[0].value)

        verdict = await _classify_reply(message, question)

        if verdict in {"yes", "no"}:
            log.info(
                "graph_resuming",
                conversation_id=conversation_id,
                at=state.next,
                verdict=verdict,
            )
            # The canonical word, not the user's phrasing -- approve_node
            # matches a fixed vocabulary and "sure, go ahead" isn't in it.
            result = await _graph.ainvoke(Command(resume=verdict), config)
        else:
            # Changed the subject. Decline the pending question to unstick
            # the graph, then answer what was actually asked. Resuming
            # first matters: approve_node still needs pending_input to
            # describe what it is cancelling.
            log.info("pending_abandoned", conversation_id=conversation_id)
            await _abandon_pending(conversation_id)
            await _graph.ainvoke(Command(resume="no"), config)
            result = await _graph.ainvoke(
                _turn(conversation_id, message, history, document), config
            )
    else:
        result = await _graph.ainvoke(
            _turn(conversation_id, message, history, document), config
        )

    # An interrupted run has no "reply" -- what the user should see is the
    # question the interrupt raised ("Reply YES to confirm...").
    pending = result.get("__interrupt__")
    if pending:
        log.info("graph_interrupted", conversation_id=conversation_id)
        return pending[0].value

    return result["reply"]
