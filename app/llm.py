"""
Build order step 3: one plain LLM call, no LangGraph yet. Prove the
AI reasoning loop works before adding router/agent complexity on top.
"""
from datetime import datetime, timezone

import anthropic
import structlog
from app.config import settings

log = structlog.get_logger()
client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

SYSTEM_PROMPT = """You are an executive assistant on WhatsApp. Keep
replies short and conversational -- this is chat, not email.

You have tools for the user's documents, calendar and inbox. Use them
rather than guessing, and call several in one turn when a request needs
more than one. Creating events and sending email are confirmed with the
user automatically, so never ask permission yourself -- just make the
call and let the confirmation happen."""


async def ask_claude(
    text: str,
    history: list[dict] | None = None,
    system: str | None = None,
    max_tokens: int = 300,
) -> str:
    """Answer `text`, optionally with prior turns for context.

    The API is stateless -- it knows only what this call carries, so the
    conversation is replayed in full every time. `history` must be in
    chronological order and end on an assistant turn; this appends the
    new user message after it.

    `system` overrides the assistant persona -- the router uses it to get
    a bare classification instead of a chatty reply.
    """
    messages = [*(history or []), {"role": "user", "content": text}]
    response = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        system=system or SYSTEM_PROMPT,
        messages=messages,
    )
    return response.content[0].text


MAX_TOOL_ROUNDS = 6


async def run_agent(
    text: str,
    conversation_id: str,
    history: list[dict] | None = None,
    system: str | None = None,
) -> dict:
    """Let Claude call tools until it can answer, or wants to change something.

    Returns either:
        {"reply": "..."}                      finished talking
        {"pending": {"name": ..., "input": ..., "messages": [...]}}
                                              wants a write; needs approval

    A manual loop rather than the SDK's tool runner because a write has to
    suspend the whole conversation into a LangGraph interrupt -- possibly
    for days, across a process restart. The runner's hooks gate within one
    process; they can't pause a turn and resume it tomorrow.
    """
    from app.tools import TOOL_DEFINITIONS, WRITE_TOOLS, run_read_tool

    # Without this the model has no idea what "tomorrow" means and will
    # anchor on any date it happens to see -- observed booking a lunch in
    # October because the calendar it had just read contained a birthday
    # there. Cheap to include, and the failure is silent without it.
    now = datetime.now(timezone.utc)
    dated_system = (
        f"{system or SYSTEM_PROMPT}\n\n"
        f"The current date and time is {now:%A %d %B %Y, %H:%M} UTC. "
        f"Resolve relative dates like 'tomorrow' against it. Times the user "
        f"gives are in their own timezone, which is applied when the event "
        f"is created -- pass them through unchanged."
    )

    messages = [*(history or []), {"role": "user", "content": text}]

    for _ in range(MAX_TOOL_ROUNDS):
        response = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=dated_system,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            text_blocks = [b.text for b in response.content if b.type == "text"]
            return {"reply": "\n".join(text_blocks).strip() or "..."}

        calls = [b for b in response.content if b.type == "tool_use"]
        messages.append({"role": "assistant", "content": response.content})

        # A write ends the loop. Anything the model already read stays in
        # `messages`, so the turn resumes with its context intact after
        # the user answers.
        write = next((c for c in calls if c.name in WRITE_TOOLS), None)
        if write is not None:
            log.info("tool_write_proposed", tool=write.name)
            return {
                "pending": {
                    "name": write.name,
                    "input": dict(write.input),
                    "tool_use_id": write.id,
                    "messages": messages,
                }
            }

        # Parallel calls must come back as tool_results in ONE user
        # message -- splitting them teaches the model to stop parallelising.
        results = []
        for call in calls:
            log.info("tool_call", tool=call.name)
            output = await run_read_tool(call.name, dict(call.input), conversation_id)
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": output,
                }
            )
        messages.append({"role": "user", "content": results})

    return {"reply": "That turned into more steps than I can handle -- try asking for one thing at a time."}
