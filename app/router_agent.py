"""
Build order step 7: decide which specialist handles a message.

The router does ONE job -- classify. It calls no tools and answers no
questions, which keeps it cheap, fast, and easy to reason about when a
message ends up somewhere unexpected.
"""

from typing import Literal, get_args

import structlog

from app.llm import ask_claude

log = structlog.get_logger()

Intent = Literal["email", "calendar", "drive", "general"]
VALID_INTENTS = set(get_args(Intent))

# Deliberately terse. A long persona makes the model chatty, and anything
# other than a bare label has to be thrown away.
ROUTER_PROMPT = """Classify the user's message into exactly one category:

calendar - meetings, scheduling, availability, events, appointments
email    - sending, reading, searching, or drafting email
drive    - questions about documents, files, reports, policies, or any
           company information that would live in a document
general  - chat, greetings, or anything not covered above

Reply with one word only: calendar, email, drive, or general."""


async def classify_intent(text: str) -> Intent:
    """Which specialist should handle this message.

    Falls back to "general" on anything unexpected. A misrouted message
    gets a slightly worse answer; a crash here loses the message
    entirely, so the failure mode is chosen deliberately.
    """
    raw = await ask_claude(text, system=ROUTER_PROMPT, max_tokens=10)

    # Models add punctuation, capitals or a trailing sentence however
    # firmly you ask them not to. Normalise rather than trust.
    intent = raw.strip().lower().strip(".!\"'")

    if intent not in VALID_INTENTS:
        log.warning("router_unparsed", raw=raw[:40])
        intent = "general"

    log.info("router_decision", intent=intent, input_text=text[:80])
    return intent  # type: ignore[return-value]
