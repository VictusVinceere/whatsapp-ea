"""
Build order step 4b: build this specialist FIRST, before email/drive --
simplest OAuth surface. Get real Google Calendar read access working,
then wrap it in the approval gate (step 6), THEN copy the pattern to
the other two agents (step 7).
"""
import structlog

log = structlog.get_logger()


async def handle_calendar_request(text: str, conversation_id: str) -> str:
    """Entry point the router hands off to. Should use LangGraph
    internally once you're past the first plain-function version:
    check availability -> propose action -> write to pending_actions
    -> return a confirmation prompt for the user."""
    # TODO: Google Calendar API call via OAuth
    # TODO: write proposed action to pending_actions table
    return "Calendar agent not wired up yet."
