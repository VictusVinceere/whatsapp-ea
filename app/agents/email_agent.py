"""Build after calendar_agent works end-to-end with the approval gate."""
import structlog

log = structlog.get_logger()


async def handle_email_request(text: str, conversation_id: str) -> str:
    # TODO: Gmail API — search/read/draft via OAuth
    # TODO: write proposed action to pending_actions table before any send
    return "Email agent not wired up yet."
