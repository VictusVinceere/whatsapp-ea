"""Build after calendar_agent works end-to-end with the approval gate."""
import structlog

log = structlog.get_logger()


async def handle_drive_request(text: str, conversation_id: str) -> str:
    # TODO: pgvector similarity search over embedded Drive docs
    # TODO: Google Drive API for reading matched files
    return "Drive RAG agent not wired up yet."
