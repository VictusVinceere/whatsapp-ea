"""Answers questions from ingested documents via retrieval.

Read-only, so no approval gate -- nothing here changes the world. The
documents currently come from local files; swapping the source to Google
Drive changes only the ingestion side, not this.
"""
import structlog

from app.db import DEFAULT_TENANT_ID
from app.rag import answer_with_context

log = structlog.get_logger()


async def handle_drive_request(text: str, conversation_id: str) -> str:
    log.info("drive_agent_handling", conversation_id=conversation_id)
    # TODO: tenant_id is hardcoded until customers have their own ids.
    # It must become per-customer before this is multi-tenant -- a
    # retrieval that searches the wrong tenant is a data breach.
    return await answer_with_context(DEFAULT_TENANT_ID, text)
