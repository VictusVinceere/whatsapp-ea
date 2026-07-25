"""
Build order step 5. Add this only after a single plain LLM call
(step 3, in process_message) is round-tripping correctly.

The router does ONE job: classify which specialist this message
belongs to. It does not call any tools itself.
"""
from typing import Literal
import structlog

log = structlog.get_logger()

Intent = Literal["email", "calendar", "drive", "general"]


async def classify_intent(text: str) -> Intent:
    """Placeholder -- replace with an LLM call (structured output,
    force it to return one of the four labels above) or a LangGraph
    node once the graph is wired up."""
    log.info("router_decision", input_text=text[:80])
    # TODO: real classification
    return "general"
