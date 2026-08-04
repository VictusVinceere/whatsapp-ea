"""
Tool definitions and the read-side executor.

Replaces the classify-then-route router. That router picked exactly one
label out of four, so "pull my last events and last email" -- which asks
for two things -- matched neither and fell through to general chat. With
tools, Claude calls whichever it needs, in whatever order, as many as it
wants. The ambiguity that broke routing simply isn't expressible here.

Reads and writes are separated by list membership, not by anything the
model decides:

  READ_TOOLS   execute immediately inside the loop
  WRITE_TOOLS  never execute here -- they end the loop and become a
               proposal for the approval gate

That split is deliberate. If the model could choose whether something
needs approval, a prompt injection in a document could talk it out of
asking.
"""

import structlog

from app.calendar_api import describe_events, list_events
from app.db import DEFAULT_TENANT_ID
from app.gmail_api import describe_messages, list_recent
from app.google import valid_access_token
from app.rag import search

log = structlog.get_logger()

READ_TOOLS = {"search_documents", "read_calendar", "read_email"}
WRITE_TOOLS = {"create_calendar_event", "send_email"}

TOOL_DEFINITIONS = [
    {
        "name": "search_documents",
        "description": (
            "Search the user's indexed documents (reports, policies, notes) "
            "for passages relevant to a question. Use for anything about "
            "company information, figures, or written policy."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look for, in natural language.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_calendar",
        "description": "List the user's upcoming calendar events.",
        "input_schema": {
            "type": "object",
            "properties": {
                "max_results": {
                    "type": "integer",
                    "description": "How many events to fetch. Default 8.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "read_email",
        "description": "List recent messages from the user's inbox.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Gmail search query, e.g. 'in:inbox' or 'from:sarah'.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "How many messages to fetch. Default 6.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "create_calendar_event",
        "description": (
            "Create an event on the user's calendar. This asks the user to "
            "confirm before anything is created, so call it as soon as you "
            "have a title, a start time and a duration."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Event title."},
                "start": {
                    "type": "string",
                    "description": "Local start time, YYYY-MM-DDTHH:MM:SS.",
                },
                "duration_minutes": {"type": "integer"},
            },
            "required": ["summary", "start", "duration_minutes"],
        },
    },
    {
        "name": "send_email",
        "description": (
            "Send an email from the user's account. The user is shown the "
            "full draft and must confirm, so do not ask for permission "
            "yourself -- just call this. Never invent a recipient address; "
            "if you don't have one, ask for it instead of calling this."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient address."},
                "subject": {"type": "string"},
                "body": {"type": "string", "description": "Full message body."},
            },
            "required": ["to", "subject", "body"],
        },
    },
]


async def run_read_tool(name: str, tool_input: dict, conversation_id: str) -> str:
    """Execute a read tool and return text for the model.

    Returns a plain sentence on failure rather than raising: the model can
    tell the user "your calendar isn't connected" and carry on, where an
    exception would abandon the whole turn.
    """
    if name == "search_documents":
        hits = await search(DEFAULT_TENANT_ID, tool_input["query"])
        if not hits:
            return "No matching documents."
        return "\n\n---\n\n".join(
            f"[{h['source_name']}]\n{h['content']}" for h in hits
        )

    token = await valid_access_token(DEFAULT_TENANT_ID, conversation_id)
    if token is None:
        return "The user has not connected their Google account."

    if name == "read_calendar":
        events = await list_events(
            token, max_results=int(tool_input.get("max_results") or 8)
        )
        return describe_events(events)

    if name == "read_email":
        messages = await list_recent(
            token,
            max_results=int(tool_input.get("max_results") or 6),
            query=tool_input.get("query") or "in:inbox",
        )
        return describe_messages(messages)

    return f"Unknown tool: {name}"
