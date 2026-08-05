"""
The Anthropic backend -- the primary, and the one whose message format
every other backend has to imitate.

Its only real work is flattening the SDK's response objects into plain
dicts. The SDK would happily take its own objects back on the next turn,
but Gemini cannot read them, so the canonical format has to be something
both providers can produce.
"""

import anthropic
import structlog

from app.config import settings

log = structlog.get_logger()

name = "anthropic"

_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)


def configured() -> bool:
    return bool(settings.anthropic_api_key)


def is_exhausted(exc: Exception) -> bool:
    """Is this a problem with *this provider*, rather than the request?

    Only these justify trying somewhere else. Anything not listed here is
    a bug in what we sent, and re-sending it elsewhere just hides it.
    """
    if isinstance(
        exc,
        (
            anthropic.AuthenticationError,  # 401 -- key revoked or wrong
            anthropic.PermissionDeniedError,  # 403 -- includes billing_error
            anthropic.RateLimitError,  # 429 -- after the SDK's own retries
            anthropic.InternalServerError,  # 5xx, and 529 overloaded
            anthropic.APIConnectionError,  # network, and APITimeoutError
        ),
    ):
        return True

    if isinstance(exc, anthropic.BadRequestError):
        # The awkward one. Credit exhaustion arrives as a 400
        # invalid_request_error -- the exact class as a malformed tool
        # schema, which must *not* fall back. Only the message separates
        # them, so this is string matching, and it is fragile by nature:
        # if Anthropic rewords the message this silently stops working
        # and the bot goes quiet instead of failing over.
        return "credit balance" in str(exc).lower()

    return False


def _flatten(blocks) -> list[dict]:
    """SDK content blocks -> plain dicts, dropping types we don't use."""
    out: list[dict] = []
    for block in blocks:
        if block.type == "text":
            out.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            out.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": dict(block.input),
                }
            )
    return out


async def generate(
    *,
    system: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    max_tokens: int = 1024,
) -> dict:
    kwargs = {
        "model": settings.anthropic_model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
    }
    if tools:
        kwargs["tools"] = tools

    response = await _client.messages.create(**kwargs)
    return {
        "stop_reason": response.stop_reason,
        "content": _flatten(response.content),
    }
