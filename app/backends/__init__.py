"""
One model call, several providers behind it.

The problem this solves is mundane: Anthropic credit hits zero and the
assistant stops answering. A second provider on a free tier keeps it
alive. The interesting part is what that costs architecturally.

**Anthropic's message format is the canonical one.** Every backend
speaks it -- Gemini translates in and out at its own boundary and
nowhere else. That is what keeps `graph.py`, `tools.py`, the approval
gate and the Postgres checkpointer completely unaware that a second
provider exists, and it means a turn can start on one provider and
finish on the other: the conversation in `messages` is portable.

**Falling back is not the same as retrying.** A backend is skipped only
when the failure is about *that backend* -- no credit, bad key, rate
limited, provider down. A malformed request fails identically on every
provider, so retrying it turns one fast error into N slow ones and
buries the actual bug. Each backend classifies its own SDK's exceptions
because there is no shared hierarchy to match on.

Note that this is *not* Anthropic's `fallbacks` request parameter. That
one triggers on safety refusals only -- rate limits, billing and server
errors are returned as-is -- so it does not help when credit runs out.
"""

import structlog

from app.backends import claude, gemini

log = structlog.get_logger()

# Order is priority. Anthropic first because it is the one the prompts and
# tool descriptions were tuned against; Gemini is the safety net, not an
# equal partner.
_ALL = (claude, gemini)


class NoBackendAvailable(RuntimeError):
    """Every provider was unusable. Distinct from a request being wrong."""


def _enabled() -> list:
    """Backends with credentials configured, in priority order.

    An unconfigured backend is skipped silently rather than failing --
    running with only Anthropic set is the normal case, not an error.
    """
    return [b for b in _ALL if b.configured()]


async def generate(
    *,
    system: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    max_tokens: int = 1024,
) -> dict:
    """One assistant turn, in Anthropic's response shape.

    Returns `{"stop_reason": "tool_use" | "end_turn", "content": [blocks]}`
    where blocks are plain dicts -- `{"type": "text", ...}` and
    `{"type": "tool_use", ...}`. Plain dicts rather than SDK objects so
    the two providers produce genuinely interchangeable output, and so
    what LangGraph checkpoints into Postgres is ordinary JSON.
    """
    backends = _enabled()
    if not backends:
        raise NoBackendAvailable("no model provider is configured")

    last: Exception | None = None
    for index, backend in enumerate(backends):
        try:
            result = await backend.generate(
                system=system, messages=messages, tools=tools, max_tokens=max_tokens
            )
        except Exception as exc:
            if not backend.is_exhausted(exc):
                # The request itself is wrong. Every other provider would
                # reject it too, so fail now with the real error rather
                # than after three more round trips.
                raise
            log.warning(
                "backend_unavailable",
                backend=backend.name,
                error=type(exc).__name__,
                detail=str(exc)[:200],
            )
            last = exc
            continue

        if index > 0:
            # Worth its own line: replies are now coming from a different
            # model than the prompts were tuned for, and that shows up as
            # quality drift long before anyone checks the billing page.
            log.warning("served_by_fallback", backend=backend.name)
        return result

    raise NoBackendAvailable(
        f"all {len(backends)} providers unavailable"
    ) from last
