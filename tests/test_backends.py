"""
What must hold for the fallback router, and what breaks quietly if it
doesn't.

The one to care about is `test_malformed_request_does_not_fall_back`.
Every other failure here is loud; that one turns a bug in our own request
into "the assistant is slow and eventually says something went wrong",
with the real error swallowed on the first provider.
"""

import anthropic
import httpx
import pytest
from google.genai import errors, types

from app.backends import NoBackendAvailable, generate
from app.backends.claude import is_exhausted as claude_exhausted
from app.backends.gemini import (
    _from_gemini,
    _to_gemini_contents,
    is_exhausted as gemini_exhausted,
)


def _anthropic_error(cls, message: str, status: int):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status, request=request)
    return cls(message, response=response, body=None)


# --------------------------------------------------------------------
# Which failures justify trying another provider
# --------------------------------------------------------------------


def test_credit_exhaustion_is_a_fallback_reason():
    """The actual error that started all this."""
    exc = _anthropic_error(
        anthropic.BadRequestError,
        "Your credit balance is too low to access the Anthropic API.",
        400,
    )
    assert claude_exhausted(exc) is True


def test_malformed_request_is_not_a_fallback_reason():
    """Same class, same status code as credit exhaustion -- opposite answer.

    A bad tool schema fails identically on every provider. Falling back
    would cost a round trip per backend and then report the *last*
    provider's error, hiding the one that actually explains the bug.
    """
    exc = _anthropic_error(
        anthropic.BadRequestError,
        "tools.0.input_schema: Extra inputs are not permitted",
        400,
    )
    assert claude_exhausted(exc) is False


@pytest.mark.parametrize(
    "cls, status",
    [
        (anthropic.AuthenticationError, 401),
        (anthropic.PermissionDeniedError, 403),
        (anthropic.RateLimitError, 429),
        (anthropic.InternalServerError, 529),
    ],
)
def test_provider_side_failures_fall_back(cls, status):
    assert claude_exhausted(_anthropic_error(cls, "nope", status)) is True


def test_gemini_quota_falls_back_but_bad_request_does_not():
    """The free tier's daily limit is a 429 -- Gemini's version of no credit."""
    quota = errors.ClientError(429, {"error": {"message": "quota exceeded"}})
    malformed = errors.ClientError(400, {"error": {"message": "bad schema"}})
    assert gemini_exhausted(quota) is True
    assert gemini_exhausted(malformed) is False


def test_gemini_reports_a_rejected_key_as_400_not_401():
    """Found by a live call, not by reading the docs.

    Gemini answers a bad API key with 400 INVALID_ARGUMENT rather than
    the 401 nearly every other API uses. Classified naively, a dead
    Gemini key looks like a malformed request and the router gives up
    instead of moving on -- the same trap as Anthropic putting spent
    credit inside a 400.
    """
    bad_key = errors.ClientError(
        400,
        {
            "error": {
                "code": 400,
                "message": "API key not valid. Please pass a valid API key.",
                "status": "INVALID_ARGUMENT",
                "details": [{"reason": "API_KEY_INVALID"}],
            }
        },
    )
    assert gemini_exhausted(bad_key) is True


# --------------------------------------------------------------------
# The router itself
# --------------------------------------------------------------------


class _Backend:
    """A stand-in provider. `raises` fires once, then it starts working."""

    def __init__(self, name, raises=None, exhausted=True):
        self.name = name
        self._raises = raises
        self._exhausted = exhausted
        self.calls = 0

    def configured(self):
        return True

    def is_exhausted(self, exc):
        return self._exhausted

    async def generate(self, **kwargs):
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return {"stop_reason": "end_turn", "content": [{"type": "text", "text": self.name}]}


@pytest.fixture
def swap_backends(monkeypatch):
    def _swap(*backends):
        monkeypatch.setattr("app.backends._ALL", backends)
    return _swap


async def test_falls_through_to_the_second_provider(swap_backends):
    dead = _Backend("dead", raises=RuntimeError("no credit"), exhausted=True)
    alive = _Backend("alive")
    swap_backends(dead, alive)

    result = await generate(system="s", messages=[{"role": "user", "content": "hi"}])

    assert result["content"][0]["text"] == "alive"
    assert dead.calls == 1 and alive.calls == 1


async def test_a_request_bug_stops_at_the_first_provider(swap_backends):
    """The invariant that keeps a real error visible."""
    broken = _Backend("broken", raises=ValueError("bad schema"), exhausted=False)
    spare = _Backend("spare")
    swap_backends(broken, spare)

    with pytest.raises(ValueError, match="bad schema"):
        await generate(system="s", messages=[{"role": "user", "content": "hi"}])

    assert spare.calls == 0, "must not burn a round trip on a doomed request"


async def test_all_providers_down_raises_its_own_error(swap_backends):
    swap_backends(
        _Backend("a", raises=RuntimeError("down")),
        _Backend("b", raises=RuntimeError("down")),
    )
    with pytest.raises(NoBackendAvailable):
        await generate(system="s", messages=[{"role": "user", "content": "hi"}])


async def test_healthy_primary_never_reaches_the_fallback(swap_backends):
    primary, fallback = _Backend("primary"), _Backend("fallback")
    swap_backends(primary, fallback)

    result = await generate(system="s", messages=[{"role": "user", "content": "hi"}])

    assert result["content"][0]["text"] == "primary"
    assert fallback.calls == 0


# --------------------------------------------------------------------
# Gemini translation
# --------------------------------------------------------------------


def test_assistant_becomes_model():
    [_, assistant] = _to_gemini_contents(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
        ]
    )
    assert assistant.role == "model"


def test_tool_result_is_paired_by_name_not_id():
    """Anthropic pairs by tool_use_id; Gemini has no ids and pairs by name.

    Getting this wrong sends a result for a function Gemini never called,
    which it answers by re-calling the tool -- an infinite-ish loop that
    looks like the model being stupid rather than a translation bug.
    """
    contents = _to_gemini_contents(
        [
            {"role": "user", "content": "what's on my calendar"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_01",
                        "name": "read_calendar",
                        "input": {"days": 1},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01",
                        "content": "Lunch at 1pm",
                    }
                ],
            },
        ]
    )
    response_part = contents[2].parts[0].function_response
    assert response_part.name == "read_calendar"
    assert response_part.response == {"output": "Lunch at 1pm"}


def test_orphan_tool_result_is_dropped_not_mislabelled():
    """History truncated mid-turn: better to omit than to guess a name."""
    contents = _to_gemini_contents(
        [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_gone", "content": "x"}
                ],
            }
        ]
    )
    assert contents == []


def test_empty_text_blocks_are_skipped():
    """Anthropic tolerates an empty text block; Gemini rejects the request."""
    contents = _to_gemini_contents(
        [{"role": "assistant", "content": [{"type": "text", "text": "   "}]}]
    )
    assert contents == []


def _fake_response(parts):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=parts))]
    )


def test_gemini_calls_get_synthesised_ids():
    """Gemini issues no ids, but everything downstream pairs results by id."""
    result = _from_gemini(
        _fake_response(
            [
                types.Part(
                    function_call=types.FunctionCall(
                        name="read_email", args={"limit": 5}
                    )
                ),
                types.Part(
                    function_call=types.FunctionCall(
                        name="read_calendar", args={"days": 1}
                    )
                ),
            ]
        )
    )
    assert result["stop_reason"] == "tool_use"
    ids = [b["id"] for b in result["content"]]
    assert len(set(ids)) == 2, "ids must be unique within a turn"
    assert result["content"][0]["input"] == {"limit": 5}


def test_plain_text_reply_ends_the_turn():
    result = _from_gemini(_fake_response([types.Part(text="No meetings today.")]))
    assert result["stop_reason"] == "end_turn"
    assert result["content"] == [{"type": "text", "text": "No meetings today."}]


def test_empty_candidates_do_not_crash():
    """A safety-blocked Gemini response has no candidates at all."""
    result = _from_gemini(types.GenerateContentResponse(candidates=[]))
    assert result["stop_reason"] == "end_turn"
    assert result["content"] == []


async def test_gemini_never_executes_tools_itself(monkeypatch):
    """Gemini's SDK runs the tool loop for you unless told not to.

    Left on, it would execute a call and return only the final text --
    so `create_calendar_event` or `send_email` would happen with no
    confirmation, silently bypassing the approval gate. This is the one
    Gemini default that is a security problem rather than a nuisance.
    """
    from app.backends import gemini

    captured = {}

    class _FakeModels:
        async def generate_content(self, *, model, contents, config):
            captured["config"] = config
            return types.GenerateContentResponse(candidates=[])

    monkeypatch.setattr(
        gemini, "_client", type("C", (), {"aio": type("A", (), {"models": _FakeModels()})()})()
    )
    monkeypatch.setattr("app.config.settings.gemini_model", "gemini-2.5-flash")

    await gemini.generate(
        system="s",
        messages=[{"role": "user", "content": "book me lunch tomorrow"}],
        tools=[
            {
                "name": "create_calendar_event",
                "description": "Create an event",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
    )

    assert captured["config"].automatic_function_calling.disable is True
