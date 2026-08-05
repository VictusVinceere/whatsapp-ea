"""
The Gemini backend -- the free-tier safety net.

Everything hard about this file is translation. Gemini's wire format
disagrees with Anthropic's in four ways that each break something
quietly if you get them wrong:

  Anthropic                        Gemini
  ---------------------------------------------------------------
  role "assistant"                 role "model"
  system= parameter                config.system_instruction
  tool_result keyed by id          function_response keyed by *name*
  tool_use blocks carry an id      function calls carry no id at all

The last two are the ones that bite. Anthropic pairs a result to its
call by `tool_use_id`; Gemini pairs by function name, so replaying a
conversation means rebuilding the id -> name map as we walk it. And
because Gemini never issues ids, this module invents them on the way
out -- otherwise the rest of the app, which pairs by id, has nothing to
pair with.

Translation is confined to this file on purpose. Everything outside it
sees Anthropic-shaped dicts and never learns a second provider exists.
"""

import uuid

import structlog
from google import genai
from google.genai import errors, types

from app.config import settings

log = structlog.get_logger()

name = "gemini"

_client = (
    genai.Client(api_key=settings.gemini_api_key)
    if settings.gemini_api_key
    else None
)


def configured() -> bool:
    return bool(settings.gemini_api_key)


def is_exhausted(exc: Exception) -> bool:
    """Same contract as the Anthropic backend: provider problem, or ours?

    Gemini's free tier fails with 429 once the daily quota is spent,
    which is this backend's equivalent of running out of credit.
    """
    if isinstance(exc, errors.ServerError):  # 5xx
        return True
    if isinstance(exc, errors.ClientError):
        # 401/403 bad key, 429 quota exhausted. A 400 is a malformed
        # request and deliberately excluded -- see the router's note on
        # why retrying those elsewhere is worse than failing.
        return exc.code in {401, 403, 429}
    return False


def _tools_to_gemini(tools: list[dict]) -> list[types.Tool]:
    """Anthropic tool definitions -> one Gemini Tool.

    `parameters_json_schema` takes raw JSON Schema, so Anthropic's
    `input_schema` passes through untouched. The older `parameters` field
    wants Gemini's own restricted Schema type and would need a converter.
    """
    return [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=t["name"],
                    description=t.get("description", ""),
                    parameters_json_schema=t["input_schema"],
                )
                for t in tools
            ]
        )
    ]


def _as_text(content) -> str:
    """A tool_result's content, which may be a string or a block list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") for b in content if isinstance(b, dict)
        )
    return str(content)


def _to_gemini_contents(messages: list[dict]) -> list[types.Content]:
    """Anthropic messages -> Gemini contents.

    Walks in order because the id -> name map has to be built from the
    assistant turn *before* the tool results that reference it.
    """
    contents: list[types.Content] = []
    id_to_name: dict[str, str] = {}

    for message in messages:
        role = "user" if message["role"] == "user" else "model"
        content = message["content"]

        if isinstance(content, str):
            if content.strip():
                contents.append(
                    types.Content(role=role, parts=[types.Part(text=content)])
                )
            continue

        parts: list[types.Part] = []
        for block in content:
            block = block if isinstance(block, dict) else block.model_dump()
            kind = block.get("type")

            if kind == "text":
                # Gemini rejects an empty text part; Anthropic tolerates one.
                if block.get("text", "").strip():
                    parts.append(types.Part(text=block["text"]))

            elif kind == "tool_use":
                id_to_name[block["id"]] = block["name"]
                parts.append(
                    types.Part(
                        function_call=types.FunctionCall(
                            name=block["name"], args=block.get("input") or {}
                        )
                    )
                )

            elif kind == "tool_result":
                # Keyed by name, not id -- so a result whose call we never
                # saw cannot be matched. That only happens if history was
                # truncated mid-turn; naming it is better than silently
                # sending a response for a function Gemini never called.
                fn = id_to_name.get(block.get("tool_use_id", ""))
                if fn is None:
                    log.warning("orphan_tool_result", id=block.get("tool_use_id"))
                    continue
                parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=fn,
                            response={"output": _as_text(block.get("content"))},
                        )
                    )
                )

        if parts:
            contents.append(types.Content(role=role, parts=parts))

    return contents


def _from_gemini(response) -> dict:
    """Gemini response -> Anthropic-shaped blocks.

    Invents a `toolu_` id per call. Nothing downstream cares that it did
    not come from a real provider -- it only has to be unique within the
    turn so results can be paired back to calls.
    """
    blocks: list[dict] = []

    candidates = response.candidates or []
    parts = candidates[0].content.parts if candidates and candidates[0].content else []

    for part in parts or []:
        if getattr(part, "text", None):
            blocks.append({"type": "text", "text": part.text})
        call = getattr(part, "function_call", None)
        if call is not None:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": f"toolu_gem_{uuid.uuid4().hex[:20]}",
                    "name": call.name,
                    "input": dict(call.args or {}),
                }
            )

    wants_tool = any(b["type"] == "tool_use" for b in blocks)
    return {
        "stop_reason": "tool_use" if wants_tool else "end_turn",
        "content": blocks,
    }


async def generate(
    *,
    system: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    max_tokens: int = 1024,
) -> dict:
    config = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=max_tokens,
    )
    if tools:
        config.tools = _tools_to_gemini(tools)
        # Gemini otherwise runs the whole tool loop itself, executing
        # calls and returning only the final text. That would bypass the
        # approval gate entirely -- a write would happen with no
        # confirmation. This hands control back after every call.
        config.automatic_function_calling = types.AutomaticFunctionCallingConfig(
            disable=True
        )

    response = await _client.aio.models.generate_content(
        model=settings.gemini_model,
        contents=_to_gemini_contents(messages),
        config=config,
    )
    return _from_gemini(response)
