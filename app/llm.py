"""
Build order step 3: one plain LLM call, no LangGraph yet. Prove the
AI reasoning loop works before adding router/agent complexity on top.
"""
import anthropic
from app.config import settings

client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

SYSTEM_PROMPT = """You are a helpful executive assistant communicating
over WhatsApp. Keep replies short and conversational -- this is a chat
interface, not email. You can search the user's documents; calendar and
email are still being wired up."""


async def ask_claude(
    text: str,
    history: list[dict] | None = None,
    system: str | None = None,
    max_tokens: int = 300,
) -> str:
    """Answer `text`, optionally with prior turns for context.

    The API is stateless -- it knows only what this call carries, so the
    conversation is replayed in full every time. `history` must be in
    chronological order and end on an assistant turn; this appends the
    new user message after it.

    `system` overrides the assistant persona -- the router uses it to get
    a bare classification instead of a chatty reply.
    """
    messages = [*(history or []), {"role": "user", "content": text}]
    response = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        system=system or SYSTEM_PROMPT,
        messages=messages,
    )
    return response.content[0].text
