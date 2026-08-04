"""
Build order step 2: voice notes.

Deepgram's Python SDK went through a major rewrite that dropped
PrerecordedOptions and the old client.listen.prerecorded.v("1") path
in favor of this flatter API. Pinned at 7.x (see uv.lock); the
constructor takes api_key as a keyword only -- a positional arg
raises TypeError from BaseClient. If this breaks again on a future
version bump, check https://github.com/deepgram/deepgram-python-sdk
for the current client.listen.* shape before assuming your own code
is wrong -- this SDK's surface has changed more than once.

The underlying call is synchronous, so we run it in a thread via
asyncio.to_thread() to avoid blocking the event loop -- same reasoning
as the async discussion earlier: a blocking call inside an async route
freezes every other in-flight request, not just this one.
"""
import asyncio
from deepgram import DeepgramClient
from app.config import settings

deepgram = DeepgramClient(api_key=settings.deepgram_api_key)


def _transcribe_sync(audio_bytes: bytes) -> str:
    response = deepgram.listen.v1.media.transcribe_file(
        request=audio_bytes,
        model="nova-3",
    )
    return response.results.channels[0].alternatives[0].transcript


async def transcribe_audio(audio_bytes: bytes) -> str:
    """WhatsApp voice notes arrive as .ogg (Opus codec) -- Deepgram
    handles this natively, no conversion needed."""
    return await asyncio.to_thread(_transcribe_sync, audio_bytes)
