"""
Build order step 4 (part 2): GET verifies the webhook with Meta once.
POST receives every incoming message. Get text messages round-tripping
(receive -> print -> send canned reply) before adding anything else.
"""
import structlog
from fastapi import APIRouter, Request, BackgroundTasks, Query, Response

from app.config import settings
from app.whatsapp import parse_incoming_message, send_text_message

log = structlog.get_logger()
router = APIRouter()


@router.get("/webhook")
async def verify_webhook(
    hub_mode: str = Query(alias="hub.mode"),
    hub_verify_token: str = Query(alias="hub.verify_token"),
    hub_challenge: str = Query(alias="hub.challenge"),
):
    """Meta calls this once when you register the webhook URL."""
    if hub_mode == "subscribe" and hub_verify_token == settings.whatsapp_verify_token:
        return Response(content=hub_challenge, media_type="text/plain")
    return Response(status_code=403)


@router.post("/webhook")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
    """Ack immediately, process in the background so Meta never times out
    waiting on an LLM call. This is the async pattern from our RabbitMQ
    discussion -- BackgroundTasks does the same job at this scale."""
    payload = await request.json()
    message = parse_incoming_message(payload)

    if message is None:
        return {"status": "ignored"}

    log.info("message_received", **{k: v for k, v in message.items() if k != "text"})
    background_tasks.add_task(process_message, message)
    return {"status": "received"}


async def process_message(message: dict):
    """This is where the pipeline grows over the build:
    step 1 (today): print + canned reply
    step 2: voice note branch -> Whisper transcription
    step 3: single LLM call, no framework
    step 5: LangGraph router -> specialist agents
    step 6: approval gate before any real action
    """
    correlation_id = message["message_id"]
    log.info("processing_start", correlation_id=correlation_id)

    if message["type"] == "text":
        reply = f"Got it: {message['text']}"  # placeholder, replace with LLM call
    elif message["type"] == "audio":
        reply = "Voice note received (transcription not wired up yet)"
    else:
        reply = "Unsupported message type for now."

    await send_text_message(to=message["from"], body=reply)
    log.info("processing_done", correlation_id=correlation_id)
