"""
Build order step 4 (part 2): GET verifies the webhook with Meta once.
POST receives every incoming message. Get text messages round-tripping
(receive -> print -> send canned reply) before adding anything else.
"""

import structlog
from fastapi import APIRouter, BackgroundTasks, Query, Request, Response

from app.agents.calendar_agent import handle_calendar_request
from app.agents.drive_agent import handle_drive_request
from app.agents.email_agent import handle_email_request
from app.config import settings
from app.db import DEFAULT_TENANT_ID, recent_messages, save_message
from app.llm import ask_claude
from app.router_agent import classify_intent
from app.transcription import transcribe_audio
from app.whatsapp import (
    download_media,
    get_media_url,
    parse_incoming_message,
    send_text_message,
)

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

    # Only handle traffic for the number this app is configured to send as.
    # Meta's webhook is app-wide, so every number on the business account
    # arrives here -- without this, a production line's messages get
    # answered by whatever happens to be running locally.
    recipient = message.get("to_phone_number_id")
    if recipient and recipient != settings.whatsapp_phone_number_id:
        log.info(
            "foreign_number_ignored",
            correlation_id=message["message_id"],
            to_phone_number_id=recipient,
        )
        return {"status": "ignored"}

    # Dev safety net: while testing on a number real customers also use,
    # only answer whitelisted senders. Unset in production.
    allowed = settings.whatsapp_allowed_senders
    if allowed and message["from"] not in allowed:
        log.info("sender_not_allowed", correlation_id=message["message_id"])
        return {"status": "ignored"}

    log.info("message_received", **{k: v for k, v in message.items() if k != "text"})
    background_tasks.add_task(process_message, message)
    return {"status": "received"}


async def process_message(message: dict):
    """This is where the pipeline grows over the build:
    step 1 (done): text loop, canned reply
    step 2 (this step): voice note branch -> Deepgram transcription
    step 3: single LLM call, no framework
    step 5: LangGraph router -> specialist agents
    step 6: approval gate before any real action

    Nothing in here is allowed to raise. BackgroundTasks has no error
    handling of its own -- an uncaught exception is printed to the server
    log by Starlette and then dropped, leaving the sender waiting for a
    reply that never arrives.
    """
    correlation_id = message["message_id"]
    log.info("processing_start", correlation_id=correlation_id)

    try:
        if message["type"] == "text":
            text = message["text"]
        elif message["type"] == "audio":
            media_url = await get_media_url(message["audio_id"])
            audio_bytes = await download_media(media_url)
            text = await transcribe_audio(audio_bytes)
            log.info("voice_transcribed", correlation_id=correlation_id, text=text[:80])
        else:
            # WhatsApp labels reactions, view-once, polls and similar as
            # "unsupported" -- the body never reaches us. Staying silent is
            # deliberate: replying meant a canned line went to whoever sent
            # it, including automated senders that never asked for a reply.
            log.info(
                "unhandled_type_ignored",
                correlation_id=correlation_id,
                type=message["type"],
            )
            return

        if not text:
            log.info("empty_text_ignored", correlation_id=correlation_id)
            return

        # Read history *before* storing this turn, or the incoming message
        # arrives twice: once inside history, once appended by ask_claude.
        history = await recent_messages(DEFAULT_TENANT_ID, message["from"])
        stored = await save_message(
            DEFAULT_TENANT_ID,
            message["from"],
            "user",
            text,
            wa_message_id=message["message_id"],
        )

        if not stored:
            log.info("duplicate_ignored", correlation_id=correlation_id)
            return

        log.info("history_loaded", correlation_id=correlation_id, turns=len(history))

        # Route to a specialist, or answer conversationally. Only the
        # "general" branch gets conversation history -- the agents answer
        # from their own source of truth (documents, calendar), and
        # replaying chat history into them just adds noise and tokens.
        intent = await classify_intent(text)
        if intent == "drive":
            reply = await handle_drive_request(text, message["from"])
        elif intent == "calendar":
            reply = await handle_calendar_request(text, message["from"])
        elif intent == "email":
            reply = await handle_email_request(text, message["from"])
        else:
            reply = await ask_claude(text, history=history)
    except Exception:
        log.exception("processing_failed", correlation_id=correlation_id)
        reply = "Sorry -- something went wrong on my end. Please try again."

    try:
        await send_text_message(to=message["from"], body=reply)
    except Exception:
        # The send is the one step with no fallback: if the Graph API
        # rejects us there is no second channel to apologise over, so log
        # it against the correlation_id and give up on this message.
        log.exception("send_failed", correlation_id=correlation_id)
        return

    # Only after the reply actually reached the user -- storing a turn we
    # failed to deliver would leave Claude referring back to something the
    # user never saw.
    try:
        await save_message(DEFAULT_TENANT_ID, message["from"], "assistant", reply)
    except Exception:
        log.exception("save_reply_failed", correlation_id=correlation_id)

    log.info("processing_done", correlation_id=correlation_id)
