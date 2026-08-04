"""
Build order step 4 (part 2): GET verifies the webhook with Meta once.
POST receives every incoming message. Get text messages round-tripping
(receive -> print -> send canned reply) before adding anything else.
"""

import pathlib

import structlog
from fastapi import APIRouter, BackgroundTasks, Query, Request, Response

from app.config import settings
from app.db import DEFAULT_TENANT_ID, recent_messages, save_message
from app.graph import run_graph
from app.ingest import extract_text
from app.rag import ingest_document
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


async def ingest_whatsapp_document(message: dict, correlation_id: str) -> str:
    """Download a forwarded file, extract its text, and index it.

    source_id is the WhatsApp media id, so re-forwarding the same file
    replaces its chunks instead of duplicating them -- the same guarantee
    app.ingest gets from the file path.
    """
    name = message.get("document_name") or "document"
    media_url = await get_media_url(message["document_id"])
    data = await download_media(media_url)

    try:
        content = extract_text(data, name)
    except Exception:
        log.exception("document_extract_failed", correlation_id=correlation_id)
        return f"I couldn't read {name} -- PDF, Word and plain text only."

    if not content.strip():
        # Almost always a scanned PDF: pages of pixels, no text layer.
        return f"{name} has no text I can read -- it may be a scan, which needs OCR."

    chunks = await ingest_document(
        DEFAULT_TENANT_ID,
        source_id=f"whatsapp:{message['document_id']}",
        source_name=pathlib.Path(name).stem,
        content=content,
    )
    log.info("document_indexed", correlation_id=correlation_id, name=name, chunks=chunks)
    return f"Indexed {name} ({chunks} sections). Ask me anything about it."


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
    document: dict | None = None
    log.info("processing_start", correlation_id=correlation_id)

    try:
        if message["type"] == "text":
            text = message["text"]
        elif message["type"] == "document":
            # Forwarding a file indexes it, then answers the caption if
            # there was one. Handled here rather than as a tool because
            # the bytes arrive with the message -- the model never sees
            # them and has nothing to decide.
            reply = await ingest_whatsapp_document(message, correlation_id)
            caption = message.get("document_caption")
            filename = message.get("document_name") or "the document"

            # Passed to the graph so save_to_drive can fetch the bytes
            # without the model ever seeing a media id.
            document = {
                "media_id": message["document_id"],
                "filename": filename,
                "mime_type": message.get("document_mime_type"),
            }

            # Indexing keeps the text; the original file is otherwise
            # discarded. Offer to keep it, but ask -- an upload is a write
            # with real-world effect, so it goes through the same gate as
            # sending an email.
            text = (
                f'I just sent you a file called "{filename}". '
                f"It is already indexed and searchable. "
                + (
                    caption
                    # An instruction, not "offer to" -- the model reads
                    # that as "ask first", which duplicates the approval
                    # gate and then never calls the tool at all.
                    or "Save it to my Google Drive so the original is kept."
                )
            )

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

        # One call now handles routing, the specialists, and the approval
        # gate. thread_id is the sender's number, so each contact gets
        # their own graph state -- including a half-finished approval that
        # survives a restart and resumes when they eventually reply.
        reply = await run_graph(
            message["from"], text, history=history, document=document
        )
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
