"""
Thin wrapper around the WhatsApp Business Cloud API.
Build order step 4: get send_text_message + parse_incoming_message
working before anything else in this project.
"""
import httpx
from app.config import settings

GRAPH_API_URL = f"https://graph.facebook.com/v19.0/{settings.whatsapp_phone_number_id}/messages"


async def send_text_message(to: str, body: str) -> dict:
    """Send a plain text WhatsApp message. Called after the approval
    gate confirms an action, or for direct conversational replies."""
    headers = {"Authorization": f"Bearer {settings.whatsapp_access_token}"}
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": body},
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(GRAPH_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()


async def get_media_url(media_id: str) -> str:
    """Voice notes arrive as a media_id, not raw audio. Fetch the
    temporary download URL first, then download separately."""
    headers = {"Authorization": f"Bearer {settings.whatsapp_access_token}"}
    url = f"https://graph.facebook.com/v19.0/{media_id}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()["url"]


async def download_media(media_url: str) -> bytes:
    """Download the actual audio bytes for Whisper transcription."""
    headers = {"Authorization": f"Bearer {settings.whatsapp_access_token}"}
    async with httpx.AsyncClient() as client:
        response = await client.get(media_url, headers=headers)
        response.raise_for_status()
        return response.content


def parse_incoming_message(payload: dict) -> dict | None:
    """Extract the useful bits from WhatsApp's verbose webhook JSON.
    Returns None if this payload isn't a user message (e.g. a status
    update webhook, which WhatsApp also sends)."""
    try:
        entry = payload["entry"][0]["changes"][0]["value"]
        if "messages" not in entry:
            return None
        message = entry["messages"][0]
        return {
            "from": message["from"],
            "message_id": message["id"],
            "type": message["type"],  # "text" | "audio" | ...
            "text": message.get("text", {}).get("body"),
            "audio_id": message.get("audio", {}).get("id"),
        }
    except (KeyError, IndexError):
        return None
