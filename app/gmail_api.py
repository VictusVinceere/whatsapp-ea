"""
Gmail reads and sends.

Same shape as calendar_api: thin httpx wrapper, takes an access token
rather than fetching one, so the caller owns refresh.

Note the scopes are not nested. gmail.readonly cannot send and gmail.send
cannot read -- unlike a filesystem, where write usually implies read.
Both are needed, and both are "restricted" scopes in Google's model, so a
production app needs verification before anyone outside the test-user
list can consent.
"""

import base64
from email.message import EmailMessage

import httpx
import structlog

log = structlog.get_logger()

BASE = "https://gmail.googleapis.com/gmail/v1/users/me"

# Which headers to pull back. Asking for metadata rather than the full
# message keeps the response small -- a thread with attachments is
# megabytes, and none of it helps answer "who emailed me?".
WANTED_HEADERS = ("From", "Subject", "Date")


async def list_recent(
    access_token: str,
    max_results: int = 5,
    query: str = "in:inbox",
) -> list[dict]:
    """Recent messages as {from, subject, date, snippet}.

    Two round trips per message is unavoidable on this API: list returns
    ids only, so each one needs a get. Fine at 5; batch if it grows.
    """
    headers = {"Authorization": f"Bearer {access_token}"}

    async with httpx.AsyncClient() as client:
        listing = await client.get(
            f"{BASE}/messages",
            headers=headers,
            params={"maxResults": max_results, "q": query},
        )
        listing.raise_for_status()
        ids = [m["id"] for m in listing.json().get("messages", [])]

        messages = []
        for message_id in ids:
            detail = await client.get(
                f"{BASE}/messages/{message_id}",
                headers=headers,
                params={
                    "format": "metadata",
                    "metadataHeaders": WANTED_HEADERS,
                },
            )
            detail.raise_for_status()
            body = detail.json()
            fields = {
                h["name"]: h["value"]
                for h in body.get("payload", {}).get("headers", [])
            }
            messages.append(
                {
                    "from": fields.get("From", "?"),
                    "subject": fields.get("Subject", "(no subject)"),
                    "date": fields.get("Date", "?"),
                    "snippet": body.get("snippet", ""),
                }
            )

    return messages


async def send_message(
    access_token: str, to: str, subject: str, body: str
) -> dict:
    """Send an email. The only call here with real-world effect.

    Gmail wants a complete RFC 2822 message, base64url encoded -- not a
    JSON object of fields. EmailMessage builds and escapes it correctly;
    hand-assembling the headers breaks on the first non-ASCII subject.
    """
    message = EmailMessage()
    message["To"] = to
    message["Subject"] = subject
    message.set_content(body)

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE}/messages/send",
            headers={"Authorization": f"Bearer {access_token}"},
            json={"raw": raw},
        )
        response.raise_for_status()
        return response.json()


def describe_messages(messages: list[dict]) -> str:
    """Flatten for a prompt."""
    if not messages:
        return "(no matching email)"
    return "\n".join(
        f"- from {m['from']} | {m['subject']} | {m['date']}\n  {m['snippet'][:160]}"
        for m in messages
    )
