"""
Google API access. Shared by every Google-backed agent -- calendar today,
Drive when the RAG agent lands -- because they all need the same thing: an
access token that hasn't expired.

Access tokens live about an hour. The refresh token doesn't expire, so the
pattern is: check the clock, refresh if needed, hand back something usable.
Callers never touch the token table directly.
"""

from datetime import datetime, timedelta, timezone

import httpx
import structlog

from app.config import settings
from app.db import get_google_tokens, save_google_tokens

log = structlog.get_logger()

GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"

# Refresh this far before the token actually dies. Without a margin, a
# token with three seconds left passes the check and then expires
# mid-request -- an intermittent 401 that is miserable to reproduce.
EXPIRY_MARGIN = timedelta(seconds=60)


async def valid_access_token(tenant_id: str, conversation_id: str) -> str | None:
    """A usable access token, refreshing first if it's close to expiry.

    Returns None when this conversation has never connected a Google
    account, or when the refresh was rejected -- both mean "ask the user
    to connect again", and neither is an error worth raising.
    """
    tokens = await get_google_tokens(tenant_id, conversation_id)
    if tokens is None:
        return None

    if tokens["expires_at"] > datetime.now(timezone.utc) + EXPIRY_MARGIN:
        return tokens["access_token"]

    refresh_token = tokens["refresh_token"]
    if not refresh_token:
        # Nothing to refresh with. Happens if the first authorization was
        # made without access_type=offline / prompt=consent.
        log.warning("google_no_refresh_token", conversation_id=conversation_id)
        return None

    async with httpx.AsyncClient() as client:
        response = await client.post(
            GOOGLE_TOKEN_URL,
            data={
                "refresh_token": refresh_token,
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "grant_type": "refresh_token",
            },
        )

    if response.status_code != 200:
        # A revoked or expired refresh token gives 400 invalid_grant. That
        # is a real state -- the user unlinked the app -- not a bug, so it
        # returns None rather than raising.
        log.warning(
            "google_refresh_failed",
            conversation_id=conversation_id,
            status=response.status_code,
            body=response.text[:200],
        )
        return None

    refreshed = response.json()

    await save_google_tokens(
        tenant_id,
        conversation_id,
        refreshed["access_token"],
        # Refresh responses never include a refresh_token. Passing None is
        # deliberate: save_google_tokens COALESCEs it, keeping the stored one.
        None,
        refreshed["expires_in"],
        refreshed.get("scope", tokens["scope"]),
    )

    log.info("google_token_refreshed", conversation_id=conversation_id)
    return refreshed["access_token"]
