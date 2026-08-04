"""
Build order step 5b: connect a Google account so the calendar agent has
something to read.

Two routes. /oauth/start sends the user to Google; /oauth/callback
receives them coming back with a code and trades it for tokens.
"""

import hashlib
import hmac
from urllib.parse import urlencode

import httpx
import structlog
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse

from app.config import settings
from app.db import DEFAULT_TENANT_ID, save_google_tokens

log = structlog.get_logger()
router = APIRouter()

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"

# Space-separated, and each one is narrower than its name suggests:
#   calendar.events  events only -- NOT GET /calendars/primary
#   gmail.readonly   read only, cannot send
#   gmail.send       send only, cannot read
# Read and write are siblings here, not nested, so both Gmail scopes are
# needed to do both jobs. Widening this list forces every connected user
# to re-consent, so add a scope when an agent needs it, not before.
#   drive.readonly   read existing files, including Google Docs export
#   drive.file       create files -- narrow by design: it grants access
#                    only to files this app creates, not the whole Drive,
#                    so an upload feature cannot become a read-everything
#                    feature by accident
SCOPES = " ".join(
    (
        "https://www.googleapis.com/auth/calendar.events",
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.send",
        "https://www.googleapis.com/auth/drive.readonly",
        "https://www.googleapis.com/auth/drive.file",
    )
)


def _sign(value: str) -> str:
    return hmac.new(
        settings.app_secret_key.encode(), value.encode(), hashlib.sha256
    ).hexdigest()


def _make_state(conversation_id: str) -> str:
    """`<conversation_id>.<signature>` -- readable, and unforgeable.

    The callback is a bare GET from Google's servers with no session, so
    the conversation id has to survive the round trip. Sending it in the
    clear would be enough to *carry* it, but then anyone could call
    /oauth/callback?state=<someone-else's-number> and attach their own
    Google account to that person's conversation. The signature is what
    makes the value tamper-evident.
    """
    return f"{conversation_id}.{_sign(conversation_id)}"


def _read_state(state: str) -> str | None:
    """Recover the conversation id, or None if the signature doesn't hold."""
    conversation_id, _, signature = state.rpartition(".")
    if not conversation_id:
        return None
    # compare_digest, not ==, so the comparison takes the same time however
    # wrong the guess is. A plain == leaks where the mismatch happened.
    if not hmac.compare_digest(signature, _sign(conversation_id)):
        return None
    return conversation_id


@router.get("/oauth/start")
async def oauth_start(conversation_id: str = Query(...)):
    """Redirect the user to Google's consent screen.

    Visited as /oauth/start?conversation_id=15550001234 -- eventually the
    bot will WhatsApp this link to whoever needs to connect an account.
    """
    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": settings.google_redirect_uri,
        "response_type": "code",
        "scope": SCOPES,
        # Without BOTH of these Google returns no refresh token, and the
        # connection silently dies an hour later.
        "access_type": "offline",
        "prompt": "consent",
        # Google returns this untouched on the callback.
        "state": _make_state(conversation_id),
    }
    return RedirectResponse(f"{GOOGLE_AUTH_URL}?{urlencode(params)}")


@router.get("/oauth/callback")
async def oauth_callback(code: str = Query(...), state: str = Query(...)):
    """Google sends the user back here with a one-time code."""
    conversation_id = _read_state(state)
    if conversation_id is None:
        # This URL is public; anyone can call it with an invented state.
        log.warning("oauth_state_invalid")
        raise HTTPException(status_code=400, detail="invalid state")

    # data=, not json= -- the OAuth2 spec says form-encoded, and Google's
    # token endpoint rejects a JSON body.
    async with httpx.AsyncClient() as client:
        response = await client.post(
            GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "redirect_uri": settings.google_redirect_uri,
                "grant_type": "authorization_code",
            },
        )
        response.raise_for_status()
        tokens = response.json()

    await save_google_tokens(
        DEFAULT_TENANT_ID,
        conversation_id,
        tokens["access_token"],
        # .get(), not [] -- absent on every authorization after the first.
        # save_google_tokens COALESCEs it so the stored one survives.
        tokens.get("refresh_token"),
        tokens["expires_in"],
        tokens.get("scope", SCOPES),
    )

    log.info(
        "google_connected",
        conversation_id=conversation_id,
        got_refresh_token=bool(tokens.get("refresh_token")),
    )
    return {"status": "connected", "conversation_id": conversation_id}
