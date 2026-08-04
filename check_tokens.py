"""Scratch check for the google_tokens upsert. Not part of the app.

Run:  uv run python check_tokens.py

Proves two things:
  1. a second auth replaces the access token
  2. a second auth does NOT wipe the refresh token, because Google only
     sends that on the first authorization
"""

import asyncio

from sqlalchemy import text

from app.db import (
    DEFAULT_TENANT_ID as T,
    async_session,
    get_google_tokens,
    save_google_tokens,
)

C = "test-conversation"


async def main() -> None:
    # Start clean so the script gives the same answer every run.
    async with async_session() as s:
        await s.execute(
            text("DELETE FROM google_tokens WHERE conversation_id = :c"), {"c": C}
        )
        await s.commit()

    # First authorization: Google returns both tokens.
    await save_google_tokens(T, C, "access-1", "refresh-1", 3600, "calendar.readonly")
    row = await get_google_tokens(T, C)
    assert row is not None, "nothing stored"
    assert row["access_token"] == "access-1", row
    assert row["refresh_token"] == "refresh-1", row

    # Re-authorization: Google omits refresh_token, so we pass None.
    await save_google_tokens(T, C, "access-2", None, 3600, "calendar.readonly")
    row = await get_google_tokens(T, C)
    assert row["access_token"] == "access-2", f"access token not replaced: {row}"
    assert row["refresh_token"] == "refresh-1", f"COALESCE failed, wiped it: {row}"

    print("all assertions passed")


if __name__ == "__main__":
    asyncio.run(main())
