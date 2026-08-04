"""
Google Calendar reads and writes.

Thin wrapper on the REST API rather than google-api-python-client: that
library is synchronous and would need wrapping in a thread anyway, and we
only need three calls. httpx async keeps it consistent with the rest of
the app.

Every function takes an access token rather than fetching one, so the
caller decides when to refresh -- see app/google.py.
"""

from datetime import datetime, timedelta, timezone

import httpx
import structlog

log = structlog.get_logger()

BASE = "https://www.googleapis.com/calendar/v3"


async def calendar_timezone(access_token: str) -> str:
    """The primary calendar's timezone, e.g. 'Asia/Samarkand'.

    Needed because "3pm tomorrow" is meaningless without one, and the
    user's timezone lives in their Google account, not our database.

    Read from the *events* endpoint rather than GET /calendars/primary,
    which looks like the obvious place but is a different permission: the
    calendar.events scope covers events, not the calendar resource, so
    that call returns 403. events.list carries timeZone at the top level
    and is already within scope.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BASE}/calendars/primary/events",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"maxResults": 1},
        )
        response.raise_for_status()
        return response.json().get("timeZone", "UTC")


async def list_events(access_token: str, max_results: int = 5) -> list[dict]:
    """Upcoming events, soonest first.

    singleEvents expands recurring events into individual occurrences --
    without it a weekly standup comes back as one entry with a recurrence
    rule, which is not what anyone means by "what's next".
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BASE}/calendars/primary/events",
            headers={"Authorization": f"Bearer {access_token}"},
            params={
                "maxResults": max_results,
                "singleEvents": "true",
                "orderBy": "startTime",
                "timeMin": datetime.now(timezone.utc).isoformat(),
            },
        )
        response.raise_for_status()
        return response.json().get("items", [])


async def insert_event(
    access_token: str,
    summary: str,
    start_iso: str,
    duration_minutes: int,
    tz: str,
) -> dict:
    """Create an event. The only call here that changes anything.

    Guarded by the approval gate in graph.py -- nothing should reach this
    without a human having said yes.
    """
    start = datetime.fromisoformat(start_iso)
    end = start + timedelta(minutes=duration_minutes)

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE}/calendars/primary/events",
            headers={"Authorization": f"Bearer {access_token}"},
            json={
                "summary": summary,
                "start": {"dateTime": start.isoformat(), "timeZone": tz},
                "end": {"dateTime": end.isoformat(), "timeZone": tz},
            },
        )
        response.raise_for_status()
        return response.json()


def describe_events(events: list[dict]) -> str:
    """Flatten the API's response into something worth putting in a prompt.

    Google returns dateTime for timed events and date for all-day ones, so
    both shapes have to be handled or all-day entries vanish.
    """
    if not events:
        return "(no upcoming events)"

    lines = []
    for event in events:
        start = event.get("start", {})
        when = start.get("dateTime") or start.get("date", "?")
        lines.append(f"- {when}: {event.get('summary', '(no title)')}")
    return "\n".join(lines)
