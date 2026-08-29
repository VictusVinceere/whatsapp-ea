# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Dependencies are managed with **uv**, not pip. `uv run` executes inside `.venv` without activating it.

```bash
uv sync                            # install deps from uv.lock
uv add <package>                   # add a dep (updates pyproject.toml + uv.lock)
uv run python -m app.db            # create/refresh the schema; safe to re-run
uv run pytest                      # run the suite; needs Postgres up (see below)
uv run uvicorn main:app --reload   # run the API on :8000
ngrok http 8000                    # second terminal; expose /webhook to Meta
```

`uv.lock` is deliberately committed (same reasoning as `package-lock.json`).

Register the ngrok URL + `/webhook` and `WHATSAPP_VERIFY_TOKEN` in the Meta developer dashboard so `GET /webhook` can complete Meta's one-time verification handshake.

**Tests are real and they hit a real database.** `pytest` + `pytest-asyncio` are declared in `[dependency-groups] dev`; run with `uv run pytest`. `[tool.pytest.ini_options]` sets `asyncio_mode = "auto"` (async tests need no decorator) and pins **both** fixture and test loop scopes to `session` — the SQLAlchemy engine is created at import time and its pooled connections bind to the loop that first used them, so a fresh loop per test hands the next test a dead connection ("another operation is in progress"). Don't loosen those loop scopes. `pythonpath = ["."]` exists because `tests/` sits beside `app/` rather than inside an installed package.

`tests/conftest.py` deliberately does **not** mock Postgres: every invariant worth protecting here is enforced *by the database* — a UNIQUE constraint, the conditional `UPDATE`, a `COALESCE` — so mocking it would mock away the thing under test. That means the suite needs the container running first:

```bash
docker start whatsapp-ea-db
uv run python -m app.db
uv run pytest
```

There is no linter or formatter configured — `ruff` is **not** a declared dependency and has no config block, so don't assume `ruff` commands exist (a stray `.ruff_cache/` is from an ad-hoc run, not project setup).

Postgres runs on **5433** (5432 is taken by another project's container) — `docker run --name whatsapp-ea-db -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=whatsapp_ea -p 5433:5432 -d pgvector/pgvector:pg16`. The schema is not auto-migrated on startup; `uv run python -m app.db` applies every statement in `SCHEMA_STATEMENTS`. All statements are `IF NOT EXISTS`, so it only ever creates — the first `ALTER` requirement means switching to Alembic.

## Architecture

A WhatsApp-fronted executive assistant. One FastAPI app; `main.py` wires `configure_logging()` and mounts the webhook router.

**Request flow.** `POST /webhook` parses the payload, acks immediately, and hands off to `process_message` via FastAPI `BackgroundTasks` — Meta times out if you block on an LLM call inside the request. `process_message` in `app/webhook.py` is the pipeline that grows over the build: branch on message type → transcribe audio → LLM → send reply.

**Layers.** `app/whatsapp.py` is the only place that talks to the Graph API (send, plus the two-step `get_media_url` → `download_media` needed because voice notes arrive as a media id, not bytes). `app/transcription.py` wraps Deepgram, `app/llm.py` wraps Anthropic. `app/router_agent.py` classifies intent and hands to one of `app/agents/{calendar,email,drive}_agent.py`. `app/db.py` holds the async engine plus raw SQL constants — there are no ORM models.

**Current state: build steps 1–3 work end to end against live WhatsApp** (text in, Deepgram transcription for voice notes, Claude reply out). Everything after that is scaffolding: the router returns a hardcoded `"general"`, all three agents return "not wired up yet", and `process_message` calls `ask_claude` directly. **`app/db.py`, `app/router_agent.py` and `app/agents/` are imported by nobody** — the `pending_actions` and `messages` tables exist but no code reads or writes them. Treat the numbered build order in `README.md` as binding: text loop → voice notes → single LLM call → Postgres → calendar agent → approval gate → router + remaining agents → deploy. Build `calendar_agent.py` for real before email/drive; it has the simplest OAuth surface and establishes the pattern the other two copy.

**Approval gate.** Any agent action with real-world effect is meant to land in `pending_actions` as `status='pending'` and only execute after `CONFIRM_ACTION_SQL` — a conditional `UPDATE ... WHERE status = 'pending'` that updates zero rows on a double-confirm race. Preserve that shape; don't replace it with a read-then-write.

## Conventions

- **Never block the event loop.** Every outgoing call uses `httpx` async, never `requests`; Postgres uses `asyncpg`, never `psycopg2`. Deepgram's v5 SDK is synchronous, so `transcription.py` wraps it in `asyncio.to_thread` — do the same for any other blocking client.
- **Config goes through `app.config.settings`.** Don't call `os.getenv()` anywhere else.
- **Log with structlog and thread `correlation_id`** (the WhatsApp `message_id`) through every log line for a message, so the full router → tool → approval → send chain is filterable. Output is JSON.
- `parse_incoming_message` returns `None` for non-message webhooks (status callbacks); callers must handle that.
- **Four guards in `receive_webhook`, all load-bearing.** Meta's webhook is app-wide, so every number on the business account arrives at the same URL: messages not addressed to `WHATSAPP_PHONE_NUMBER_ID` are dropped, and `WHATSAPP_ALLOWED_SENDERS` (when set) restricts who gets answered — both exist because a dev build once auto-replied 14 times to a stranger on a live business line. Meta also redelivers every message, so `_SEEN_MESSAGE_IDS` dedupes at the ack, before the LLM call. And `type: "unsupported"` (reactions, polls, view-once) carries no body — those get silence, never a canned reply.
- `app/llm.py` imports `anthropic` directly, but `pyproject.toml` only declares `langchain-anthropic` — the SDK is available transitively. Add `anthropic` explicitly if you touch that file's dependency surface.
