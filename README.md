# WhatsApp AI Executive Assistant

## Setup

Install uv once, if you don't have it:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, from the project root:
```bash
uv sync                 # creates .venv, installs deps, writes uv.lock
cp .env.example .env    # fill in real values -- see WhatsApp setup below
```

`uv run` executes inside the project's venv without you needing to
activate it manually. Adding a new dependency later: `uv add <package>`
(updates pyproject.toml + uv.lock together, same idea as `npm install --save`).

### Postgres

Port 5433, not 5432 — 5432 is usually taken by another project's
container. The `pgvector` image is plain Postgres 16 with the vector
extension available, so step 7's RAG work doesn't need a new image later.

```bash
docker run --name whatsapp-ea-db \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=whatsapp_ea \
  -p 5433:5432 \
  -d pgvector/pgvector:pg16

uv run python -m app.db      # creates every table + index; safe to re-run
```

That `docker run` is once. After a reboot the container is stopped, not
gone — `docker start whatsapp-ea-db`. Running `docker run` a second time
fails on the name, and your data lives in that container, so don't
`docker rm` it unless you mean to lose the tables. A connection refused
on 5433 almost always means it's just stopped.

`app/db.py` holds the whole schema as SQL constants listed in
`SCHEMA_STATEMENTS`. To add a table: write the constant, append it to the
tuple, re-run. Every statement is `IF NOT EXISTS`, which is what makes
re-running safe — and also the ceiling of this approach. It creates, it
never alters. The first time you need to change an existing column,
switch to Alembic instead of editing these.

Check `DATABASE_URL` points at **5433**. Left at 5432 it will silently
connect to whatever other Postgres you have running and start writing
tables into it.

### Run it

```bash
uv run uvicorn main:app --reload
```

In a second terminal:
```bash
ngrok http 8000
```

## WhatsApp setup

The fiddliest part of this project, and none of it is Python. Budget an
afternoon the first time.

### 1. Pick a sender number

Two kinds, and the difference decides everything else:

| | Meta test number | Your own business number |
|---|---|---|
| Cost | free | per-conversation |
| Setup | exists already | business verification |
| Recipients | **allow-list only, max 5** | anyone (within the rules below) |
| Ban risk | none | real |

Use the **test number** for development. Point a dev app at a live
business line and real customers get answered by half-built code — see
the safety rails below, which exist because that happened.

Find both numbers under **developers.facebook.com → your app →
WhatsApp → API Setup**. `WHATSAPP_PHONE_NUMBER_ID` is the numeric **Phone
number ID** on that page, not the phone number itself.

For the test number, add your own phone under the **To** dropdown →
*Manage phone number list*. Meta sends it a code. Without this, every
send fails with error **131030**.

### 2. Get a permanent token

The token on the API Setup page expires in **24 hours**. When sends start
failing with error **190**, that's what happened. For one that doesn't:

1. business.facebook.com → **Business Settings**
2. **Users → System Users → Add** (role: Admin)
3. **Assign Assets** → your WhatsApp app → full control
4. **Generate New Token** → expiry **Never**
5. Tick `whatsapp_business_messaging` and `whatsapp_business_management`

Shown once. Verify what you got:

```bash
curl -s "https://graph.facebook.com/v19.0/debug_token?input_token=$T&access_token=$T" \
  | python3 -m json.tool     # expires_at: 0 means permanent
```

### 3. Webhook

With ngrok running, in **WhatsApp → Configuration**:

- **Callback URL**: `https://<your-ngrok>.ngrok-free.app/webhook`
- **Verify token**: whatever `WHATSAPP_VERIFY_TOKEN` is set to

Meta immediately calls `GET /webhook` to verify. Then — the step everyone
misses — click **Manage** next to *Webhook fields* and subscribe to
**`messages`**. Without it Meta verifies your URL happily and never sends
you anything.

The free ngrok URL changes on every restart, so this gets re-pasted each
session. **Clear the callback URL when you stop**, or the number spends
the night forwarding messages into a dead tunnel.

### 4. Rules that look like bugs

- **The webhook is app-wide, not per-number.** Every number on the
  business account delivers to the same URL. `receive_webhook` filters on
  `to_phone_number_id` for exactly this reason.
- **24-hour window.** You can only send free-form text to someone who
  messaged you in the last 24 hours. Outside it you need an approved
  template. Customer-initiated conversations open the window.
- **Meta redelivers.** The same message arrives more than once —
  observed twice each, consistently. Dedupe on `message_id` or you reply
  twice to everything.
- **`type: "unsupported"`** covers reactions, view-once, polls. The body
  never reaches you, so there is nothing to answer. Stay silent.

### 5. Dev safety rails

Both live in `receive_webhook`, both exist because of a real incident
where a dev build sent 14 automated replies to a stranger:

- `WHATSAPP_PHONE_NUMBER_ID` — only messages sent *to* this number are
  processed; everything else is dropped as `foreign_number_ignored`.
- `WHATSAPP_ALLOWED_SENDERS` — comma-separated numbers the bot may
  answer. Anything else is `sender_not_allowed`. **Empty means answer
  everyone**, which is production behaviour. Set it whenever you test
  against a number real people also message.

## Build order — do not skip ahead

1. ~~**Text loop**~~ — done. Real message in, Claude reply out.
2. ~~**Voice notes**~~ — done. `get_media_url` → `download_media` →
   Deepgram (not Whisper; see `app/transcription.py`).
3. ~~**Single LLM call**~~ — done. `ask_claude` in `app/llm.py`, direct
   Anthropic SDK, no framework.
4. **Postgres** — schema done (`pending_actions`, `messages`), *no code
   uses it yet*. `db.py` is imported by nobody. Next: `save_message` /
   `recent_messages`, then wire into `process_message` so the assistant
   remembers a conversation.
5. **One specialist agent**: build `calendar_agent.py` for real —
   OAuth + a real Calendar read call.
6. **Approval gate**: wrap the calendar agent's actions in
   `pending_actions`, using `CONFIRM_ACTION_SQL`'s conditional update
   to avoid a double-confirm race.
7. **Router + remaining agents**: only once 5+6 work cleanly, wire up
   `router_agent.py` for real and copy the pattern to email/drive.
   Drive is the RAG one — needs `CREATE EXTENSION vector`, a chunks
   table, and an embedding provider. **Anthropic has no embeddings
   API**; Voyage AI is the recommended pairing.
8. **Deploy** (last): Docker → DigitalOcean K8s → Terraform. Not before
   step 7 works locally end to end.

## Notes

- Every outgoing call uses `httpx` (async), never `requests`
  (blocking — freezes the event loop for every other in-flight request).
- `correlation_id` = the WhatsApp `message_id`, threaded through every
  log line for a given message so a full decision trace is filterable.
- The approval gate's `CONFIRM_ACTION_SQL` is the same
  `attribute_not_exists(slot_id)` pattern from the DynamoDB scheduler,
  applied to Postgres — worth mentioning as-is in interviews. The point
  is that the check and the write are one atomic statement, so there is
  no gap for a second confirm to slip through, unlike read-then-write.
- `messages` carries **two** ids: `tenant_id` (whose assistant) and
  `conversation_id` (who is talking to it). Multi-tenancy is far cheaper
  to design in now than to retrofit — and a similarity search that
  forgets to filter by tenant is a data breach, not a bug.
- `langgraph` and `langchain-anthropic` are declared in `pyproject.toml`
  but imported nowhere. They are a plan, not a dependency. `app/llm.py`
  imports `anthropic` directly, which is available transitively.
