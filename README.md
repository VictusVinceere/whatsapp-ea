# WhatsApp AI Executive Assistant

## Setup

Install uv once, if you don't have it:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, from the project root:
```bash
uv sync                 # creates .venv, installs deps, writes uv.lock
cp .env.example .env    # fill in real values
uv run uvicorn main:app --reload
```

`uv run` executes inside the project's venv without you needing to
activate it manually. Adding a new dependency later: `uv add <package>`
(updates pyproject.toml + uv.lock together, same idea as `npm install --save`).

In a second terminal:
```bash
ngrok http 8000
```
Paste the ngrok URL + `/webhook` into the Meta developer dashboard's
WhatsApp webhook config, along with your WHATSAPP_VERIFY_TOKEN.

## Build order — do not skip ahead

1. **Text loop**: send yourself a WhatsApp message, confirm it hits
   `POST /webhook`, confirm `process_message` sends a canned reply back.
   This is the only milestone that matters today — nothing else works
   until this does.
2. **Voice notes**: branch on `message["type"] == "audio"`, use
   `get_media_url` + `download_media` from `app/whatsapp.py`, pipe to
   Whisper.
3. **Single LLM call**: replace the canned reply in `process_message`
   with a real call to Claude/GPT. No framework yet.
4. **Postgres**: `docker run` a local Postgres, run
   `CREATE_PENDING_ACTIONS_TABLE` from `app/db.py`.
5. **One specialist agent**: build `calendar_agent.py` for real —
   OAuth + a real Calendar read call.
6. **Approval gate**: wrap the calendar agent's actions in
   `pending_actions`, using `CONFIRM_ACTION_SQL`'s conditional update
   to avoid a double-confirm race.
7. **Router + remaining agents**: only once 5+6 work cleanly, wire up
   `router_agent.py` for real and copy the pattern to email/drive.
8. **Deploy** (last): Docker → DigitalOcean K8s → Terraform. Not before
   step 7 works locally end to end.

## Notes

- Every outgoing call uses `httpx` (async), never `requests`
  (blocking — freezes the event loop for every other in-flight request).
- `correlation_id` = the WhatsApp `message_id`, threaded through every
  log line for a given message so a full decision trace is filterable.
- The approval gate's `CONFIRM_ACTION_SQL` is the same
  `attribute_not_exists(slot_id)` pattern from the DynamoDB scheduler,
  applied to Postgres — worth mentioning as-is in interviews.
