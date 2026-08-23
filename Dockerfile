# Two stages: build deps once, ship only what runs.
FROM python:3.13-slim AS builder

# uv, same tool as local development, so the lockfile is honoured exactly.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies before source. Docker caches layers, and this one only
# rebuilds when the lockfile changes -- editing app code then costs
# seconds rather than re-resolving every package.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app
COPY main.py ./
RUN uv sync --frozen --no-dev


FROM python:3.13-slim

# Two system libraries no Python package declares, because both are
# expected to exist already -- they do on a developer laptop, which is why
# this only breaks once it is containerised:
#   libgomp1 -- onnxruntime (under fastembed) links against OpenMP
#   libpq5   -- psycopg 3, which LangGraph's Postgres checkpointer uses
# Without libpq5 the app dies at import with "no pq wrapper available",
# which names psycopg but not the missing C library.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libpq5 curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root. The app writes nothing outside the model cache, so it has no
# reason to run with more privilege than that.
RUN useradd --create-home --uid 1000 assistant
WORKDIR /app

COPY --from=builder --chown=assistant:assistant /app/.venv /app/.venv
COPY --chown=assistant:assistant app ./app
COPY --chown=assistant:assistant main.py ./

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/home/assistant/.cache/fastembed

USER assistant

# Download the embedding model at build time. Otherwise the first message
# after every deploy pays a ~200MB download, and a container with no
# persistent volume pays it again on every restart.
# The model name is read from app/rag.py rather than repeated here, so
# the two cannot drift. A placeholder DATABASE_URL is needed because
# app/db.py builds its engine at import time, and importing app.rag pulls
# it in -- nothing connects during the build, it just has to parse.
RUN DATABASE_URL="postgresql+asyncpg://build:build@localhost/build" python -c "\
from fastembed import TextEmbedding; \
from app.rag import EMBED_MODEL; \
TextEmbedding(model_name=EMBED_MODEL); \
print('embedding model cached')"

EXPOSE 8000

# Hits the app's own health route, so an unhealthy container is one that
# cannot serve requests -- not merely one whose process is alive.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/ || exit 1

# Ensure the schema before serving. app.db is CREATE TABLE IF NOT EXISTS
# throughout, so this is safe on every start and makes the container
# self-sufficient against an empty database -- otherwise a fresh deploy
# starts happily and then fails on the first real message with
# UndefinedTableError, which surfaces to the user as a generic error reply.
# exec so uvicorn becomes PID 1 and receives stop signals directly.
CMD ["sh", "-c", "python -m app.db && exec uvicorn main:app --host 0.0.0.0 --port 8000"]
