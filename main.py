from contextlib import asynccontextmanager

from fastapi import FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.graph import build_graph, checkpointer_dsn, set_graph
from app.logging_config import configure_logging
from app.oauth import router as oauth_router
from app.webhook import router as webhook_router

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Compile the graph once, for the life of the process.

    The checkpointer owns a connection pool, so it can't be created
    per-request -- and its context manager has to stay open for as long as
    the graph is used, which is what this handler is for. cp.setup()
    creates LangGraph's own checkpoint tables; it's idempotent.
    """
    async with AsyncPostgresSaver.from_conn_string(checkpointer_dsn()) as checkpointer:
        await checkpointer.setup()
        set_graph(build_graph().compile(checkpointer=checkpointer))
        yield


app = FastAPI(title="WhatsApp AI Executive Assistant", lifespan=lifespan)
app.include_router(webhook_router)
app.include_router(oauth_router)


@app.get("/")
async def health_check():
    return {"status": "ok"}
