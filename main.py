from fastapi import FastAPI

from app.logging_config import configure_logging
from app.oauth import router as oauth_router
from app.webhook import router as webhook_router

configure_logging()

app = FastAPI(title="WhatsApp AI Executive Assistant")
app.include_router(webhook_router)
app.include_router(oauth_router)


@app.get("/")
async def health_check():
    return {"status": "ok"}
