from fastapi import FastAPI
from app.webhook import router as webhook_router
from app.logging_config import configure_logging

configure_logging()

app = FastAPI(title="WhatsApp AI Executive Assistant")
app.include_router(webhook_router)


@app.get("/")
async def health_check():
    return {"status": "ok"}
