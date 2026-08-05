"""
Central place to load and validate environment variables.
Everything else in the app should import settings from here,
never call os.getenv() directly elsewhere.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    whatsapp_access_token: str = os.getenv("WHATSAPP_ACCESS_TOKEN", "")
    whatsapp_phone_number_id: str = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
    whatsapp_verify_token: str = os.getenv("WHATSAPP_VERIFY_TOKEN", "")

    # Dev safety net: comma-separated sender numbers the bot is allowed to
    # answer. Empty means "answer everyone" (production behaviour). Set it
    # while testing against a number that real customers also message, so a
    # customer never gets a reply from a half-built assistant.
    whatsapp_allowed_senders: set[str] = {
        n.strip() for n in os.getenv("WHATSAPP_ALLOWED_SENDERS", "").split(",") if n.strip()
    }

    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    # Fallback provider, used only when Anthropic is unusable -- no
    # credit, bad key, rate limited, down. Unset means "no fallback",
    # which is a valid configuration, not an error.
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    deepgram_api_key: str = os.getenv("DEEPGRAM_API_KEY", "")

    database_url: str = os.getenv("DATABASE_URL", "")

    # DEBUG additionally logs message content -- transcripts of voice
    # notes, retrieved chunks. Useful when a reply makes no sense and you
    # need to see what the model actually received; not something to
    # leave on against a live line, where it ships customer speech to
    # whatever collects your logs.
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()

    google_client_id: str = os.getenv("GOOGLE_CLIENT_ID", "")
    google_client_secret: str = os.getenv("GOOGLE_CLIENT_SECRET", "")
    google_redirect_uri: str = os.getenv("GOOGLE_REDIRECT_URI", "")

    # Signs the OAuth `state` parameter so /oauth/callback can tell a real
    # round trip from a forged one. Ours alone -- never sent anywhere.
    # Rotating it invalidates any in-flight authorization, nothing stored.
    app_secret_key: str = os.getenv("APP_SECRET_KEY", "")


settings = Settings()
