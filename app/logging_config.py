"""
One correlation_id per WhatsApp message, threaded through every log
line for that message, so you can filter and see the full decision
chain: router choice -> tool calls -> approval -> send.
"""
import structlog


def configure_logging():
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ]
    )
