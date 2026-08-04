"""
One correlation_id per WhatsApp message, threaded through every log
line for that message, so you can filter and see the full decision
chain: router choice -> tool calls -> approval -> send.

Two things this file has to get right, both of which only start to
matter once the logs leave your terminal and land in something that
queries them:

1. *Every* line is JSON, including uvicorn's and httpx's. structlog only
   sees calls made through structlog; the rest of the world uses stdlib
   `logging`. Left alone the process emits two formats on one stream, and
   the plain-text half -- which is where HTTP-layer failures show up --
   is the half you cannot filter. ProcessorFormatter fixes that by
   running stdlib records through the same processor chain.

2. correlation_id binds once per message instead of being passed by
   hand. It used to be an explicit kwarg at fourteen call sites, which
   works right up until someone adds a fifteenth and forgets.
"""

import logging

import structlog

from app.config import settings

# Processors shared by both paths. Anything that must apply to every log
# line regardless of who emitted it goes here.
_SHARED = [
    # Pulls in whatever process_message bound for this message. Must come
    # first, so an explicit kwarg on a call site still wins over it.
    structlog.contextvars.merge_contextvars,
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.add_log_level,
    # log.exception() attaches exc_info; without this the JSON renderer
    # falls back to repr() on the raw (type, value, tb) tuple and the
    # traceback is unreadable. Must precede the renderer.
    structlog.processors.format_exc_info,
]


def configure_logging() -> None:
    level = getattr(logging, settings.log_level, logging.INFO)

    structlog.configure(
        processors=[
            *_SHARED,
            # Hands off to the stdlib handler below rather than rendering
            # here, so structlog and stdlib records share one formatter
            # and therefore one output format.
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        # Filtering happens when the logger is bound, so a suppressed
        # log.debug() costs almost nothing -- no formatting, no record.
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    handler = logging.StreamHandler()
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            # Runs only for records that came from stdlib logging, so
            # uvicorn's lines end up with the same timestamp and level
            # fields as ours.
            foreign_pre_chain=_SHARED,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
        )
    )

    # Replace handlers rather than appending. uvicorn installs its own
    # during startup, and leaving those attached prints everything twice.
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True

    # httpx logs the full request URL at INFO. For the Graph API that URL
    # carries nothing secret, but media downloads are pre-signed links --
    # so this stays at WARNING rather than putting one in every log sink.
    logging.getLogger("httpx").setLevel(logging.WARNING)
