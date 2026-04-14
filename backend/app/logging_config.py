"""Console logging for request lifecycle tracing (import `setup_logging` early from `app.main`)."""

import logging
import sys

LOG_FORMAT = "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup_logging(level: int = logging.INFO) -> None:
    """
    Attach a stream handler to the `app` logger once.
    Safe to call multiple times (e.g. under uvicorn reload).
    """
    global _configured
    if _configured:
        return
    log = logging.getLogger("app")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    log.setLevel(level)
    log.addHandler(handler)
    log.propagate = False
    _configured = True
