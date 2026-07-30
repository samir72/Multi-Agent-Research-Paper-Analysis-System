"""
Propagates the active LangFuse trace ID into standard Python logging so
application logs can be filtered/correlated by trace_id, independent of the
LangFuse UI.

Uses a contextvar rather than a global so it's safe across concurrent runs on
the same process, and is automatically inherited by any worker thread started
via `contextvars.copy_context().run(...)` (the pattern AnalyzerAgent's
ThreadPoolExecutor already uses for LangFuse's own OTEL context).
"""
import contextvars
import logging
from typing import Optional

_trace_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "current_trace_id", default=None
)


def set_current_trace_id(trace_id: Optional[str]) -> contextvars.Token:
    """Bind trace_id to the current context. Returns a token for reset_current_trace_id()."""
    return _trace_id_var.set(trace_id)


def reset_current_trace_id(token: contextvars.Token) -> None:
    """Restore the contextvar to its prior value, undoing set_current_trace_id()."""
    _trace_id_var.reset(token)


def get_current_trace_id() -> Optional[str]:
    return _trace_id_var.get()


class TraceIdLogFilter(logging.Filter):
    """
    Injects the active trace_id (or "-" if none) into every LogRecord as
    record.trace_id.

    Must be attached to a logging.Handler, not a Logger: filters on a Logger
    only apply to records logged directly through that logger instance, not
    to records from child loggers that just propagate up to its handlers.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = _trace_id_var.get() or "-"
        return True
