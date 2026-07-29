"""
LangFuse client initialization and instrumentation utilities.
"""
import logging
import os
from typing import Optional, Callable, Any
from functools import wraps
from contextlib import contextmanager

from utils.config import get_langfuse_config

logger = logging.getLogger(__name__)

# Global LangFuse client instance
_langfuse_client = None
_langfuse_enabled = False


def initialize_langfuse():
    """
    Initialize the global LangFuse client.

    This should be called once at application startup.
    If LangFuse is not configured or disabled, this is a no-op.

    Returns:
        Langfuse client instance or None if not configured
    """
    global _langfuse_client, _langfuse_enabled

    config = get_langfuse_config()

    if not config.is_configured():
        logger.info("LangFuse is not configured or disabled. Skipping initialization.")
        _langfuse_enabled = False
        return None

    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(**config.get_init_params())
        _langfuse_enabled = True

        logger.info(f"LangFuse initialized successfully. Host: {config.host}")
        return _langfuse_client

    except ImportError:
        logger.warning("LangFuse package not installed. Install with: pip install langfuse")
        _langfuse_enabled = False
        return None
    except Exception as e:
        logger.error(f"Failed to initialize LangFuse: {e}")
        _langfuse_enabled = False
        return None


def check_langfuse_auth() -> tuple[bool, str]:
    """
    Perform a live credential check against the LangFuse API.

    Builds a short-lived client independent of the module-level singleton,
    so it can be called standalone (e.g. from a CLI) without requiring
    initialize_langfuse() to have run first, and without mutating global state.

    Returns:
        (ok, message) — ok is True only if the API confirms the keys are valid.
    """
    config = get_langfuse_config()

    if not config.public_key or not config.secret_key:
        return False, "LANGFUSE_PUBLIC_KEY and/or LANGFUSE_SECRET_KEY not set"

    try:
        from langfuse import Langfuse

        client = Langfuse(**config.get_init_params())

        if not hasattr(client, "auth_check"):
            return False, "Installed langfuse SDK is too old (missing auth_check()). Try: pip install -U langfuse"

        if client.auth_check():
            return True, f"Credentials valid for host {config.host}"
        return False, "auth_check() returned False — keys did not authenticate"

    except ImportError:
        return False, "langfuse package not installed. Install with: pip install langfuse"
    except Exception as e:
        return False, str(e)


def get_langfuse_client():
    """
    Get the global LangFuse client instance.

    Returns:
        Langfuse client or None if not initialized
    """
    global _langfuse_client
    if _langfuse_client is None:
        initialize_langfuse()
    return _langfuse_client


def is_langfuse_enabled() -> bool:
    """Check if LangFuse is enabled and initialized."""
    return _langfuse_enabled


def instrument_openai():
    """
    Instrument Azure OpenAI client with LangFuse tracing.

    This wraps the OpenAI client to automatically trace all LLM calls.
    Call this before creating any AzureOpenAI clients.
    """
    if not is_langfuse_enabled():
        logger.info("LangFuse not enabled. Skipping OpenAI instrumentation.")
        return

    try:
        from langfuse.openai import openai

        # This patches the global OpenAI client
        logger.info("Azure OpenAI instrumented with LangFuse tracing")

    except ImportError:
        logger.warning("Langfuse OpenAI integration not available. Install with: pip install langfuse")
    except Exception as e:
        logger.error(f"Failed to instrument OpenAI with LangFuse: {e}")


def observe(
    name: Optional[str] = None,
    capture_input: bool = True,
    capture_output: bool = True,
    as_type: str = "span",
):
    """
    Decorator to trace function execution with LangFuse (v3 SDK).

    Args:
        name: Optional custom name for the span/generation
        capture_input: Whether to capture function input
        capture_output: Whether to capture function output
        as_type: Type of observation ("span", "generation", "agent", "tool", ...)

    Usage:
        @observe(name="retriever_agent", as_type="span")
        def retriever_node(state: AgentState) -> AgentState:
            return retriever_agent.run(state)

    Note: wraps with LangFuse's real `observe` decorator at decoration time
    (safe — v3 resolves its client lazily per-call), but the enable/disable
    check happens inside the returned wrapper, evaluated on every call. This
    matters because `@observe(...)` is applied to agent methods and node
    functions at MODULE IMPORT time, which happens before
    initialize_langfuse() runs during app startup — gating at decoration
    time would permanently disable tracing regardless of configuration.
    """

    def decorator(func: Callable) -> Callable:
        try:
            from langfuse import observe as langfuse_observe
        except ImportError:
            logger.warning("LangFuse package not installed. '%s' will run without tracing.", func.__name__)
            return func

        try:
            traced_func = langfuse_observe(
                name=name or func.__name__,
                as_type=as_type,
                capture_input=capture_input,
                capture_output=capture_output,
            )(func)
        except Exception as e:
            logger.error(f"Error applying LangFuse @observe to '{func.__name__}': {e}")
            return func

        @wraps(func)
        def gated_wrapper(*args, **kwargs):
            if is_langfuse_enabled():
                return traced_func(*args, **kwargs)
            return func(*args, **kwargs)

        return gated_wrapper

    return decorator


def flush_langfuse():
    """
    Flush LangFuse client to ensure all observations are sent.

    Call this at the end of a workflow or before shutdown.
    """
    if not is_langfuse_enabled():
        return

    try:
        client = get_langfuse_client()
        if client:
            client.flush()
            logger.debug("LangFuse client flushed")
    except Exception as e:
        logger.error(f"Failed to flush LangFuse client: {e}")


def shutdown_langfuse():
    """
    Shutdown LangFuse client and cleanup.

    Call this at application shutdown.
    """
    global _langfuse_client, _langfuse_enabled

    if not is_langfuse_enabled():
        return

    try:
        flush_langfuse()
        _langfuse_client = None
        _langfuse_enabled = False
        logger.info("LangFuse client shutdown complete")
    except Exception as e:
        logger.error(f"Failed to shutdown LangFuse client: {e}")


# Context manager for scoped tracing
@contextmanager
def workflow_trace(
    name: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    metadata: Optional[dict] = None,
):
    """
    Open one root span for an entire workflow run and tag it with
    session_id/user_id so every nested @observe span/generation created
    during the `with` block (including auto-instrumented OpenAI calls)
    attaches to a single LangFuse trace instead of becoming disconnected
    top-level traces.

    Safely no-ops (yields None) when LangFuse is disabled/unavailable, OR
    when the installed langfuse SDK doesn't support start_as_current_span
    (e.g. a pre-v3 install resolved by an unpinned environment) — a
    tracing-layer failure must never take down the actual workflow.
    Exceptions raised by the CALLER's code inside the `with` block (e.g.
    a real error from app.invoke()) are NOT swallowed and propagate
    normally so existing caller-side error handling is unaffected.

    Usage:
        with workflow_trace("research_workflow_run", session_id=thread_id):
            final_state = app.invoke(initial_state, config=config)
    """
    if not is_langfuse_enabled():
        yield None
        return

    client = get_langfuse_client()
    if client is None:
        yield None
        return

    try:
        span_cm = client.start_as_current_span(name=name, metadata=metadata)
    except Exception as e:
        logger.error(f"Failed to start LangFuse root span '{name}' (tracing disabled for this run): {e}")
        yield None
        return

    with span_cm as span:
        try:
            client.update_current_trace(session_id=session_id, user_id=user_id, metadata=metadata)
        except Exception as e:
            logger.error(f"Failed to tag LangFuse trace '{name}': {e}")
        yield span
