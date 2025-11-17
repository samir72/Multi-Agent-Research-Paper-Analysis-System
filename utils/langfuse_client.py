"""
LangFuse client initialization and instrumentation utilities.
"""
import logging
import os
from typing import Optional, Callable, Any
from functools import wraps

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
    Decorator to trace function execution with LangFuse.

    Args:
        name: Optional custom name for the span/generation
        capture_input: Whether to capture function input
        capture_output: Whether to capture function output
        as_type: Type of observation ("span", "generation", "event")

    Usage:
        @observe(name="retriever_agent", as_type="span")
        def retriever_node(state: AgentState) -> AgentState:
            return retriever_agent.run(state)
    """

    def decorator(func: Callable) -> Callable:
        # If LangFuse not enabled, return original function
        if not is_langfuse_enabled():
            return func

        try:
            from langfuse.decorators import langfuse_context, observe as langfuse_observe

            # Use the actual LangFuse decorator
            return langfuse_observe(
                name=name or func.__name__, capture_input=capture_input, capture_output=capture_output, as_type=as_type
            )(func)

        except ImportError:
            logger.warning("LangFuse decorators not available. Function will run without tracing.")
            return func
        except Exception as e:
            logger.error(f"Error applying LangFuse decorator: {e}")
            return func

    return decorator


def start_trace(
    name: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Optional[Any]:
    """
    Start a new LangFuse trace.

    Args:
        name: Trace name
        user_id: Optional user identifier
        session_id: Optional session identifier
        metadata: Optional metadata dictionary

    Returns:
        Trace object or None if LangFuse not enabled
    """
    if not is_langfuse_enabled():
        return None

    try:
        client = get_langfuse_client()
        trace = client.trace(name=name, user_id=user_id, session_id=session_id, metadata=metadata)

        logger.debug(f"Started trace: {name} (session: {session_id})")
        return trace

    except Exception as e:
        logger.error(f"Failed to start LangFuse trace: {e}")
        return None


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
class trace_context:
    """
    Context manager for LangFuse trace.

    Usage:
        with trace_context("workflow", session_id="123") as trace:
            # Your code here
            pass
    """

    def __init__(self, name: str, user_id: Optional[str] = None, session_id: Optional[str] = None, metadata: Optional[dict] = None):
        self.name = name
        self.user_id = user_id
        self.session_id = session_id
        self.metadata = metadata
        self.trace = None

    def __enter__(self):
        self.trace = start_trace(self.name, self.user_id, self.session_id, self.metadata)
        return self.trace

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.error(f"Trace {self.name} ended with error: {exc_val}")
        flush_langfuse()
        return False
