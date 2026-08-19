"""
LangGraph workflow graph builder for multi-agent RAG system.
"""
import logging
from typing import Optional, Iterator, Dict, Any
import asyncio
import nest_asyncio

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from utils.langgraph_state import AgentState
from utils.langfuse_client import workflow_trace
from utils.trace_context import set_current_trace_id, reset_current_trace_id
from orchestration.nodes import (
    retriever_node,
    analyzer_paper_node,
    filter_node,
    synthesis_node,
    citation_node,
    finalize_node,
    should_continue_after_retriever,
    should_continue_after_filter,
)

logger = logging.getLogger(__name__)

# Enable nested event loops for Gradio compatibility
nest_asyncio.apply()

# Preserves the old AnalyzerAgent.run()'s hard cap of 4 concurrent papers
# (min(4, len(papers))) now that parallelism is driven by LangGraph's own
# Send-dispatched thread pool instead of a manual ThreadPoolExecutor -- without
# this, the executor's default sizing (min(32, os.cpu_count()+4)) applies,
# a real and easy-to-miss increase in concurrent LLM calls.
ANALYZER_MAX_CONCURRENCY = 4


def create_workflow_graph(
    retriever_agent,
    analyzer_agent,
    synthesis_agent,
    citation_agent,
    use_checkpointing: bool = True,
) -> Any:
    """
    Create LangGraph workflow for multi-agent RAG system.

    Args:
        retriever_agent: RetrieverAgent instance
        analyzer_agent: AnalyzerAgent instance
        synthesis_agent: SynthesisAgent instance
        citation_agent: CitationAgent instance
        use_checkpointing: Whether to enable workflow checkpointing

    Returns:
        Compiled LangGraph application
    """
    logger.info("Creating LangGraph workflow graph")

    # Create state graph
    workflow = StateGraph(AgentState)

    # Add nodes with agent instances bound
    workflow.add_node(
        "retriever",
        lambda state: retriever_node(state, retriever_agent)
    )

    # "analyzer" is fanned out via Send (see should_continue_after_retriever
    # below) -- one invocation per paper, each receiving just {"paper": p}
    # rather than the full AgentState.
    workflow.add_node(
        "analyzer",
        lambda send_arg: analyzer_paper_node(send_arg, analyzer_agent)
    )

    workflow.add_node(
        "filter",
        lambda state: filter_node(state, analyzer_agent)
    )

    workflow.add_node(
        "synthesis",
        lambda state: synthesis_node(state, synthesis_agent)
    )

    workflow.add_node(
        "citation",
        lambda state: citation_node(state, citation_agent)
    )

    workflow.add_node(
        "finalize",
        finalize_node
    )

    # Set entry point
    workflow.set_entry_point("retriever")

    # Add conditional edge after retriever: fans out to one "analyzer"
    # invocation per paper via Send when papers exist (Send objects carry
    # their own target node name, bypassing this mapping), or routes to END
    # via the "end" string when none were found.
    workflow.add_conditional_edges(
        "retriever",
        lambda state: should_continue_after_retriever(state, analyzer_agent),
        {
            "end": END,
        }
    )

    # Add edge from analyzer to filter
    workflow.add_edge("analyzer", "filter")

    # Add conditional edge after filter
    workflow.add_conditional_edges(
        "filter",
        should_continue_after_filter,
        {
            "continue": "synthesis",
            "end": END,
        }
    )

    # Add edges for synthesis, citation, and finalize
    workflow.add_edge("synthesis", "citation")
    workflow.add_edge("citation", "finalize")
    workflow.add_edge("finalize", END)

    # Compile workflow
    if use_checkpointing:
        checkpointer = MemorySaver()
        app = workflow.compile(checkpointer=checkpointer)
        logger.info("Workflow compiled with checkpointing enabled")
    else:
        app = workflow.compile()
        logger.info("Workflow compiled without checkpointing")

    return app


async def run_workflow_async(
    app: Any,
    initial_state: AgentState,
    thread_id: Optional[str] = None,
) -> Iterator[AgentState]:
    """
    Run LangGraph workflow asynchronously with streaming.

    Args:
        app: Compiled LangGraph application
        initial_state: Initial workflow state
        thread_id: Optional thread ID for checkpointing

    Yields:
        State updates after each node execution
    """
    config = {
        "configurable": {"thread_id": thread_id or "default"},
        "max_concurrency": ANALYZER_MAX_CONCURRENCY,
    }

    logger.info(f"Starting async workflow execution (thread_id: {thread_id})")

    try:
        async for event in app.astream(initial_state, config=config):
            # Event is a dict with node name as key
            for node_name, node_state in event.items():
                logger.debug(f"Node '{node_name}' completed")
                yield node_state

    except Exception as e:
        logger.error(f"Error during workflow execution: {e}")
        # Yield error state
        initial_state["errors"].append(f"Workflow error: {str(e)}")
        yield initial_state


def _run_workflow_streaming(
    app: Any,
    initial_state: AgentState,
    thread_id: Optional[str] = None,
) -> Iterator[AgentState]:
    """
    Run LangGraph workflow with streaming (internal generator function).

    Args:
        app: Compiled LangGraph application
        initial_state: Initial workflow state
        thread_id: Optional thread ID for checkpointing

    Yields:
        State updates after each node execution
    """
    # Create new event loop for streaming
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        async def stream_wrapper():
            async for state in run_workflow_async(app, initial_state, thread_id):
                yield state

        async_gen = stream_wrapper()

        # Convert async generator to sync generator
        while True:
            try:
                yield loop.run_until_complete(async_gen.__anext__())
            except StopAsyncIteration:
                break
    finally:
        loop.close()


def run_workflow(
    app: Any,
    initial_state: AgentState,
    thread_id: Optional[str] = None,
    use_streaming: bool = False,
) -> Any:
    """
    Run LangGraph workflow (sync wrapper for Gradio compatibility).

    Args:
        app: Compiled LangGraph application
        initial_state: Initial workflow state
        thread_id: Optional thread ID for checkpointing
        use_streaming: Whether to stream intermediate results

    Returns:
        Final state (if use_streaming=False) or generator of states (if use_streaming=True)
    """
    config = {
        "configurable": {"thread_id": thread_id or "default"},
        "max_concurrency": ANALYZER_MAX_CONCURRENCY,
    }

    logger.info(f"Starting workflow execution (thread_id: {thread_id}, streaming: {use_streaming})")

    try:
        if use_streaming:
            # Return generator for streaming
            return _run_workflow_streaming(app, initial_state, thread_id)
        else:
            # Non-streaming execution - just return final state
            with workflow_trace(
                "research_workflow_run",
                session_id=thread_id,
                user_id=initial_state.get("user_id"),
                metadata={
                    "query": initial_state.get("query"),
                    "category": initial_state.get("category"),
                    "num_papers": initial_state.get("num_papers"),
                },
            ) as trace_id:
                if trace_id:
                    initial_state["trace_id"] = trace_id
                token = set_current_trace_id(trace_id)
                try:
                    final_state = app.invoke(initial_state, config=config)
                finally:
                    reset_current_trace_id(token)
            logger.info(f"Workflow execution completed (trace_id: {trace_id})")
            return final_state

    except Exception as e:
        logger.error(f"Error during workflow execution: {e}")
        initial_state["errors"].append(f"Workflow execution error: {str(e)}")
        return initial_state


def get_workflow_state(
    app: Any,
    thread_id: str,
) -> Optional[AgentState]:
    """
    Get current state of a workflow execution.

    Args:
        app: Compiled LangGraph application
        thread_id: Thread ID of the workflow

    Returns:
        Current state or None if not found
    """
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = app.get_state(config)
        return state.values if state else None

    except Exception as e:
        logger.error(f"Error getting workflow state: {e}")
        return None
