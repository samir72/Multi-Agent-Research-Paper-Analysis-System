"""
LangGraph node wrapper functions for agent execution.

These lightweight wrappers integrate existing agents into the LangGraph workflow
while adding LangFuse observability.

IMPORTANT -- partial-update contract: every node below returns only the state
keys it actually changed (a delta dict), never the whole mutated state object.
This is required because AgentState.analyses/errors/token_usage use LangGraph
reducers (see utils/langgraph_state.py) for the analyzer's Send-based fan-out;
a reducer re-applies on every node return that includes that key, so returning
the full state (which still carries the already-accumulated value) would
re-merge and duplicate it on every subsequent node. Confirmed empirically
before this refactor -- see plan history. New nodes must follow the same
partial-update pattern, not the old "mutate state in place and return it" one.
"""
import logging
import time
from typing import Dict, Any

from langgraph.types import Send

from utils.langfuse_client import observe
from utils.langgraph_state import AgentState

logger = logging.getLogger(__name__)


def _errors_delta(before, after):
    """New error messages appended during this node call (before is a prefix of after)."""
    return after[len(before):]


def _token_usage_delta(before, after):
    """Per-key positive/negative deltas between two token_usage snapshots."""
    return {k: after.get(k, 0) - before.get(k, 0) for k in after if after.get(k, 0) != before.get(k, 0)}


@observe(name="retriever_agent", as_type="span")
def retriever_node(state: AgentState, retriever_agent) -> Dict[str, Any]:
    """
    Retriever node: Search arXiv, download PDFs, chunk, embed, and store.

    Args:
        state: Current workflow state
        retriever_agent: RetrieverAgent instance

    Returns:
        Partial state update with papers/chunks (and errors/token_usage deltas, if any)
    """
    logger.info("=== Retriever Node Started ===")
    errors_before = list(state.get("errors", []))
    tokens_before = dict(state.get("token_usage", {}))

    try:
        # RetrieverAgent.run() still takes/returns the full state (unchanged
        # agent contract) -- this wrapper diffs the result to produce a delta.
        updated_state = retriever_agent.run(state)

        logger.info(f"Retriever node completed. Papers: {len(updated_state.get('papers', []))}, "
                   f"Chunks: {len(updated_state.get('chunks', []))}")

        delta: Dict[str, Any] = {
            "papers": updated_state.get("papers", []),
            "chunks": updated_state.get("chunks", []),
        }
        new_errors = _errors_delta(errors_before, updated_state.get("errors", []))
        if new_errors:
            delta["errors"] = new_errors
        token_delta = _token_usage_delta(tokens_before, updated_state.get("token_usage", {}))
        if token_delta:
            delta["token_usage"] = token_delta
        return delta

    except Exception as e:
        logger.error(f"Error in retriever node: {e}")
        return {"errors": [f"Retriever node error: {str(e)}"]}


def should_continue_after_retriever(state: AgentState, analyzer_agent):
    """
    Fan out to one "analyzer" node invocation per paper via LangGraph's Send
    API (replaces the old manual ThreadPoolExecutor in AnalyzerAgent.run()),
    or end the workflow if no papers were found.

    This is also the "start of a new analyzer batch" boundary -- resets the
    circuit breaker and token counters on analyzer_agent, mirroring what the
    old AnalyzerAgent.run() used to do at the top of a batch. analyzer_paper_node
    accumulates into analyzer_agent.batch_tokens (thread-safe, existing lock)
    exactly as before; filter_node reads it back exactly once, after all Send
    branches have completed (LangGraph's Pregel barrier guarantees this).

    Returns:
        "end" if no papers found, otherwise a list of Send objects (one per paper)
    """
    papers = state.get("papers", [])
    if len(papers) == 0:
        logger.warning("No papers retrieved. Ending workflow.")
        return "end"

    analyzer_agent.consecutive_failures = 0
    analyzer_agent.batch_tokens = {"input": 0, "output": 0}
    logger.info(f"Fanning out to analyze {len(papers)} papers in parallel (circuit breaker/token counters reset)")

    return [Send("analyzer", {"paper": paper}) for paper in papers]


@observe(name="analyzer_agent", as_type="span")
def analyzer_paper_node(send_arg: Dict[str, Any], analyzer_agent) -> Dict[str, Any]:
    """
    Analyzer node: analyze a single paper. Invoked once per paper via Send,
    dispatched by LangGraph's own context-propagating thread pool (real OS-thread
    concurrency under the existing synchronous app.invoke() -- confirmed against
    installed langgraph source, see plan history). Replaces the old batch
    AnalyzerAgent.run()/ThreadPoolExecutor path.

    Args:
        send_arg: Send payload, {"paper": Paper} -- analyze_paper() needs no
            other context (confirmed: it never reads query/self.query)
        analyzer_agent: AnalyzerAgent instance (shared across all Send branches)

    Returns:
        Partial update: {"analyses": [Analysis]} on success, or
        {"analyses": [], "errors": [msg]} on failure -- mirrors the old
        run()'s as_completed exception handling (e.g. circuit breaker trips):
        the failed paper is omitted from analyses, not added as a degraded
        entry (analyze_paper()'s own try/except already handles the
        "degraded Analysis with confidence_score=0.0" case internally and
        does not raise for that case).
    """
    paper = send_arg["paper"]
    try:
        analysis = analyzer_agent.analyze_paper(paper)
        logger.info(f"Successfully analyzed paper {paper.arxiv_id}")
        return {"analyses": [analysis]}
    except Exception as e:
        error_msg = f"Failed to analyze paper {paper.arxiv_id}: {str(e)}"
        logger.error(error_msg)
        return {"analyses": [], "errors": [error_msg]}


@observe(name="filter_low_confidence", as_type="span")
def filter_node(state: AgentState, analyzer_agent) -> Dict[str, Any]:
    """
    Filter node: Remove low-confidence analyses. Runs exactly once, after all
    of the analyzer's Send-fanned branches have completed and merged into
    state["analyses"] (LangGraph Pregel barrier semantics -- verified via
    smoke test). Also the single point where the analyzer batch's aggregate
    token usage (accumulated by analyzer_agent.batch_tokens under its
    existing lock, across all per-paper branches) is read back into state,
    since this node is guaranteed to run after every branch is done.

    Args:
        state: Current workflow state
        analyzer_agent: AnalyzerAgent instance (for batch_tokens aggregate)

    Returns:
        Partial state update with filtered_analyses, token_usage, and
        errors (if any)
    """
    logger.info("=== Filter Node Started ===")

    try:
        analyses = state.get("analyses", [])

        # Filter out analyses with confidence_score = 0.0 (failed analyses)
        filtered = [a for a in analyses if a.confidence_score > 0.0]

        delta: Dict[str, Any] = {"filtered_analyses": filtered}

        batch_tokens = getattr(analyzer_agent, "batch_tokens", None)
        if batch_tokens:
            delta["token_usage"] = {
                "input_tokens": batch_tokens.get("input", 0),
                "output_tokens": batch_tokens.get("output", 0),
            }
            logger.info(f"Total analyzer batch tokens: {batch_tokens.get('input', 0)} input, "
                       f"{batch_tokens.get('output', 0)} output")

        logger.info(f"Filter node completed. Retained: {len(filtered)}/{len(analyses)} analyses (confidence > 0.0)")

        if len(filtered) == 0:
            logger.warning("No valid analyses after filtering")
            delta["errors"] = ["All paper analyses failed or had zero confidence"]

        return delta

    except Exception as e:
        logger.error(f"Error in filter node: {e}")
        return {"filtered_analyses": [], "errors": [f"Filter node error: {str(e)}"]}


@observe(name="synthesis_agent", as_type="span")
def synthesis_node(state: AgentState, synthesis_agent) -> Dict[str, Any]:
    """
    Synthesis node: Compare findings across papers.

    Args:
        state: Current workflow state
        synthesis_agent: SynthesisAgent instance

    Returns:
        Partial state update with synthesis (and errors/token_usage deltas, if any)
    """
    logger.info("=== Synthesis Node Started ===")
    errors_before = list(state.get("errors", []))
    tokens_before = dict(state.get("token_usage", {}))

    try:
        # SynthesisAgent.run() still takes/returns the full state (unchanged
        # agent contract) -- this wrapper diffs the result to produce a delta.
        updated_state = synthesis_agent.run(state)

        logger.info("Synthesis node completed")

        delta: Dict[str, Any] = {"synthesis": updated_state.get("synthesis")}
        new_errors = _errors_delta(errors_before, updated_state.get("errors", []))
        if new_errors:
            delta["errors"] = new_errors
        token_delta = _token_usage_delta(tokens_before, updated_state.get("token_usage", {}))
        if token_delta:
            delta["token_usage"] = token_delta
        return delta

    except Exception as e:
        logger.error(f"Error in synthesis node: {e}")
        return {"errors": [f"Synthesis node error: {str(e)}"]}


@observe(name="citation_agent", as_type="span")
def citation_node(state: AgentState, citation_agent) -> Dict[str, Any]:
    """
    Citation node: Generate citations and validate output.

    Args:
        state: Current workflow state
        citation_agent: CitationAgent instance

    Returns:
        Partial state update with validated_output (and errors delta, if any)
    """
    logger.info("=== Citation Node Started ===")
    errors_before = list(state.get("errors", []))

    try:
        updated_state = citation_agent.run(state)

        logger.info("Citation node completed")

        delta: Dict[str, Any] = {"validated_output": updated_state.get("validated_output")}
        new_errors = _errors_delta(errors_before, updated_state.get("errors", []))
        if new_errors:
            delta["errors"] = new_errors
        return delta

    except Exception as e:
        logger.error(f"Error in citation node: {e}")
        return {"errors": [f"Citation node error: {str(e)}"]}


def should_continue_after_filter(state: AgentState) -> str:
    """
    Decide whether to continue after filter based on valid analyses.

    Returns:
        "continue" if valid analyses exist, "end" otherwise
    """
    filtered = state.get("filtered_analyses", [])
    if len(filtered) == 0:
        logger.warning("No valid analyses after filtering. Ending workflow.")
        return "end"
    return "continue"


@observe(name="finalize_node", as_type="span")
def finalize_node(state: AgentState) -> Dict[str, Any]:
    """
    Finalize node: Calculate processing time and update ValidatedOutput.

    This is the last step in the workflow, executed after citation.

    Args:
        state: Current workflow state

    Returns:
        Partial state update with processing_time (and validated_output, if present)
    """
    logger.info("=== Finalize Node Started ===")

    try:
        # Calculate processing time from start_time
        start_time = state.get("start_time", time.time())
        processing_time = time.time() - start_time
        logger.info(f"Total processing time: {processing_time:.1f}s")

        delta: Dict[str, Any] = {"processing_time": processing_time}

        # Update ValidatedOutput with actual processing_time
        validated_output = state.get("validated_output")
        if validated_output:
            validated_output.processing_time = processing_time
            delta["validated_output"] = validated_output
            logger.info(f"Updated ValidatedOutput with processing_time: {processing_time:.1f}s")
        else:
            logger.warning("No ValidatedOutput found in state")

        logger.info("=== Finalize Node Completed ===")
        return delta

    except Exception as e:
        logger.error(f"Error in finalize node: {e}")
        return {"errors": [f"Finalize node error: {str(e)}"]}
