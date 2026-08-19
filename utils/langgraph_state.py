"""
LangGraph state schema for the multi-agent workflow.
"""
import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict
from utils.schemas import Paper, PaperChunk, Analysis, SynthesisResult, ValidatedOutput


def merge_token_usage(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    """
    Reducer for AgentState.token_usage: sums matching keys across partial
    updates. Needed because multiple analyzer branches (fanned out via
    LangGraph's Send API) each contribute their own per-paper token delta
    concurrently within one superstep -- plain dicts aren't addable via
    operator.add, so this is a small custom reducer instead.
    """
    return {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}


class AgentState(TypedDict, total=False):
    """
    State dictionary that flows through the LangGraph workflow.

    This TypedDict provides type hints for LangGraph's state management
    while maintaining compatibility with the existing dictionary-based state.

    IMPORTANT: `analyses`, `errors`, and `token_usage` use LangGraph reducers
    (operator.add / merge_token_usage) because the analyzer node is fanned out
    in parallel via Send (one node invocation per paper) -- concurrent branches
    writing to the same key need a reducer to merge correctly instead of the
    default last-write-wins behavior, which would silently drop all but one
    branch's contribution. Because a reducer re-applies on *every* node return
    that includes that key (not just fanned-out branches), every node in
    orchestration/nodes.py returns partial deltas (only the keys it actually
    changed) rather than the whole mutated state dict -- returning the full
    state for a reducer-tracked key would re-merge already-accumulated data
    and duplicate it on every subsequent node.
    """
    # Input fields
    query: str  # User's research question
    category: Optional[str]  # arXiv category filter (e.g., "cs.AI")
    num_papers: int  # Number of papers to analyze

    # Retriever outputs
    papers: List[Paper]  # Papers retrieved from arXiv
    chunks: List[PaperChunk]  # Chunked paper content

    # Analyzer outputs (fanned out via Send -- see note above)
    analyses: Annotated[List[Analysis], operator.add]  # Individual paper analyses
    filtered_analyses: List[Analysis]  # Analyses with confidence > 0

    # Synthesis output
    synthesis: Optional[SynthesisResult]  # Cross-paper synthesis

    # Citation output
    validated_output: Optional[ValidatedOutput]  # Final validated output

    # Metadata and tracking
    errors: Annotated[List[str], operator.add]  # Accumulated error messages
    token_usage: Annotated[Dict[str, int], merge_token_usage]  # Token usage tracking
    start_time: float  # Unix timestamp
    processing_time: float  # Total workflow duration, set by finalize_node
    model_desc: Dict[str, str]  # Model metadata

    # LangFuse tracing metadata
    trace_id: Optional[str]  # LangFuse trace ID
    session_id: Optional[str]  # User session ID
    user_id: Optional[str]  # User identifier (for multi-user systems)


def create_initial_state(
    query: str,
    category: Optional[str],
    num_papers: int,
    model_desc: Dict[str, str],
    start_time: float,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> AgentState:
    """
    Create initial state for LangGraph workflow.

    Args:
        query: User's research question
        category: arXiv category filter
        num_papers: Number of papers to analyze
        model_desc: Model metadata (llm_model, embedding_model)
        start_time: Unix timestamp
        session_id: Optional session identifier
        user_id: Optional user identifier

    Returns:
        Initial AgentState dictionary
    """
    return {
        "query": query,
        "category": category,
        "num_papers": num_papers,
        "papers": [],
        "chunks": [],
        "analyses": [],
        "filtered_analyses": [],
        "synthesis": None,
        "validated_output": None,
        "errors": [],
        "token_usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "embedding_tokens": 0,
        },
        "start_time": start_time,
        "model_desc": model_desc,
        "trace_id": None,
        "session_id": session_id,
        "user_id": user_id,
    }
