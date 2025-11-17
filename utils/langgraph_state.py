"""
LangGraph state schema for the multi-agent workflow.
"""
from typing import Any, Dict, List, Optional, TypedDict
from utils.schemas import Paper, PaperChunk, Analysis, SynthesisResult, ValidatedOutput


class AgentState(TypedDict, total=False):
    """
    State dictionary that flows through the LangGraph workflow.

    This TypedDict provides type hints for LangGraph's state management
    while maintaining compatibility with the existing dictionary-based state.
    """
    # Input fields
    query: str  # User's research question
    category: Optional[str]  # arXiv category filter (e.g., "cs.AI")
    num_papers: int  # Number of papers to analyze

    # Retriever outputs
    papers: List[Paper]  # Papers retrieved from arXiv
    chunks: List[PaperChunk]  # Chunked paper content

    # Analyzer outputs
    analyses: List[Analysis]  # Individual paper analyses
    filtered_analyses: List[Analysis]  # Analyses with confidence > 0

    # Synthesis output
    synthesis: Optional[SynthesisResult]  # Cross-paper synthesis

    # Citation output
    validated_output: Optional[ValidatedOutput]  # Final validated output

    # Metadata and tracking
    errors: List[str]  # Accumulated error messages
    token_usage: Dict[str, int]  # Token usage tracking
    start_time: float  # Unix timestamp
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
