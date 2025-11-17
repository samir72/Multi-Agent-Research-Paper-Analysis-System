"""
LangGraph node wrapper functions for agent execution.

These lightweight wrappers integrate existing agents into the LangGraph workflow
while adding LangFuse observability.
"""
import logging
import time
from typing import Dict, Any

from utils.langfuse_client import observe
from utils.langgraph_state import AgentState

logger = logging.getLogger(__name__)


@observe(name="retriever_agent", as_type="span")
def retriever_node(state: AgentState, retriever_agent) -> AgentState:
    """
    Retriever node: Search arXiv, download PDFs, chunk, embed, and store.

    Args:
        state: Current workflow state
        retriever_agent: RetrieverAgent instance

    Returns:
        Updated state with papers and chunks
    """
    logger.info("=== Retriever Node Started ===")

    try:
        # Run retriever agent
        updated_state = retriever_agent.run(state)

        logger.info(f"Retriever node completed. Papers: {len(updated_state.get('papers', []))}, "
                   f"Chunks: {len(updated_state.get('chunks', []))}")

        return updated_state

    except Exception as e:
        logger.error(f"Error in retriever node: {e}")
        state["errors"].append(f"Retriever node error: {str(e)}")
        return state


@observe(name="analyzer_agent", as_type="span")
def analyzer_node(state: AgentState, analyzer_agent) -> AgentState:
    """
    Analyzer node: Analyze individual papers using RAG.

    Args:
        state: Current workflow state
        analyzer_agent: AnalyzerAgent instance

    Returns:
        Updated state with analyses
    """
    logger.info("=== Analyzer Node Started ===")

    try:
        # Run analyzer agent
        updated_state = analyzer_agent.run(state)

        logger.info(f"Analyzer node completed. Analyses: {len(updated_state.get('analyses', []))}")

        return updated_state

    except Exception as e:
        logger.error(f"Error in analyzer node: {e}")
        state["errors"].append(f"Analyzer node error: {str(e)}")
        return state


@observe(name="filter_low_confidence", as_type="span")
def filter_node(state: AgentState) -> AgentState:
    """
    Filter node: Remove low-confidence analyses.

    Args:
        state: Current workflow state

    Returns:
        Updated state with filtered_analyses
    """
    logger.info("=== Filter Node Started ===")

    try:
        analyses = state.get("analyses", [])

        # Filter out analyses with confidence_score = 0.0 (failed analyses)
        filtered = [a for a in analyses if a.confidence_score > 0.0]

        state["filtered_analyses"] = filtered

        logger.info(f"Filter node completed. Retained: {len(filtered)}/{len(analyses)} analyses (confidence > 0.0)")

        if len(filtered) == 0:
            logger.warning("No valid analyses after filtering")
            state["errors"].append("All paper analyses failed or had zero confidence")

        return state

    except Exception as e:
        logger.error(f"Error in filter node: {e}")
        state["errors"].append(f"Filter node error: {str(e)}")
        state["filtered_analyses"] = []
        return state


@observe(name="synthesis_agent", as_type="span")
def synthesis_node(state: AgentState, synthesis_agent) -> AgentState:
    """
    Synthesis node: Compare findings across papers.

    Args:
        state: Current workflow state
        synthesis_agent: SynthesisAgent instance

    Returns:
        Updated state with synthesis
    """
    logger.info("=== Synthesis Node Started ===")

    try:
        # Run synthesis agent
        updated_state = synthesis_agent.run(state)

        logger.info("Synthesis node completed")

        return updated_state

    except Exception as e:
        logger.error(f"Error in synthesis node: {e}")
        state["errors"].append(f"Synthesis node error: {str(e)}")
        return state


@observe(name="citation_agent", as_type="span")
def citation_node(state: AgentState, citation_agent) -> AgentState:
    """
    Citation node: Generate citations and validate output.

    Args:
        state: Current workflow state
        citation_agent: CitationAgent instance

    Returns:
        Updated state with validated_output
    """
    logger.info("=== Citation Node Started ===")

    try:
        # Run citation agent
        updated_state = citation_agent.run(state)

        logger.info("Citation node completed")

        return updated_state

    except Exception as e:
        logger.error(f"Error in citation node: {e}")
        state["errors"].append(f"Citation node error: {str(e)}")
        return state


# Conditional edge functions for LangGraph routing

def should_continue_after_retriever(state: AgentState) -> str:
    """
    Decide whether to continue after retriever based on papers found.

    Returns:
        "continue" if papers found, "end" otherwise
    """
    papers = state.get("papers", [])
    if len(papers) == 0:
        logger.warning("No papers retrieved. Ending workflow.")
        return "end"
    return "continue"


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
def finalize_node(state: AgentState) -> AgentState:
    """
    Finalize node: Calculate processing time and update ValidatedOutput.

    This is the last step in the workflow, executed after citation.

    Args:
        state: Current workflow state

    Returns:
        Updated state with final processing_time
    """
    logger.info("=== Finalize Node Started ===")

    try:
        # Calculate processing time from start_time
        start_time = state.get("start_time", time.time())
        processing_time = time.time() - start_time
        logger.info(f"Total processing time: {processing_time:.1f}s")

        # Update processing_time in state
        state["processing_time"] = processing_time

        # Update ValidatedOutput with actual processing_time
        validated_output = state.get("validated_output")
        if validated_output:
            # Create updated ValidatedOutput with actual processing_time
            validated_output.processing_time = processing_time
            state["validated_output"] = validated_output
            logger.info(f"Updated ValidatedOutput with processing_time: {processing_time:.1f}s")
        else:
            logger.warning("No ValidatedOutput found in state")

        logger.info("=== Finalize Node Completed ===")
        return state

    except Exception as e:
        logger.error(f"Error in finalize node: {e}")
        state["errors"].append(f"Finalize node error: {str(e)}")
        return state
