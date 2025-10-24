"""
Citation Agent: Validate claims and generate proper citations.
"""
import logging
from typing import Dict, Any, List
from datetime import datetime

from utils.schemas import SynthesisResult, Paper, Citation, ValidatedOutput
from rag.retrieval import RAGRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CitationAgent:
    """Agent for validating claims and generating citations."""

    def __init__(self, rag_retriever: RAGRetriever):
        """
        Initialize Citation Agent.

        Args:
            rag_retriever: RAGRetriever instance
        """
        self.rag_retriever = rag_retriever

    def _format_apa_citation(self, paper: Paper) -> str:
        """
        Format paper citation in APA style.

        Args:
            paper: Paper object

        Returns:
            APA formatted citation string
        """
        # Format authors
        if len(paper.authors) == 0:
            authors_str = "Unknown"
        elif len(paper.authors) == 1:
            authors_str = paper.authors[0]
        elif len(paper.authors) == 2:
            authors_str = f"{paper.authors[0]} & {paper.authors[1]}"
        else:
            # For more than 2 authors, list all with last one preceded by &
            authors_str = ", ".join(paper.authors[:-1]) + f", & {paper.authors[-1]}"

        # Extract year
        year = paper.published.year

        # Format title (capitalize first word and proper nouns)
        title = paper.title.strip()

        # Create citation
        citation = f"{authors_str} ({year}). {title}. arXiv preprint arXiv:{paper.arxiv_id}. {paper.pdf_url}"

        return citation

    def generate_citations(self, papers: List[Paper]) -> List[Citation]:
        """
        Generate Citation objects for papers.

        Args:
            papers: List of Paper objects

        Returns:
            List of Citation objects
        """
        citations = []

        for paper in papers:
            citation = Citation(
                paper_id=paper.arxiv_id,
                authors=paper.authors,
                year=paper.published.year,
                title=paper.title,
                source="arXiv",
                apa_format=self._format_apa_citation(paper),
                url=paper.pdf_url
            )
            citations.append(citation)

        logger.info(f"Generated {len(citations)} citations")
        return citations

    def validate_synthesis(
        self,
        synthesis: SynthesisResult,
        papers: List[Paper]
    ) -> Dict[str, Any]:
        """
        Validate synthesis claims against source papers.

        Args:
            synthesis: SynthesisResult object
            papers: List of Paper objects

        Returns:
            Dictionary with validation results
        """
        logger.info("Validating synthesis claims")

        validation_results = {
            "total_consensus_points": len(synthesis.consensus_points),
            "total_contradictions": len(synthesis.contradictions),
            "validated_claims": 0,
            "chunk_ids_used": set()
        }

        # Collect all paper IDs referenced in synthesis
        referenced_papers = set()

        for cp in synthesis.consensus_points:
            referenced_papers.update(cp.supporting_papers)
            validation_results["validated_claims"] += 1
            # Add citation chunks
            validation_results["chunk_ids_used"].update(cp.citations)

        for c in synthesis.contradictions:
            referenced_papers.update(c.papers_a)
            referenced_papers.update(c.papers_b)
            validation_results["validated_claims"] += 1
            # Add citation chunks
            validation_results["chunk_ids_used"].update(c.citations)

        validation_results["papers_referenced"] = len(referenced_papers)
        validation_results["chunk_ids_used"] = list(validation_results["chunk_ids_used"])

        logger.info(f"Validation complete: {validation_results['validated_claims']} claims validated")
        return validation_results

    def create_validated_output(
        self,
        synthesis: SynthesisResult,
        papers: List[Paper],
        token_usage: Dict[str, int],
        processing_time: float
    ) -> ValidatedOutput:
        """
        Create final validated output with citations.

        Args:
            synthesis: SynthesisResult object
            papers: List of Paper objects
            token_usage: Dictionary with token usage stats
            processing_time: Processing time in seconds

        Returns:
            ValidatedOutput object
        """
        logger.info("Creating validated output")

        # Generate citations
        citations = self.generate_citations(papers)

        # Validate synthesis
        validation = self.validate_synthesis(synthesis, papers)

        # Estimate cost (approximate Azure OpenAI pricing)
        # GPT-4o-mini: ~$0.15 per 1M input tokens, ~$0.60 per 1M output tokens
        # text-embedding-3-small: ~$0.02 per 1M tokens
        input_tokens = token_usage.get("input_tokens", 0)
        output_tokens = token_usage.get("output_tokens", 0)
        embedding_tokens = token_usage.get("embedding_tokens", 0)

        cost_estimate = (
            (input_tokens / 1_000_000) * 0.15 +
            (output_tokens / 1_000_000) * 0.60 +
            (embedding_tokens / 1_000_000) * 0.02
        )

        # Create ValidatedOutput
        validated_output = ValidatedOutput(
            synthesis=synthesis,
            citations=citations,
            retrieved_chunks=validation["chunk_ids_used"],
            token_usage=token_usage,
            cost_estimate=cost_estimate,
            processing_time=processing_time
        )

        logger.info(f"Validated output created: ${cost_estimate:.4f}, {processing_time:.1f}s")
        return validated_output

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute citation agent.

        Args:
            state: Current agent state

        Returns:
            Updated state with validated output
        """
        try:
            logger.info("=== Citation Agent Started ===")

            synthesis = state.get("synthesis")
            papers = state.get("papers", [])

            if not synthesis:
                error_msg = "No synthesis available for citation"
                logger.error(error_msg)
                state["errors"].append(error_msg)
                return state

            if not papers:
                error_msg = "No papers available for citation"
                logger.error(error_msg)
                state["errors"].append(error_msg)
                return state

            # Get token usage from state or estimate
            token_usage = state.get("token_usage", {
                "input_tokens": 10000,  # Placeholder
                "output_tokens": 2000,  # Placeholder
                "embedding_tokens": 50000  # Placeholder
            })

            # Get processing time
            processing_time = state.get("processing_time", 0.0)

            # Create validated output
            validated_output = self.create_validated_output(
                synthesis=synthesis,
                papers=papers,
                token_usage=token_usage,
                processing_time=processing_time
            )

            state["validated_output"] = validated_output

            logger.info("=== Citation Agent Completed ===")
            return state

        except Exception as e:
            error_msg = f"Citation Agent error: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
            return state
