"""
Synthesis Agent: Compare findings across papers and identify patterns.
"""
import os
import json
import logging
from typing import Dict, Any, List
from openai import AzureOpenAI

from utils.schemas import Analysis, SynthesisResult, ConsensusPoint, Contradiction, Paper
from rag.retrieval import RAGRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SynthesisAgent:
    """Agent for synthesizing findings across multiple papers."""

    def __init__(
        self,
        rag_retriever: RAGRetriever,
        #model: str = "Phi-4-multimodal-instruct",
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        temperature: float = 0.0
    ):
        """
        Initialize Synthesis Agent.

        Args:
            rag_retriever: RAGRetriever instance
            model: Azure OpenAI model deployment name
            temperature: Temperature for generation (0 for deterministic)
        """
        self.rag_retriever = rag_retriever
        self.model = model
        self.temperature = temperature

        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            #api_version="2024-02-01",
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )

    def _create_synthesis_prompt(
        self,
        papers: List[Paper],
        analyses: List[Analysis],
        query: str
    ) -> str:
        """Create prompt for synthesis."""
        # Format paper summaries
        paper_summaries = []
        for paper, analysis in zip(papers, analyses):
            summary = f"""
Paper ID: {paper.arxiv_id}
Title: {paper.title}
Authors: {", ".join(paper.authors)}

Analysis:
- Methodology: {analysis.methodology}
- Key Findings: {", ".join(analysis.key_findings)}
- Conclusions: {analysis.conclusions}
- Contributions: {", ".join(analysis.main_contributions)}
- Limitations: {", ".join(analysis.limitations)}
"""
            paper_summaries.append(summary)

        prompt = f"""You are a research synthesis expert. Analyze the following papers in relation to the user's research question.

Research Question: {query}

Papers Analyzed:
{"=" * 80}
{chr(10).join(paper_summaries)}
{"=" * 80}

Synthesize these findings and provide:
1. Consensus points - areas where papers agree
2. Contradictions - areas where papers disagree
3. Research gaps - what's missing or needs further investigation
4. Executive summary addressing the research question

Provide your synthesis in the following JSON format:
{{
    "consensus_points": [
        {{
            "statement": "Clear consensus statement",
            "supporting_papers": ["arxiv_id1", "arxiv_id2"],
            "citations": ["Specific evidence from papers"],
            "confidence": 0.0-1.0
        }}
    ],
    "contradictions": [
        {{
            "topic": "Topic of disagreement",
            "viewpoint_a": "First viewpoint",
            "papers_a": ["arxiv_id1"],
            "viewpoint_b": "Second viewpoint",
            "papers_b": ["arxiv_id2"],
            "citations": ["Evidence for both sides"]
        }}
    ],
    "research_gaps": [
        "Gap 1: What's missing",
        "Gap 2: What needs further research"
    ],
    "summary": "Executive summary addressing the research question with synthesis of all findings",
    "confidence_score": 0.0-1.0
}}

Important:
- Ground all statements in the provided analyses
- Be specific about which papers support which claims
- Identify both agreements and disagreements
- Provide confidence scores based on consistency and evidence strength
"""
        return prompt

    def synthesize(
        self,
        papers: List[Paper],
        analyses: List[Analysis],
        query: str
    ) -> SynthesisResult:
        """
        Synthesize findings across papers.

        Args:
            papers: List of Paper objects
            analyses: List of Analysis objects
            query: Original research question

        Returns:
            SynthesisResult object
        """
        try:
            logger.info(f"Synthesizing {len(papers)} papers")

            # Create synthesis prompt
            prompt = self._create_synthesis_prompt(papers, analyses, query)

            # Call Azure OpenAI with temperature=0
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a research synthesis expert. Provide accurate, grounded synthesis based only on the provided analyses."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )

            # Parse response
            synthesis_data = json.loads(response.choices[0].message.content)

            # Create structured objects
            consensus_points = [
                ConsensusPoint(**cp) for cp in synthesis_data.get("consensus_points", [])
            ]

            contradictions = [
                Contradiction(**c) for c in synthesis_data.get("contradictions", [])
            ]

            # Create SynthesisResult
            synthesis = SynthesisResult(
                consensus_points=consensus_points,
                contradictions=contradictions,
                research_gaps=synthesis_data.get("research_gaps", []),
                summary=synthesis_data.get("summary", ""),
                confidence_score=synthesis_data.get("confidence_score", 0.5),
                papers_analyzed=[p.arxiv_id for p in papers]
            )

            logger.info(f"Synthesis completed with confidence {synthesis.confidence_score:.2f}")
            return synthesis

        except Exception as e:
            logger.error(f"Error during synthesis: {str(e)}")
            # Return minimal synthesis on error
            return SynthesisResult(
                consensus_points=[],
                contradictions=[],
                research_gaps=["Synthesis failed - unable to identify gaps"],
                summary="Synthesis failed due to an error",
                confidence_score=0.0,
                papers_analyzed=[p.arxiv_id for p in papers]
            )

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute synthesis agent.

        Args:
            state: Current agent state

        Returns:
            Updated state with synthesis
        """
        try:
            logger.info("=== Synthesis Agent Started ===")

            papers = state.get("papers", [])
            analyses = state.get("analyses", [])
            query = state.get("query", "")

            if not papers or not analyses:
                error_msg = "No papers or analyses available for synthesis"
                logger.error(error_msg)
                state["errors"].append(error_msg)
                return state

            if len(papers) != len(analyses):
                error_msg = f"Mismatch: {len(papers)} papers but {len(analyses)} analyses"
                logger.warning(error_msg)
                # Use minimum length
                min_len = min(len(papers), len(analyses))
                papers = papers[:min_len]
                analyses = analyses[:min_len]

            # Perform synthesis
            synthesis = self.synthesize(papers, analyses, query)
            state["synthesis"] = synthesis

            logger.info("=== Synthesis Agent Completed ===")
            return state

        except Exception as e:
            error_msg = f"Synthesis Agent error: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
            return state
