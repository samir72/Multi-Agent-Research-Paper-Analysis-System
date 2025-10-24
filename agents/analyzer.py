"""
Analyzer Agent: Analyze individual papers using RAG context.
"""
import os
import json
import logging
from typing import Dict, Any, List
from openai import AzureOpenAI

from utils.schemas import Analysis, Paper
from rag.retrieval import RAGRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalyzerAgent:
    """Agent for analyzing individual papers with RAG."""

    def __init__(
        self,
        rag_retriever: RAGRetriever,
        #model: str = "Phi-4-multimodal-instruct",
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        temperature: float = 0.0
    ):
        """
        Initialize Analyzer Agent.

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

    def _create_analysis_prompt(
        self,
        paper: Paper,
        context: str
    ) -> str:
        """Create prompt for paper analysis."""
        prompt = f"""You are a research paper analyst. Analyze the following paper using ONLY the provided context.

Paper Title: {paper.title}
Authors: {", ".join(paper.authors)}
Abstract: {paper.abstract}

Context from Paper:
{context}

Analyze this paper and extract the following information. You MUST ground every statement in the provided context.

Provide your analysis in the following JSON format:
{{
    "methodology": "Description of research methodology used",
    "key_findings": ["Finding 1", "Finding 2", "Finding 3"],
    "conclusions": "Main conclusions of the paper",
    "limitations": ["Limitation 1", "Limitation 2"],
    "main_contributions": ["Contribution 1", "Contribution 2"],
    "citations": ["Context references used"]
}}

Important:
- Use ONLY information from the provided context
- Be specific and cite which parts of the context support your statements
- If information is not available in the context, indicate "Not available in provided context"
- Provide confidence score based on context completeness
"""
        return prompt

    def analyze_paper(
        self,
        paper: Paper,
        top_k_chunks: int = 10
    ) -> Analysis:
        """
        Analyze a single paper.

        Args:
            paper: Paper object
            top_k_chunks: Number of chunks to retrieve for context

        Returns:
            Analysis object
        """
        try:
            logger.info(f"Analyzing paper: {paper.arxiv_id}")

            # Retrieve relevant chunks for this paper
            # Use broad queries to get comprehensive coverage
            queries = [
                "methodology approach methods",
                "results findings experiments",
                "conclusions contributions implications",
                "limitations future work challenges"
            ]

            all_chunks = []
            chunk_ids = set()

            for query in queries:
                result = self.rag_retriever.retrieve(
                    query=query,
                    top_k=top_k_chunks // len(queries),
                    paper_ids=[paper.arxiv_id]
                )
                for chunk in result["chunks"]:
                    if chunk["chunk_id"] not in chunk_ids:
                        all_chunks.append(chunk)
                        chunk_ids.add(chunk["chunk_id"])

            # Format context
            context = self.rag_retriever.format_context(all_chunks)

            # Create prompt
            prompt = self._create_analysis_prompt(paper, context)

            # Call Azure OpenAI with temperature=0
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a research paper analyst. Provide accurate, grounded analysis based only on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )

            # Parse response
            analysis_data = json.loads(response.choices[0].message.content)

            # Calculate confidence based on context completeness
            confidence = min(len(all_chunks) / top_k_chunks, 1.0)

            # Create Analysis object
            analysis = Analysis(
                paper_id=paper.arxiv_id,
                methodology=analysis_data.get("methodology", "Not available"),
                key_findings=analysis_data.get("key_findings", []),
                conclusions=analysis_data.get("conclusions", "Not available"),
                limitations=analysis_data.get("limitations", []),
                citations=analysis_data.get("citations", []),
                main_contributions=analysis_data.get("main_contributions", []),
                confidence_score=confidence
            )

            logger.info(f"Analysis completed for {paper.arxiv_id} with confidence {confidence:.2f}")
            return analysis

        except Exception as e:
            logger.error(f"Error analyzing paper {paper.arxiv_id}: {str(e)}")
            # Return minimal analysis on error
            return Analysis(
                paper_id=paper.arxiv_id,
                methodology="Analysis failed",
                key_findings=[],
                conclusions="Analysis failed",
                limitations=[],
                citations=[],
                main_contributions=[],
                confidence_score=0.0
            )

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute analyzer agent.

        Args:
            state: Current agent state

        Returns:
            Updated state with analyses
        """
        try:
            logger.info("=== Analyzer Agent Started ===")

            papers = state.get("papers", [])
            if not papers:
                error_msg = "No papers to analyze"
                logger.error(error_msg)
                state["errors"].append(error_msg)
                return state

            # Analyze each paper
            analyses = []
            for paper in papers:
                try:
                    analysis = self.analyze_paper(paper)
                    analyses.append(analysis)
                except Exception as e:
                    error_msg = f"Failed to analyze paper {paper.arxiv_id}: {str(e)}"
                    logger.error(error_msg)
                    state["errors"].append(error_msg)

            if not analyses:
                error_msg = "Failed to analyze any papers"
                logger.error(error_msg)
                state["errors"].append(error_msg)
                return state

            state["analyses"] = analyses
            logger.info(f"=== Analyzer Agent Completed: {len(analyses)} papers analyzed ===")
            return state

        except Exception as e:
            error_msg = f"Analyzer Agent error: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
            return state
