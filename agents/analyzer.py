"""
Analyzer Agent: Analyze individual papers using RAG context.
"""
import os
import json
import logging
import threading
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from utils.schemas import Analysis, Paper
from rag.retrieval import RAGRetriever

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnalyzerAgent:
    """Agent for analyzing individual papers with RAG."""

    def __init__(
        self,
        rag_retriever: RAGRetriever,
        #model: str = "phi-4-multimodal-instruct",
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        temperature: float = 0.0,
        timeout: int = 60
    ):
        """
        Initialize Analyzer Agent.

        Args:
            rag_retriever: RAGRetriever instance
            model: Azure OpenAI model deployment name
            temperature: Temperature for generation (0 for deterministic)
            timeout: Request timeout in seconds (default: 60)
        """
        self.rag_retriever = rag_retriever
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

        # Circuit breaker for consecutive failures
        self.consecutive_failures = 0
        self.max_consecutive_failures = 2

        # Thread-safe token tracking for parallel processing
        self.token_lock = threading.Lock()
        self.batch_tokens = {"input": 0, "output": 0}

        # Initialize Azure OpenAI client with timeout
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            #api_version="2024-02-01",
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            timeout=timeout,
            max_retries=2  # SDK-level retries
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
- For string fields (methodology, conclusions): use "Not available in provided context" if information is missing
- For array fields (key_findings, limitations, main_contributions, citations): use ["Not available in provided context"] or [] if information is missing
- ALWAYS maintain correct JSON types: strings for text fields, arrays for list fields
- Provide confidence score based on context completeness
"""
        return prompt

    def _normalize_analysis_response(self, data: dict) -> dict:
        """
        Normalize LLM response to ensure list fields are lists, not strings.

        This handles cases where the LLM returns a string (e.g., "Not available in provided context")
        for fields that should be lists, preventing Pydantic validation errors.

        Args:
            data: Raw analysis data dictionary from LLM

        Returns:
            Normalized dictionary with correct types for all fields
        """
        list_fields = ['key_findings', 'limitations', 'main_contributions', 'citations']

        for field in list_fields:
            if field in data and isinstance(data[field], str):
                # Convert string to single-element list
                data[field] = [data[field]] if data[field] else []
                logger.warning(
                    f"Normalized '{field}' from string to list. "
                    f"Original value: '{data[field][0] if data[field] else ''}'"
                )

        return data

    def analyze_paper(
        self,
        paper: Paper,
        top_k_chunks: int = 10
    ) -> Analysis:
        """
        Analyze a single paper with retry logic and circuit breaker.

        Args:
            paper: Paper object
            top_k_chunks: Number of chunks to retrieve for context

        Returns:
            Analysis object
        """
        # Circuit breaker: Skip if too many consecutive failures
        if self.consecutive_failures >= self.max_consecutive_failures:
            logger.warning(
                f"Circuit breaker active: Skipping {paper.arxiv_id} after "
                f"{self.consecutive_failures} consecutive failures"
            )
            raise Exception("Circuit breaker active - too many consecutive failures")

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

            # Call Azure OpenAI with temperature=0 and output limits
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a research paper analyst. Provide accurate, grounded analysis based only on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=1500,  # Limit output to prevent slow responses
                response_format={"type": "json_object"}
            )

            # Track token usage (thread-safe)
            if hasattr(response, 'usage') and response.usage:
                with self.token_lock:
                    self.batch_tokens["input"] += response.usage.prompt_tokens
                    self.batch_tokens["output"] += response.usage.completion_tokens
                    logger.info(f"Analyzer token usage for {paper.arxiv_id}: "
                              f"{response.usage.prompt_tokens} input, "
                              f"{response.usage.completion_tokens} output")

            # Parse response
            analysis_data = json.loads(response.choices[0].message.content)

            # Normalize response to ensure list fields are lists (not strings)
            analysis_data = self._normalize_analysis_response(analysis_data)

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

            # Reset circuit breaker on success
            self.consecutive_failures = 0

            return analysis

        except Exception as e:
            # Increment circuit breaker on failure
            self.consecutive_failures += 1

            logger.error(
                f"Error analyzing paper {paper.arxiv_id} ({str(e)}). "
                f"Consecutive failures: {self.consecutive_failures}"
            )

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
        Execute analyzer agent with parallel processing.

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

            # Reset circuit breaker for new batch
            self.consecutive_failures = 0
            logger.info("Circuit breaker reset for new batch")

            # Reset token counters for new batch
            self.batch_tokens = {"input": 0, "output": 0}

            # Analyze papers in parallel (max 4 concurrent for optimal throughput)
            max_workers = min(4, len(papers))
            logger.info(f"Analyzing {len(papers)} papers with {max_workers} parallel workers")

            analyses = []
            failed_papers = []

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all papers for analysis
                future_to_paper = {
                    executor.submit(self.analyze_paper, paper): paper
                    for paper in papers
                }

                # Collect results as they complete
                for future in as_completed(future_to_paper):
                    paper = future_to_paper[future]
                    try:
                        analysis = future.result()
                        analyses.append(analysis)
                        logger.info(f"Successfully analyzed paper {paper.arxiv_id}")
                    except Exception as e:
                        error_msg = f"Failed to analyze paper {paper.arxiv_id}: {str(e)}"
                        logger.error(error_msg)
                        state["errors"].append(error_msg)
                        failed_papers.append(paper.arxiv_id)

            # Accumulate batch tokens to state
            state["token_usage"]["input_tokens"] += self.batch_tokens["input"]
            state["token_usage"]["output_tokens"] += self.batch_tokens["output"]
            logger.info(f"Total analyzer batch tokens: {self.batch_tokens['input']} input, "
                       f"{self.batch_tokens['output']} output")

            if not analyses:
                error_msg = "Failed to analyze any papers"
                logger.error(error_msg)
                state["errors"].append(error_msg)
                return state

            if failed_papers:
                logger.warning(f"Failed to analyze {len(failed_papers)} papers: {failed_papers}")

            state["analyses"] = analyses
            logger.info(f"=== Analyzer Agent Completed: {len(analyses)}/{len(papers)} papers analyzed ===")
            return state

        except Exception as e:
            error_msg = f"Analyzer Agent error: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
            return state
