"""
Retriever Agent: Search arXiv, download papers, and chunk for RAG.
Includes intelligent fallback from MCP/FastMCP to direct arXiv API.
"""
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from utils.arxiv_client import ArxivClient
from utils.pdf_processor import PDFProcessor
from utils.schemas import AgentState, PaperChunk, Paper
from rag.vector_store import VectorStore
from rag.embeddings import EmbeddingGenerator
from utils.langfuse_client import observe

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import MCP clients for type hints
try:
    from utils.fastmcp_arxiv_client import FastMCPArxivClient
except ImportError:
    FastMCPArxivClient = None



class RetrieverAgent:
    """Agent for retrieving and processing papers from arXiv with intelligent fallback."""

    def __init__(
        self,
        arxiv_client: Any,
        pdf_processor: PDFProcessor,
        vector_store: VectorStore,
        embedding_generator: EmbeddingGenerator,
        fallback_client: Optional[Any] = None
    ):
        """
        Initialize Retriever Agent with fallback support.

        Args:
            arxiv_client: Primary client (ArxivClient or FastMCPArxivClient)
            pdf_processor: PDFProcessor instance
            vector_store: VectorStore instance
            embedding_generator: EmbeddingGenerator instance
            fallback_client: Optional fallback client (usually direct ArxivClient) used if primary fails
        """
        self.arxiv_client = arxiv_client
        self.pdf_processor = pdf_processor
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.fallback_client = fallback_client

        # Log client configuration
        client_name = type(arxiv_client).__name__
        logger.info(f"RetrieverAgent initialized with primary client: {client_name}")
        if fallback_client:
            fallback_name = type(fallback_client).__name__
            logger.info(f"Fallback client configured: {fallback_name}")

    def _search_with_fallback(
        self,
        query: str,
        max_results: int,
        category: Optional[str]
    ) -> Optional[List[Paper]]:
        """
        Search for papers with automatic fallback.

        Args:
            query: Search query
            max_results: Maximum number of papers
            category: Optional category filter

        Returns:
            List of Paper objects, or None if both primary and fallback fail
        """
        # Try primary client
        try:
            logger.info(f"Searching with primary client ({type(self.arxiv_client).__name__})")
            papers = self.arxiv_client.search_papers(
                query=query,
                max_results=max_results,
                category=category
            )
            if papers:
                logger.info(f"Primary client found {len(papers)} papers")
                return papers
            else:
                logger.warning("Primary client returned no papers")
        except Exception as e:
            logger.error(f"Primary client search failed: {str(e)}")

        # Try fallback client if available
        if self.fallback_client:
            try:
                logger.warning(f"Attempting fallback with {type(self.fallback_client).__name__}")
                papers = self.fallback_client.search_papers(
                    query=query,
                    max_results=max_results,
                    category=category
                )
                if papers:
                    logger.info(f"Fallback client found {len(papers)} papers")
                    return papers
                else:
                    logger.error("Fallback client returned no papers")
            except Exception as e:
                logger.error(f"Fallback client search failed: {str(e)}")

        logger.error("All search attempts failed")
        return None

    def _download_with_fallback(self, paper: Paper) -> Optional[Path]:
        """
        Download paper with automatic fallback.

        Args:
            paper: Paper object to download

        Returns:
            Path to downloaded PDF, or None if both primary and fallback fail
        """
        # Try primary client
        try:
            path = self.arxiv_client.download_paper(paper)
            if path:
                logger.debug(f"Primary client downloaded {paper.arxiv_id}")
                return path
            else:
                logger.warning(f"Primary client failed to download {paper.arxiv_id}")
        except Exception as e:
            logger.error(f"Primary client download error for {paper.arxiv_id}: {str(e)}")

        # Try fallback client if available
        if self.fallback_client:
            try:
                logger.debug(f"Attempting fallback download for {paper.arxiv_id}")
                path = self.fallback_client.download_paper(paper)
                if path:
                    logger.info(f"Fallback client downloaded {paper.arxiv_id}")
                    return path
                else:
                    logger.error(f"Fallback client failed to download {paper.arxiv_id}")
            except Exception as e:
                logger.error(f"Fallback client download error for {paper.arxiv_id}: {str(e)}")

        logger.error(f"All download attempts failed for {paper.arxiv_id}")
        return None

    @observe(name="retriever_agent_run", as_type="generation")
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute retriever agent.

        Args:
            state: Current agent state

        Returns:
            Updated state with papers and chunks
        """
        try:
            logger.info("=== Retriever Agent Started ===")

            query = state.get("query")
            category = state.get("category")
            num_papers = state.get("num_papers", 5)

            logger.info(f"Query: {query}")
            logger.info(f"Category: {category}")
            logger.info(f"Number of papers: {num_papers}")

            # Step 1: Search arXiv (with fallback)
            logger.info("Step 1: Searching arXiv...")
            papers = self._search_with_fallback(
                query=query,
                max_results=num_papers,
                category=category
            )

            if not papers:
                error_msg = "No papers found for the given query (tried all available clients)"
                logger.error(error_msg)
                state["errors"].append(error_msg)
                return state

            logger.info(f"Found {len(papers)} papers")

            # Validate paper data quality after MCP parsing
            validated_papers = []
            for paper in papers:
                try:
                    # Check for critical data quality issues
                    issues = []

                    # Validate authors field
                    if not isinstance(paper.authors, list):
                        issues.append(f"authors is {type(paper.authors).__name__} instead of list")
                    elif len(paper.authors) == 0:
                        issues.append("authors list is empty")

                    # Validate categories field
                    if not isinstance(paper.categories, list):
                        issues.append(f"categories is {type(paper.categories).__name__} instead of list")

                    # Validate string fields
                    if not isinstance(paper.title, str):
                        issues.append(f"title is {type(paper.title).__name__} instead of str")
                    if not isinstance(paper.pdf_url, str):
                        issues.append(f"pdf_url is {type(paper.pdf_url).__name__} instead of str")
                    if not isinstance(paper.abstract, str):
                        issues.append(f"abstract is {type(paper.abstract).__name__} instead of str")

                    if issues:
                        logger.warning(f"Paper {paper.arxiv_id} has data quality issues: {', '.join(issues)}")
                        # Note: Thanks to Pydantic validators, these should already be fixed
                        # This is just a diagnostic check

                    validated_papers.append(paper)

                except Exception as e:
                    error_msg = f"Failed to validate paper {getattr(paper, 'arxiv_id', 'unknown')}: {str(e)}"
                    logger.error(error_msg)
                    state["errors"].append(error_msg)
                    # Skip this paper but continue with others

            if not validated_papers:
                error_msg = "All papers failed validation checks"
                logger.error(error_msg)
                state["errors"].append(error_msg)
                return state

            logger.info(f"Validated {len(validated_papers)} papers (filtered out {len(papers) - len(validated_papers)})")
            state["papers"] = validated_papers

            # Step 2: Download papers (with fallback)
            logger.info("Step 2: Downloading papers...")
            pdf_paths = []
            for paper in papers:
                path = self._download_with_fallback(paper)
                if path:
                    pdf_paths.append((paper, path))
                else:
                    logger.warning(f"Failed to download paper {paper.arxiv_id} (all clients failed)")

            logger.info(f"Downloaded {len(pdf_paths)} papers")

            # Step 3: Process PDFs and chunk
            logger.info("Step 3: Processing PDFs and chunking...")
            all_chunks = []
            for paper, pdf_path in pdf_paths:
                try:
                    chunks = self.pdf_processor.process_paper(pdf_path, paper)
                    if chunks:
                        all_chunks.extend(chunks)
                        logger.info(f"Processed {len(chunks)} chunks from {paper.arxiv_id}")
                    else:
                        error_msg = f"Failed to process paper {paper.arxiv_id}"
                        logger.warning(error_msg)
                        state["errors"].append(error_msg)
                except Exception as e:
                    error_msg = f"Error processing paper {paper.arxiv_id}: {str(e)}"
                    logger.error(error_msg)
                    state["errors"].append(error_msg)

            if not all_chunks:
                error_msg = "Failed to extract text from any papers"
                logger.error(error_msg)
                state["errors"].append(error_msg)
                return state

            logger.info(f"Total chunks created: {len(all_chunks)}")
            state["chunks"] = all_chunks

            # Step 4: Generate embeddings
            logger.info("Step 4: Generating embeddings...")
            chunk_texts = [chunk.content for chunk in all_chunks]
            embeddings = self.embedding_generator.generate_embeddings_batch(chunk_texts)
            logger.info(f"Generated {len(embeddings)} embeddings")

            # Estimate embedding tokens (Azure doesn't return usage for embeddings)
            # Estimate ~300 tokens per chunk on average
            estimated_embedding_tokens = len(chunk_texts) * 300
            state["token_usage"]["embedding_tokens"] += estimated_embedding_tokens
            logger.info(f"Estimated embedding tokens: {estimated_embedding_tokens}")

            # Step 5: Store in vector database
            logger.info("Step 5: Storing in vector database...")
            self.vector_store.add_chunks(all_chunks, embeddings)

            logger.info("=== Retriever Agent Completed Successfully ===")
            return state

        except Exception as e:
            error_msg = f"Retriever Agent error: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
            return state
