"""
Retriever Agent: Search arXiv, download papers, and chunk for RAG.
"""
import logging
from typing import Dict, Any
from pathlib import Path

from utils.arxiv_client import ArxivClient
from utils.pdf_processor import PDFProcessor
from utils.schemas import AgentState, PaperChunk
from rag.vector_store import VectorStore
from rag.embeddings import EmbeddingGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrieverAgent:
    """Agent for retrieving and processing papers from arXiv."""

    def __init__(
        self,
        arxiv_client: ArxivClient,
        pdf_processor: PDFProcessor,
        vector_store: VectorStore,
        embedding_generator: EmbeddingGenerator
    ):
        """
        Initialize Retriever Agent.

        Args:
            arxiv_client: ArxivClient instance
            pdf_processor: PDFProcessor instance
            vector_store: VectorStore instance
            embedding_generator: EmbeddingGenerator instance
        """
        self.arxiv_client = arxiv_client
        self.pdf_processor = pdf_processor
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator

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

            # Step 1: Search arXiv
            logger.info("Step 1: Searching arXiv...")
            papers = self.arxiv_client.search_papers(
                query=query,
                max_results=num_papers,
                category=category
            )

            if not papers:
                error_msg = "No papers found for the given query"
                logger.error(error_msg)
                state["errors"].append(error_msg)
                return state

            logger.info(f"Found {len(papers)} papers")
            state["papers"] = papers

            # Step 2: Download papers
            logger.info("Step 2: Downloading papers...")
            pdf_paths = []
            for paper in papers:
                path = self.arxiv_client.download_paper(paper)
                if path:
                    pdf_paths.append((paper, path))
                else:
                    logger.warning(f"Failed to download paper {paper.arxiv_id}")

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
