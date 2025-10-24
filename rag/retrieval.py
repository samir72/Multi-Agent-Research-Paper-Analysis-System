"""
RAG retrieval functions with context formatting.
"""
import logging
from typing import List, Optional, Dict, Any

from rag.vector_store import VectorStore
from rag.embeddings import EmbeddingGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGRetriever:
    """RAG retrieval with semantic search and context formatting."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_generator: EmbeddingGenerator,
        top_k: int = 5
    ):
        """
        Initialize RAG retriever.

        Args:
            vector_store: Vector store instance
            embedding_generator: Embedding generator instance
            top_k: Number of chunks to retrieve
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        paper_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Search query
            top_k: Number of chunks to retrieve (overrides default)
            paper_ids: Optional filter by paper IDs

        Returns:
            Dictionary with retrieved chunks and metadata
        """
        k = top_k or self.top_k

        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)

        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=k,
            paper_ids=paper_ids
        )

        # Format results
        chunks = []
        for i, chunk_id in enumerate(results["ids"][0]):
            chunks.append({
                "chunk_id": chunk_id,
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if "distances" in results else None
            })

        logger.info(f"Retrieved {len(chunks)} chunks for query: {query[:50]}...")

        return {
            "query": query,
            "chunks": chunks,
            "chunk_ids": [c["chunk_id"] for c in chunks]
        }

    def format_context(
        self,
        chunks: List[Dict[str, Any]],
        include_metadata: bool = True
    ) -> str:
        """
        Format retrieved chunks into context string.

        Args:
            chunks: List of chunk dictionaries
            include_metadata: Whether to include metadata in context

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            metadata = chunk["metadata"]
            content = chunk["content"]

            if include_metadata:
                header = f"[Chunk {i}] Paper: {metadata.get('title', 'Unknown')}\n"
                header += f"Authors: {metadata.get('authors', 'Unknown')}\n"
                if metadata.get('section'):
                    header += f"Section: {metadata['section']}\n"
                if metadata.get('page_number'):
                    header += f"Page: {metadata['page_number']}\n"
                header += f"Source: {metadata.get('arxiv_url', 'Unknown')}\n"
                header += "-" * 80 + "\n"
                context_parts.append(header + content)
            else:
                context_parts.append(content)

        return "\n\n".join(context_parts)

    def retrieve_with_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        paper_ids: Optional[List[str]] = None,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Retrieve chunks and format into context.

        Args:
            query: Search query
            top_k: Number of chunks to retrieve
            paper_ids: Optional filter by paper IDs
            include_metadata: Whether to include metadata in context

        Returns:
            Dictionary with context string and chunk information
        """
        # Retrieve chunks
        retrieval_result = self.retrieve(query, top_k, paper_ids)

        # Format context
        context = self.format_context(
            retrieval_result["chunks"],
            include_metadata
        )

        return {
            "query": query,
            "context": context,
            "chunks": retrieval_result["chunks"],
            "chunk_ids": retrieval_result["chunk_ids"]
        }

    def retrieve_for_paper(
        self,
        paper_id: str,
        query: str,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Retrieve chunks from a specific paper.

        Args:
            paper_id: arXiv paper ID
            query: Search query
            top_k: Number of chunks to retrieve

        Returns:
            Dictionary with retrieved chunks
        """
        return self.retrieve_with_context(
            query=query,
            top_k=top_k,
            paper_ids=[paper_id]
        )

    def retrieve_multi_paper(
        self,
        paper_ids: List[str],
        query: str,
        top_k_per_paper: int = 3
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve chunks from multiple papers separately.

        Args:
            paper_ids: List of paper IDs
            query: Search query
            top_k_per_paper: Number of chunks per paper

        Returns:
            Dictionary mapping paper IDs to retrieved chunks
        """
        results = {}

        for paper_id in paper_ids:
            paper_result = self.retrieve_for_paper(
                paper_id=paper_id,
                query=query,
                top_k=top_k_per_paper
            )
            results[paper_id] = paper_result["chunks"]

        return results
