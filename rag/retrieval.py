"""
RAG retrieval functions with context formatting.
"""
import logging
from typing import List, Optional, Dict, Any

from rag.vector_store import VectorStore
from rag.embeddings import EmbeddingGenerator
from utils.langfuse_client import observe

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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

    @observe(name="rag_retrieve", as_type="span")
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
                # Optimized: Concise headers to reduce token usage
                header = f"[Chunk {i}] {metadata.get('title', 'Unknown')}\n"
                if metadata.get('section'):
                    header += f"Section: {metadata['section']} | "
                if metadata.get('page_number'):
                    header += f"Page {metadata['page_number']}"
                header += "\n" + "=" * 40 + "\n"
                context_parts.append(header + content)
            else:
                context_parts.append(content)

        return "\n\n".join(context_parts)

