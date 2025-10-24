"""
ChromaDB vector store with persistent storage.
"""
import logging
from typing import List, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings

from utils.schemas import PaperChunk
from rag.embeddings import EmbeddingGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB vector store for paper chunks."""

    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "research_papers"
    ):
        """
        Initialize vector store.

        Args:
            persist_directory: Directory for persistent storage
            collection_name: Name of the collection
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Research paper chunks for RAG"}
        )

        logger.info(f"Vector store initialized with {self.collection.count()} chunks")

    def add_chunks(
        self,
        chunks: List[PaperChunk],
        embeddings: List[List[float]]
    ):
        """
        Add chunks to vector store.

        Args:
            chunks: List of PaperChunk objects
            embeddings: List of embedding vectors
        """
        if not chunks or not embeddings:
            logger.warning("No chunks or embeddings provided")
            return

        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match")

        # Prepare data for ChromaDB
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "paper_id": chunk.paper_id,
                "section": chunk.section or "unknown",
                "page_number": chunk.page_number or 0,
                "arxiv_url": chunk.arxiv_url,
                "title": chunk.metadata.get("title", ""),
                "authors": ",".join(chunk.metadata.get("authors", [])),
                "chunk_index": chunk.metadata.get("chunk_index", 0)
            }
            for chunk in chunks
        ]

        # Check for existing chunks and filter
        existing_ids = set(self.collection.get(ids=ids)["ids"])
        new_indices = [i for i, chunk_id in enumerate(ids) if chunk_id not in existing_ids]

        if not new_indices:
            logger.info("All chunks already exist in vector store")
            return

        # Add only new chunks
        new_ids = [ids[i] for i in new_indices]
        new_documents = [documents[i] for i in new_indices]
        new_metadatas = [metadatas[i] for i in new_indices]
        new_embeddings = [embeddings[i] for i in new_indices]

        self.collection.add(
            ids=new_ids,
            documents=new_documents,
            embeddings=new_embeddings,
            metadatas=new_metadatas
        )

        logger.info(f"Added {len(new_ids)} new chunks to vector store")

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        paper_ids: Optional[List[str]] = None
    ) -> dict:
        """
        Search for similar chunks.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            paper_ids: Optional filter by paper IDs

        Returns:
            Dictionary with search results
        """
        # Build where clause for filtering
        where = None
        if paper_ids:
            if len(paper_ids) == 1:
                where = {"paper_id": paper_ids[0]}
            else:
                where = {"paper_id": {"$in": paper_ids}}

        # Perform search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where
        )

        logger.info(f"Found {len(results['ids'][0])} results")
        return results

    def get_chunks_by_paper(self, paper_id: str) -> List[dict]:
        """
        Get all chunks for a specific paper.

        Args:
            paper_id: arXiv paper ID

        Returns:
            List of chunk dictionaries
        """
        results = self.collection.get(
            where={"paper_id": paper_id}
        )

        chunks = []
        for i, chunk_id in enumerate(results["ids"]):
            chunks.append({
                "chunk_id": chunk_id,
                "content": results["documents"][i],
                "metadata": results["metadatas"][i]
            })

        logger.info(f"Retrieved {len(chunks)} chunks for paper {paper_id}")
        return chunks

    def delete_paper(self, paper_id: str):
        """
        Delete all chunks for a specific paper.

        Args:
            paper_id: arXiv paper ID
        """
        self.collection.delete(
            where={"paper_id": paper_id}
        )
        logger.info(f"Deleted chunks for paper {paper_id}")

    def clear(self):
        """Clear all data from the vector store."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Research paper chunks for RAG"}
        )
        logger.info("Vector store cleared")

    def get_stats(self) -> dict:
        """
        Get vector store statistics.

        Returns:
            Dictionary with stats
        """
        count = self.collection.count()

        # Get unique papers
        all_metadata = self.collection.get()["metadatas"]
        unique_papers = set(m["paper_id"] for m in all_metadata) if all_metadata else set()

        return {
            "total_chunks": count,
            "unique_papers": len(unique_papers),
            "collection_name": self.collection_name,
            "persist_directory": str(self.persist_directory)
        }
