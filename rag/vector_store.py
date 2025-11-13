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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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

