"""
Semantic caching system for cost optimization.
"""
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import hashlib
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticCache:
    """Semantic cache using embeddings and cosine similarity."""

    def __init__(
        self,
        cache_dir: str = "data/cache",
        similarity_threshold: float = 0.95
    ):
        """
        Initialize semantic cache.

        Args:
            cache_dir: Directory to store cache files
            similarity_threshold: Cosine similarity threshold for cache hits
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.similarity_threshold = similarity_threshold
        self.cache_file = self.cache_dir / "semantic_cache.json"
        self.cache_data = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading cache: {str(e)}")
                return {}
        return {}

    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")

    def _cosine_similarity(
        self,
        embedding1: list,
        embedding2: list
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _generate_key(self, query: str, category: Optional[str] = None) -> str:
        """Generate cache key from query and category."""
        content = f"{query}_{category or 'none'}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(
        self,
        query: str,
        query_embedding: list,
        category: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Try to retrieve cached result.

        Args:
            query: Search query
            query_embedding: Query embedding vector
            category: Optional category filter

        Returns:
            Cached result if found, None otherwise
        """
        try:
            # Check for exact match first
            exact_key = self._generate_key(query, category)
            if exact_key in self.cache_data:
                logger.info("Exact cache hit")
                return self.cache_data[exact_key]["result"]

            # Check for semantic similarity
            best_similarity = 0.0
            best_result = None

            for key, cached_item in self.cache_data.items():
                # Only compare with same category
                if cached_item.get("category") != (category or "none"):
                    continue

                cached_embedding = cached_item.get("embedding")
                if not cached_embedding:
                    continue

                similarity = self._cosine_similarity(query_embedding, cached_embedding)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_result = cached_item["result"]

            if best_similarity >= self.similarity_threshold:
                logger.info(f"Semantic cache hit with similarity {best_similarity:.3f}")
                return best_result

            logger.info("Cache miss")
            return None

        except Exception as e:
            logger.error(f"Error retrieving from cache: {str(e)}")
            return None

    def set(
        self,
        query: str,
        query_embedding: list,
        result: Dict[str, Any],
        category: Optional[str] = None
    ):
        """
        Store result in cache.

        Args:
            query: Search query
            query_embedding: Query embedding vector
            result: Result to cache
            category: Optional category filter
        """
        try:
            key = self._generate_key(query, category)

            self.cache_data[key] = {
                "query": query,
                "category": category or "none",
                "embedding": query_embedding,
                "result": result
            }

            self._save_cache()
            logger.info(f"Cached result for query: {query[:50]}...")

        except Exception as e:
            logger.error(f"Error storing in cache: {str(e)}")

    def clear(self):
        """Clear all cache data."""
        self.cache_data = {}
        self._save_cache()
        logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        return {
            "total_entries": len(self.cache_data),
            "cache_file_size": self.cache_file.stat().st_size if self.cache_file.exists() else 0,
            "similarity_threshold": self.similarity_threshold
        }
