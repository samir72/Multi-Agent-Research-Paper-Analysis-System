"""
Azure OpenAI embeddings with batching for cost optimization.
"""
import os
import logging
from typing import List
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings using Azure OpenAI with batching."""

    def __init__(
        self,
        batch_size: int = 16,
        #embedding_model: str = "text-embedding-3-small"
        embedding_model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
    ):
        """
        Initialize embedding generator.

        Args:
            batch_size: Number of texts to batch per request
            embedding_model: Azure OpenAI embedding model deployment name
        """
        self.batch_size = batch_size
        self.embedding_model = embedding_model

        # Initialize Azure OpenAI client
        try:
            self.client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                #api_version="2024-02-01",
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
            logger.info("Azure OpenAI client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector

        Raises:
            Exception: If embedding generation fails
        """
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            embedding = response.data[0].embedding
            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            Exception: If embedding generation fails
        """
        all_embeddings = []

        try:
            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]

                logger.info(f"Generating embeddings for batch {i // self.batch_size + 1}")

                response = self.client.embeddings.create(
                    input=batch,
                    model=self.embedding_model
                )

                # Extract embeddings in correct order
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

            logger.info(f"Generated {len(all_embeddings)} embeddings")
            return all_embeddings

        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings.

        Returns:
            Embedding dimension
        """
        # text-embedding-3-small has 1536 dimensions
        test_embedding = self.generate_embedding("test")
        return len(test_embedding)
