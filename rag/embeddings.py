"""
Azure OpenAI embeddings with batching for cost optimization.
"""
import os
import logging
from typing import List
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from utils.langfuse_client import observe

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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

        # Validate configuration
        if not self.embedding_model:
            raise ValueError(
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME environment variable is not set. "
                "This is required for generating embeddings. Please set it in your .env file."
            )

        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

        if not api_key or not endpoint:
            raise ValueError(
                "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set. "
                "Please configure them in your .env file."
            )

        # Initialize Azure OpenAI client
        try:
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=endpoint
            )
            logger.info(f"Azure OpenAI client initialized for embeddings (deployment: {self.embedding_model})")
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
            ValueError: If input text is empty or model not configured
            Exception: If embedding generation fails
        """
        # Validate input
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty or whitespace-only")

        if not self.embedding_model:
            raise ValueError("Embedding model not configured. Set AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME environment variable")

        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            embedding = response.data[0].embedding
            return embedding

        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg or "Resource not found" in error_msg:
                logger.error(
                    f"\n{'='*80}\n"
                    f"❌ AZURE OPENAI EMBEDDING DEPLOYMENT NOT FOUND (404 Error)\n"
                    f"{'='*80}\n"
                    f"Deployment name: {self.embedding_model}\n"
                    f"Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')}\n"
                    f"\n"
                    f"POSSIBLE CAUSES:\n"
                    f"  1. Deployment '{self.embedding_model}' doesn't exist in your Azure resource\n"
                    f"  2. Deployment name is misspelled\n"
                    f"  3. Using wrong Azure OpenAI resource\n"
                    f"\n"
                    f"HOW TO FIX:\n"
                    f"  Option A: Create deployment in Azure Portal\n"
                    f"    1. Go to https://portal.azure.com\n"
                    f"    2. Navigate to your Azure OpenAI resource\n"
                    f"    3. Go to 'Model deployments' → 'Manage Deployments'\n"
                    f"    4. Create deployment with model 'text-embedding-3-small'\n"
                    f"       and name '{self.embedding_model}'\n"
                    f"\n"
                    f"  Option B: Update AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME\n"
                    f"    1. Check existing embedding deployments in Azure Portal\n"
                    f"    2. Update .env or HuggingFace Spaces secrets with correct name\n"
                    f"    3. Common names: text-embedding-3-small, text-embedding-ada-002\n"
                    f"\n"
                    f"  Option C: Run diagnostic script\n"
                    f"    python scripts/validate_azure_embeddings.py\n"
                    f"\n"
                    f"Original error: {error_msg}\n"
                    f"{'='*80}"
                )
            else:
                logger.error(f"Error generating embedding: {error_msg}")
            raise

    @observe(name="generate_embeddings_batch", as_type="span")
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
            ValueError: If texts is empty or model not configured
            Exception: If embedding generation fails
        """
        # Validate input
        if not self.embedding_model:
            raise ValueError("Embedding model not configured. Set AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME environment variable")

        # Filter out empty strings
        valid_texts = [text for text in texts if text and text.strip()]

        if not valid_texts:
            raise ValueError("No valid texts to embed. All texts are empty or whitespace-only")

        if len(valid_texts) != len(texts):
            logger.warning(f"Filtered out {len(texts) - len(valid_texts)} empty texts from batch")

        all_embeddings = []

        try:
            # Process in batches
            for i in range(0, len(valid_texts), self.batch_size):
                batch = valid_texts[i:i + self.batch_size]

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
            error_msg = str(e)
            if "404" in error_msg or "Resource not found" in error_msg:
                logger.error(
                    f"\n{'='*80}\n"
                    f"❌ AZURE OPENAI EMBEDDING DEPLOYMENT NOT FOUND (404 Error)\n"
                    f"{'='*80}\n"
                    f"Deployment name: {self.embedding_model}\n"
                    f"Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')}\n"
                    f"\n"
                    f"POSSIBLE CAUSES:\n"
                    f"  1. Deployment '{self.embedding_model}' doesn't exist in your Azure resource\n"
                    f"  2. Deployment name is misspelled\n"
                    f"  3. Using wrong Azure OpenAI resource\n"
                    f"\n"
                    f"HOW TO FIX:\n"
                    f"  Option A: Create deployment in Azure Portal\n"
                    f"    1. Go to https://portal.azure.com\n"
                    f"    2. Navigate to your Azure OpenAI resource\n"
                    f"    3. Go to 'Model deployments' → 'Manage Deployments'\n"
                    f"    4. Create deployment with model 'text-embedding-3-small'\n"
                    f"       and name '{self.embedding_model}'\n"
                    f"\n"
                    f"  Option B: Update AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME\n"
                    f"    1. Check existing embedding deployments in Azure Portal\n"
                    f"    2. Update .env or HuggingFace Spaces secrets with correct name\n"
                    f"    3. Common names: text-embedding-3-small, text-embedding-ada-002\n"
                    f"\n"
                    f"  Option C: Run diagnostic script\n"
                    f"    python scripts/validate_azure_embeddings.py\n"
                    f"\n"
                    f"Original error: {error_msg}\n"
                    f"{'='*80}"
                )
            else:
                logger.error(f"Error generating batch embeddings: {error_msg}")
            raise

