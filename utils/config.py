"""
Configuration management for model pricing and settings.
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class PricingConfig:
    """Manage model pricing configuration with JSON + env override support."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize pricing configuration.

        Args:
            config_path: Path to pricing JSON file (optional)
        """
        if config_path is None:
            # Default to config/pricing.json relative to project root
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config" / "pricing.json"

        self.config_path = Path(config_path)
        self.pricing_data = self._load_pricing_config()

    def _load_pricing_config(self) -> Dict:
        """Load pricing configuration from JSON file."""
        try:
            if not self.config_path.exists():
                logger.warning(f"Pricing config not found at {self.config_path}, using defaults")
                return self._get_default_pricing()

            with open(self.config_path, 'r') as f:
                data = json.load(f)
                logger.info(f"Loaded pricing config from {self.config_path}")
                return data
        except Exception as e:
            logger.error(f"Error loading pricing config: {e}, using defaults")
            return self._get_default_pricing()

    def _get_default_pricing(self) -> Dict:
        """Return default pricing if config file not found."""
        return {
            "models": {
                "phi-4-multimodal-instruct": {
                    "input_price_per_1m": 0.08,
                    "output_price_per_1m": 0.32
                },
                "gpt-4o-mini": {
                    "input_price_per_1m": 0.15,
                    "output_price_per_1m": 0.60
                }
            },
            "embeddings": {
                "text-embedding-3-small": {
                    "price_per_1m": 0.02
                }
            }
        }

    def get_model_pricing(self, model_name: str) -> Dict[str, float]:
        """
        Get pricing for a specific model.

        Args:
            model_name: Model deployment name

        Returns:
            Dict with input_price_per_1m, output_price_per_1m
        """
        # Check for environment variable overrides first
        env_input = os.getenv("PRICING_INPUT_PER_1M")
        env_output = os.getenv("PRICING_OUTPUT_PER_1M")

        if env_input and env_output:
            logger.info(f"Using pricing from environment variables: "
                       f"input=${env_input}, output=${env_output}")
            return {
                "input_price_per_1m": float(env_input),
                "output_price_per_1m": float(env_output)
            }

        # Fall back to JSON config
        model_pricing = self.pricing_data.get("models", {}).get(model_name)

        if model_pricing:
            logger.info(f"Using pricing for {model_name} from config: "
                       f"input=${model_pricing['input_price_per_1m']}, "
                       f"output=${model_pricing['output_price_per_1m']}")
            return {
                "input_price_per_1m": model_pricing["input_price_per_1m"],
                "output_price_per_1m": model_pricing["output_price_per_1m"]
            }

        # Default fallback
        logger.warning(f"Model {model_name} not found in config, using phi-4 defaults")
        return {
            "input_price_per_1m": 0.08,
            "output_price_per_1m": 0.32
        }

    def get_embedding_pricing(self, embedding_model: str) -> float:
        """
        Get pricing for embedding model.

        Args:
            embedding_model: Embedding model name

        Returns:
            Price per 1M tokens
        """
        # Check environment variable override
        env_embedding = os.getenv("PRICING_EMBEDDING_PER_1M")

        if env_embedding:
            logger.info(f"Using embedding pricing from env: ${env_embedding}")
            return float(env_embedding)

        # Fall back to JSON config
        embedding_pricing = self.pricing_data.get("embeddings", {}).get(embedding_model)

        if embedding_pricing:
            price = embedding_pricing["price_per_1m"]
            logger.info(f"Using embedding pricing for {embedding_model}: ${price}")
            return price

        # Default fallback
        logger.warning(f"Embedding model {embedding_model} not found, using default $0.02")
        return 0.02


# Global instance (lazy loaded)
_pricing_config = None


def get_pricing_config() -> PricingConfig:
    """Get or create global pricing config instance."""
    global _pricing_config
    if _pricing_config is None:
        _pricing_config = PricingConfig()
    return _pricing_config
