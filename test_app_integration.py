"""
Quick integration test to verify the app works with refactored MCP client.
"""
import os
import sys
from pathlib import Path

# Set environment to use MCP
os.environ["USE_MCP_ARXIV"] = "true"
os.environ["MCP_ARXIV_STORAGE_PATH"] = "data/test_integration_papers"

# Ensure we're in the project directory
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from app import ResearchPaperAnalyzer
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_retriever_agent():
    """Test that RetrieverAgent works with refactored MCP client."""
    logger.info("=" * 80)
    logger.info("Testing RetrieverAgent with refactored MCP client")
    logger.info("=" * 80)

    try:
        # Initialize analyzer
        analyzer = ResearchPaperAnalyzer()

        # Check that MCP client was selected
        logger.info(f"\nArxiv client type: {type(analyzer.arxiv_client).__name__}")

        if type(analyzer.arxiv_client).__name__ != "MCPArxivClient":
            logger.error("✗ Expected MCPArxivClient but got different client")
            return False

        # Test search via retriever
        logger.info("\nTesting search through RetrieverAgent...")
        test_state = {
            "query": "transformer architecture",
            "category": "cs.AI",
            "num_papers": 2,
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "embedding_tokens": 0},
            "errors": []
        }

        # Run retriever
        result_state = analyzer.retriever_agent.run(test_state)

        # Check results
        if "papers" in result_state and len(result_state["papers"]) > 0:
            logger.info(f"\n✓ Successfully retrieved {len(result_state['papers'])} papers")
            for i, paper in enumerate(result_state["papers"], 1):
                logger.info(f"  {i}. {paper.title[:80]}...")
                logger.info(f"     arXiv ID: {paper.arxiv_id}")
            return True
        else:
            logger.error("\n✗ No papers retrieved")
            return False

    except Exception as e:
        logger.error(f"\n✗ Integration test failed: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_retriever_agent()

    logger.info("\n" + "=" * 80)
    if success:
        logger.info("✓ Integration test PASSED")
    else:
        logger.info("✗ Integration test FAILED")
    logger.info("=" * 80)

    sys.exit(0 if success else 1)
