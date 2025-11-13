"""
Test script for refactored MCP arXiv client with in-process handlers.
"""
import os
import sys
from pathlib import Path

# Ensure we're in the project directory
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from utils.mcp_arxiv_client import MCPArxivClient
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_search():
    """Test paper search functionality."""
    logger.info("=" * 80)
    logger.info("TEST 1: Search for papers")
    logger.info("=" * 80)

    try:
        # Initialize client
        client = MCPArxivClient(storage_path="data/test_mcp_papers")

        # Search for papers
        query = "multi-agent reinforcement learning"
        logger.info(f"Searching for: {query}")

        papers = client.search_papers(query=query, max_results=3)

        logger.info(f"\nFound {len(papers)} papers:")
        for i, paper in enumerate(papers, 1):
            logger.info(f"\n{i}. {paper.title}")
            logger.info(f"   arXiv ID: {paper.arxiv_id}")
            logger.info(f"   Authors: {', '.join(paper.authors[:3])}...")
            logger.info(f"   Categories: {', '.join(paper.categories)}")

        return papers

    except Exception as e:
        logger.error(f"Search test failed: {str(e)}", exc_info=True)
        return []

def test_download(papers):
    """Test paper download functionality."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Download paper")
    logger.info("=" * 80)

    if not papers:
        logger.warning("No papers to download (search failed)")
        return

    try:
        client = MCPArxivClient(storage_path="data/test_mcp_papers")

        # Download first paper
        paper = papers[0]
        logger.info(f"Downloading: {paper.title}")
        logger.info(f"arXiv ID: {paper.arxiv_id}")

        pdf_path = client.download_paper(paper)

        if pdf_path and pdf_path.exists():
            logger.info(f"\n✓ Successfully downloaded to: {pdf_path}")
            logger.info(f"  File size: {pdf_path.stat().st_size / 1024:.2f} KB")
        else:
            logger.error("✗ Download failed - file not found")

    except Exception as e:
        logger.error(f"Download test failed: {str(e)}", exc_info=True)

def test_list_cached():
    """Test listing cached papers."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: List cached papers")
    logger.info("=" * 80)

    try:
        client = MCPArxivClient(storage_path="data/test_mcp_papers")

        cached_papers = client.get_cached_papers()

        logger.info(f"\nFound {len(cached_papers)} cached papers:")
        for i, path in enumerate(cached_papers[:5], 1):
            logger.info(f"  {i}. {path.name} ({path.stat().st_size / 1024:.2f} KB)")

        if len(cached_papers) > 5:
            logger.info(f"  ... and {len(cached_papers) - 5} more")

    except Exception as e:
        logger.error(f"List cached test failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    logger.info("Testing refactored MCP arXiv client with in-process handlers\n")

    # Run tests
    papers = test_search()
    test_download(papers)
    test_list_cached()

    logger.info("\n" + "=" * 80)
    logger.info("All tests completed!")
    logger.info("=" * 80)
