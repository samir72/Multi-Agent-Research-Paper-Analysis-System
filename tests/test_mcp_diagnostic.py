#!/usr/bin/env python3
"""
Diagnostic script to test MCP arXiv client setup and troubleshoot download issues.

This script will:
1. Connect to the MCP server
2. List all available tools
3. Test search functionality
4. Test download functionality with detailed logging
5. Check file system paths and permissions
"""

import os
import sys
import logging
import asyncio
from pathlib import Path

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.mcp_arxiv_client import MCPArxivClient


async def test_mcp_setup():
    """Run comprehensive MCP diagnostics."""

    print("=" * 80)
    print("MCP arXiv Client Diagnostic Test")
    print("=" * 80)

    # Step 1: Check environment variables
    print("\n[1] Environment Configuration:")
    use_mcp = os.getenv("USE_MCP_ARXIV", "false")
    storage_path = os.getenv("MCP_ARXIV_STORAGE_PATH", "data/mcp_papers")
    print(f"  USE_MCP_ARXIV: {use_mcp}")
    print(f"  MCP_ARXIV_STORAGE_PATH: {storage_path}")

    # Step 2: Check storage directory
    print("\n[2] Storage Directory:")
    storage_path_obj = Path(storage_path)
    print(f"  Path: {storage_path_obj.resolve()}")
    print(f"  Exists: {storage_path_obj.exists()}")
    if storage_path_obj.exists():
        pdf_files = list(storage_path_obj.glob("*.pdf"))
        print(f"  Contains {len(pdf_files)} PDF files")
        if pdf_files:
            print(f"  Files: {[f.name for f in pdf_files[:5]]}")

    # Step 3: Initialize MCP client
    print("\n[3] Initializing MCP Client:")
    try:
        client = MCPArxivClient(storage_path=storage_path)
        print("  ✓ Client initialized successfully")
    except Exception as e:
        print(f"  ✗ Failed to initialize client: {str(e)}")
        return

    # Step 4: Test search
    print("\n[4] Testing Search Functionality:")
    try:
        papers = await client.search_papers_async(
            query="attention mechanism",
            max_results=2
        )
        print(f"  ✓ Search successful, found {len(papers)} papers")
        if papers:
            print(f"  First paper: {papers[0].title[:60]}...")
            print(f"  Paper ID: {papers[0].arxiv_id}")
            test_paper = papers[0]
        else:
            print("  ✗ No papers found")
            return
    except Exception as e:
        print(f"  ✗ Search failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # Step 5: Test download
    print("\n[5] Testing Download Functionality:")
    try:
        print(f"  Attempting to download: {test_paper.arxiv_id}")
        print(f"  PDF URL: {test_paper.pdf_url}")

        pdf_path = await client.download_paper_async(test_paper)

        if pdf_path:
            print(f"  ✓ Download successful!")
            print(f"  File path: {pdf_path}")
            print(f"  File exists: {pdf_path.exists()}")
            if pdf_path.exists():
                file_size = pdf_path.stat().st_size
                print(f"  File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
        else:
            print("  ✗ Download returned None")

    except Exception as e:
        print(f"  ✗ Download failed: {str(e)}")
        import traceback
        traceback.print_exc()

    # Step 6: Check storage directory after download
    print("\n[6] Storage Directory After Download:")
    if storage_path_obj.exists():
        pdf_files = list(storage_path_obj.glob("*.pdf"))
        print(f"  Contains {len(pdf_files)} PDF files")
        if pdf_files:
            print(f"  Files: {[f.name for f in pdf_files]}")

    # Step 7: Cleanup
    print("\n[7] Cleaning Up:")
    try:
        await client.close()
        print("  ✓ MCP session closed")
    except Exception as e:
        print(f"  Warning: Error closing session: {str(e)}")

    print("\n" + "=" * 80)
    print("Diagnostic Test Complete")
    print("=" * 80)


def main():
    """Run the diagnostic test."""
    try:
        asyncio.run(test_mcp_setup())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nUnexpected error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
