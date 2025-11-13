#!/usr/bin/env python3
"""
Debug script to test MCP arXiv client with enhanced error handling.
This script helps diagnose issues with MCP server connections and downloads.
"""
import os
import sys
import logging
from pathlib import Path

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.mcp_arxiv_client import MCPArxivClient
from utils.schemas import Paper
from datetime import datetime


def test_client_initialization():
    """Test client initialization and storage setup."""
    print("\n" + "="*80)
    print("TEST 1: Client Initialization")
    print("="*80)

    try:
        client = MCPArxivClient(storage_path="./data/mcp_papers")
        print(f"✓ Client initialized successfully")
        print(f"  Storage path: {client.storage_path}")
        print(f"  Storage exists: {client.storage_path.exists()}")
        return client
    except Exception as e:
        print(f"✗ Client initialization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_search_papers(client):
    """Test paper search functionality."""
    print("\n" + "="*80)
    print("TEST 2: Search Papers")
    print("="*80)

    if not client:
        print("⊘ Skipped - no client available")
        return []

    try:
        papers = client.search_papers("machine learning", max_results=2)
        print(f"✓ Search completed successfully")
        print(f"  Found {len(papers)} papers")
        for i, paper in enumerate(papers, 1):
            print(f"  {i}. {paper.title[:60]}...")
            print(f"     arXiv ID: {paper.arxiv_id}")
        return papers
    except Exception as e:
        print(f"✗ Search failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


def test_download_paper(client, papers):
    """Test paper download functionality."""
    print("\n" + "="*80)
    print("TEST 3: Download Paper")
    print("="*80)

    if not client or not papers:
        print("⊘ Skipped - no client or papers available")
        return

    paper = papers[0]
    print(f"Attempting to download: {paper.title[:60]}...")
    print(f"arXiv ID: {paper.arxiv_id}")
    print(f"Expected path: {client.storage_path / f'{paper.arxiv_id}.pdf'}")

    try:
        pdf_path = client.download_paper(paper)

        if pdf_path:
            print(f"✓ Download completed successfully")
            print(f"  File path: {pdf_path}")
            print(f"  File exists: {pdf_path.exists()}")
            if pdf_path.exists():
                print(f"  File size: {pdf_path.stat().st_size / 1024:.1f} KB")
        else:
            print(f"✗ Download returned None (check logs above for details)")
            print(f"  This could indicate:")
            print(f"    - MCP server error")
            print(f"    - Storage path mismatch")
            print(f"    - Network/API issue")

    except Exception as e:
        print(f"✗ Download failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()


def test_storage_contents(client):
    """Check storage directory contents."""
    print("\n" + "="*80)
    print("TEST 4: Storage Directory Contents")
    print("="*80)

    if not client:
        print("⊘ Skipped - no client available")
        return

    try:
        pdf_files = list(client.storage_path.glob("*.pdf"))
        print(f"Storage path: {client.storage_path}")
        print(f"Total PDF files: {len(pdf_files)}")

        if pdf_files:
            print("\nFiles in storage:")
            for i, pdf_file in enumerate(pdf_files[:10], 1):
                size_kb = pdf_file.stat().st_size / 1024
                print(f"  {i}. {pdf_file.name} ({size_kb:.1f} KB)")

            if len(pdf_files) > 10:
                print(f"  ... and {len(pdf_files) - 10} more files")
        else:
            print("  (no PDF files found)")

    except Exception as e:
        print(f"✗ Storage check failed: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """Run all diagnostic tests."""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "MCP arXiv Client Diagnostic Tool" + " "*26 + "║")
    print("╚" + "="*78 + "╝")

    # Check environment
    print("\nEnvironment Configuration:")
    print(f"  USE_MCP_ARXIV: {os.getenv('USE_MCP_ARXIV', 'not set')}")
    print(f"  MCP_ARXIV_STORAGE_PATH: {os.getenv('MCP_ARXIV_STORAGE_PATH', 'not set')}")

    # Run tests
    client = test_client_initialization()
    test_storage_contents(client)
    papers = test_search_papers(client)
    test_download_paper(client, papers)

    # Final summary
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    print("Review the logs above to identify any issues.")
    print("\nCommon issues and solutions:")
    print("  1. 'Cannot mix str and non-str arguments' error:")
    print("     → Now handled with robust type checking in _call_tool")
    print("  2. 'File not found after download':")
    print("     → Check MCP server storage path configuration")
    print("     → Review 'MCP response' logs to see what server returned")
    print("  3. 'Connection failed':")
    print("     → Ensure MCP server is running and accessible")
    print("     → Check server command in logs")
    print("\n")


if __name__ == "__main__":
    main()
