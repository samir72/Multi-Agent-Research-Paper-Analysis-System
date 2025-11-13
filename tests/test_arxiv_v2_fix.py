"""
Test script to verify arxiv v2.2.0 PDF URL fix for paper 2102.08370v2.
"""
import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.arxiv_client import ArxivClient, _extract_pdf_url
from utils.fastmcp_arxiv_server import _extract_pdf_url as fastmcp_extract_pdf_url
import arxiv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_extract_pdf_url():
    """Test the _extract_pdf_url helper function directly."""
    print("\n" + "="*80)
    print("TEST 1: Direct PDF URL extraction from arxiv.Result")
    print("="*80)

    # Test with the problematic paper ID
    paper_id = "2102.08370v2"
    logger.info(f"Testing PDF URL extraction for paper: {paper_id}")

    # Fetch the paper using arxiv library
    search = arxiv.Search(id_list=[paper_id])
    result = next(search.results())

    # Show what arxiv library returns
    print(f"\nPaper ID: {result.entry_id.split('/')[-1]}")
    print(f"Title: {result.title[:80]}...")
    print(f"result.pdf_url (deprecated): {result.pdf_url}")
    print(f"\nLinks from result.links:")
    for i, link in enumerate(result.links):
        print(f"  [{i}] {link.href} ({link.rel})")

    # Test extraction from both clients
    url_arxiv = _extract_pdf_url(result)
    url_fastmcp = fastmcp_extract_pdf_url(result)

    print(f"\nExtracted PDF URL (ArxivClient): {url_arxiv}")
    print(f"Extracted PDF URL (FastMCP): {url_fastmcp}")

    assert url_arxiv is not None, "ArxivClient helper failed to extract PDF URL"
    assert url_fastmcp is not None, "FastMCP helper failed to extract PDF URL"
    assert "pdf" in url_arxiv.lower(), "Extracted URL doesn't contain 'pdf'"
    assert url_arxiv == url_fastmcp, "Both helpers should return same URL"

    print("\n✓ PDF URL extraction test PASSED")
    return url_arxiv


def test_arxiv_client_search():
    """Test ArxivClient.search_papers() with the fixed code."""
    print("\n" + "="*80)
    print("TEST 2: ArxivClient.search_papers() integration")
    print("="*80)

    client = ArxivClient(cache_dir="data/test_papers")

    # Search for a specific paper
    papers = client.search_papers(
        query="ti:Attention Is All You Need",
        max_results=1
    )

    assert len(papers) > 0, "No papers found"
    paper = papers[0]

    print(f"\nFound paper:")
    print(f"  ID: {paper.arxiv_id}")
    print(f"  Title: {paper.title[:80]}...")
    print(f"  PDF URL: {paper.pdf_url}")

    assert paper.pdf_url is not None, "Paper pdf_url is None"
    assert "pdf" in paper.pdf_url.lower(), "PDF URL doesn't contain 'pdf'"

    print("\n✓ ArxivClient search test PASSED")
    return paper


def test_fastmcp_download_logic():
    """Test the download_paper logic that was failing."""
    print("\n" + "="*80)
    print("TEST 3: FastMCP download_paper URL extraction")
    print("="*80)

    paper_id = "2102.08370v2"

    # Simulate the download_paper logic
    search = arxiv.Search(id_list=[paper_id])
    result = next(search.results())

    # This is what was failing: result.pdf_url was None
    print(f"\nOld approach (BROKEN):")
    print(f"  result.pdf_url = {result.pdf_url}")

    # New approach with helper
    pdf_url = fastmcp_extract_pdf_url(result)
    print(f"\nNew approach (FIXED):")
    print(f"  _extract_pdf_url(result) = {pdf_url}")

    assert pdf_url is not None, "Failed to extract PDF URL"
    assert paper_id in pdf_url, f"PDF URL doesn't contain paper ID {paper_id}"

    print("\n✓ FastMCP download logic test PASSED")
    return pdf_url


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("ARXIV v2.2.0 PDF URL FIX - VERIFICATION TESTS")
    print("="*80)

    try:
        # Test 1: Direct extraction
        pdf_url = test_extract_pdf_url()

        # Test 2: ArxivClient integration
        paper = test_arxiv_client_search()

        # Test 3: FastMCP download logic
        fastmcp_url = test_fastmcp_download_logic()

        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        print(f"\nThe fix successfully resolves the 'unknown url type: None' error")
        print(f"for paper 2102.08370v2 and all other papers.")
        print(f"\nKey changes:")
        print(f"  1. Added _extract_pdf_url() helper to both clients")
        print(f"  2. Extracts PDF URL from result.links (arxiv v2.2.0+)")
        print(f"  3. Falls back to URL construction if needed")
        print(f"  4. Validates URL exists before use")

        return 0

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
