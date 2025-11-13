"""
Test data validation fixes for MCP paper parsing and PDF processing.
This test verifies that malformed data (dicts instead of lists) is handled correctly.
"""
import sys
from datetime import datetime
from utils.schemas import Paper
from utils.pdf_processor import PDFProcessor


def test_paper_schema_validators():
    """Test that Paper schema validators correctly normalize malformed data."""
    print("\n" + "="*80)
    print("TEST 1: Paper Schema Validators")
    print("="*80)

    # Test 1: Authors as dict (malformed)
    print("\n1. Testing authors as dict (malformed data)...")
    try:
        paper = Paper(
            arxiv_id="test.001",
            title="Test Paper",
            authors={"author1": "John Doe", "author2": "Jane Smith"},  # Dict instead of list!
            abstract="Test abstract",
            pdf_url="https://arxiv.org/pdf/test.001.pdf",
            published=datetime.now(),
            categories=["cs.AI"]
        )
        print(f"   ✓ Paper created successfully")
        print(f"   Authors type: {type(paper.authors)}")
        print(f"   Authors value: {paper.authors}")
        assert isinstance(paper.authors, list), "Authors should be normalized to list"
        print(f"   ✓ Authors correctly normalized to list")
    except Exception as e:
        print(f"   ✗ Failed: {str(e)}")
        return False

    # Test 2: Categories as dict (malformed)
    print("\n2. Testing categories as dict (malformed data)...")
    try:
        paper = Paper(
            arxiv_id="test.002",
            title="Test Paper 2",
            authors=["John Doe"],
            abstract="Test abstract",
            pdf_url="https://arxiv.org/pdf/test.002.pdf",
            published=datetime.now(),
            categories={"cat1": "cs.AI", "cat2": "cs.LG"}  # Dict instead of list!
        )
        print(f"   ✓ Paper created successfully")
        print(f"   Categories type: {type(paper.categories)}")
        print(f"   Categories value: {paper.categories}")
        assert isinstance(paper.categories, list), "Categories should be normalized to list"
        print(f"   ✓ Categories correctly normalized to list")
    except Exception as e:
        print(f"   ✗ Failed: {str(e)}")
        return False

    # Test 3: Multiple fields malformed
    print("\n3. Testing multiple fields malformed...")
    try:
        paper = Paper(
            arxiv_id="test.003",
            title={"title": "Test Paper 3"},  # Dict!
            authors={"names": ["John Doe", "Jane Smith"]},  # Dict with nested list!
            abstract={"summary": "Test abstract"},  # Dict!
            pdf_url={"url": "https://arxiv.org/pdf/test.003.pdf"},  # Dict!
            published=datetime.now(),
            categories={"categories": ["cs.AI"]}  # Dict with nested list!
        )
        print(f"   ✓ Paper created successfully")
        print(f"   Title type: {type(paper.title)}, value: {paper.title}")
        print(f"   Authors type: {type(paper.authors)}, value: {paper.authors}")
        print(f"   Abstract type: {type(paper.abstract)}, value: {paper.abstract[:50]}...")
        print(f"   PDF URL type: {type(paper.pdf_url)}, value: {paper.pdf_url}")
        print(f"   Categories type: {type(paper.categories)}, value: {paper.categories}")

        assert isinstance(paper.title, str), "Title should be normalized to string"
        assert isinstance(paper.authors, list), "Authors should be normalized to list"
        assert isinstance(paper.abstract, str), "Abstract should be normalized to string"
        assert isinstance(paper.pdf_url, str), "PDF URL should be normalized to string"
        assert isinstance(paper.categories, list), "Categories should be normalized to list"
        print(f"   ✓ All fields correctly normalized")
    except Exception as e:
        print(f"   ✗ Failed: {str(e)}")
        return False

    print("\n" + "="*80)
    print("✓ ALL PAPER SCHEMA VALIDATION TESTS PASSED")
    print("="*80)
    return True


def test_pdf_processor_resilience():
    """Test that PDFProcessor handles malformed Paper objects gracefully."""
    print("\n" + "="*80)
    print("TEST 2: PDFProcessor Resilience")
    print("="*80)

    processor = PDFProcessor(chunk_size=100, chunk_overlap=10)

    # Create a paper with properly validated data
    print("\n1. Testing PDF processor with validated Paper object...")
    try:
        paper = Paper(
            arxiv_id="test.004",
            title="Test Paper",
            authors={"author1": "John Doe"},  # Will be normalized by validators
            abstract="Test abstract",
            pdf_url="https://arxiv.org/pdf/test.004.pdf",
            published=datetime.now(),
            categories=["cs.AI"]
        )

        # Create a simple test text
        test_text = "This is a test document. " * 100

        chunks = processor.chunk_text(test_text, paper)
        print(f"   ✓ Created {len(chunks)} chunks successfully")
        print(f"   First chunk metadata authors type: {type(chunks[0].metadata['authors'])}")
        print(f"   First chunk metadata authors: {chunks[0].metadata['authors']}")

        assert isinstance(chunks[0].metadata['authors'], list), "Chunk metadata authors should be list"
        print(f"   ✓ Chunk metadata correctly contains list for authors")

    except Exception as e:
        print(f"   ✗ Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*80)
    print("✓ PDF PROCESSOR RESILIENCE TESTS PASSED")
    print("="*80)
    return True


if __name__ == "__main__":
    print("\n" + "="*80)
    print("DATA VALIDATION FIX VERIFICATION TESTS")
    print("="*80)
    print("\nThese tests verify that the fixes for malformed MCP data work correctly:")
    print("- Paper schema validators normalize dict fields to proper types")
    print("- PDF processor handles validated Paper objects without errors")
    print("="*80)

    test1_pass = test_paper_schema_validators()
    test2_pass = test_pdf_processor_resilience()

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Paper Schema Validators: {'✓ PASS' if test1_pass else '✗ FAIL'}")
    print(f"PDF Processor Resilience: {'✓ PASS' if test2_pass else '✗ FAIL'}")
    print("="*80)

    if test1_pass and test2_pass:
        print("\n✓ ALL TESTS PASSED - The data validation fixes are working correctly!")
        print("\nThe system should now handle malformed MCP responses gracefully.")
        sys.exit(0)
    else:
        print("\n✗ SOME TESTS FAILED - Please review the errors above")
        sys.exit(1)
