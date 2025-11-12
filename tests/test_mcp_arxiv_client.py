"""
Unit tests for MCP arXiv Client.
"""
import os
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import Dict, Any
import json

from utils.mcp_arxiv_client import MCPArxivClient
from utils.schemas import Paper
from mcp.types import CallToolResult, TextContent


@pytest.fixture
def mock_mcp_session():
    """Create a mock MCP session."""
    session = AsyncMock()
    session.call_tool = AsyncMock()
    return session


@pytest.fixture
def mcp_client(tmp_path):
    """Create MCPArxivClient with temporary storage."""
    return MCPArxivClient(storage_path=str(tmp_path))


@pytest.fixture
def sample_mcp_paper_data():
    """Sample paper data as returned by MCP tools."""
    return {
        "id": "2401.00001",
        "title": "Deep Learning for Image Classification",
        "authors": ["John Doe", "Jane Smith"],
        "summary": "This paper presents a novel approach to image classification.",
        "pdf_url": "https://arxiv.org/pdf/2401.00001.pdf",
        "published": "2024-01-01T00:00:00Z",
        "categories": ["cs.CV", "cs.AI"]
    }


@pytest.fixture
def sample_paper():
    """Create a sample Paper object."""
    return Paper(
        arxiv_id="2401.00001",
        title="Deep Learning for Image Classification",
        authors=["John Doe", "Jane Smith"],
        abstract="This paper presents a novel approach to image classification.",
        pdf_url="https://arxiv.org/pdf/2401.00001.pdf",
        published=datetime(2024, 1, 1),
        categories=["cs.CV", "cs.AI"]
    )


class TestMCPArxivClient:
    """Test suite for MCPArxivClient."""

    def test_init(self, tmp_path):
        """Test client initialization."""
        client = MCPArxivClient(storage_path=str(tmp_path))
        assert client.storage_path == tmp_path
        assert tmp_path.exists()

    def test_init_default_path(self):
        """Test initialization with default storage path."""
        with patch.dict(os.environ, {"MCP_ARXIV_STORAGE_PATH": "data/test_mcp"}):
            client = MCPArxivClient()
            assert client.storage_path == Path("data/test_mcp")

    def test_parse_mcp_paper_success(self, mcp_client, sample_mcp_paper_data):
        """Test parsing MCP paper data into Paper object."""
        paper = mcp_client._parse_mcp_paper(sample_mcp_paper_data)

        assert isinstance(paper, Paper)
        assert paper.arxiv_id == "2401.00001"
        assert paper.title == "Deep Learning for Image Classification"
        assert paper.authors == ["John Doe", "Jane Smith"]
        assert paper.abstract == "This paper presents a novel approach to image classification."
        assert paper.pdf_url == "https://arxiv.org/pdf/2401.00001.pdf"
        assert paper.categories == ["cs.CV", "cs.AI"]

    def test_parse_mcp_paper_with_abstract_field(self, mcp_client):
        """Test parsing when MCP returns 'abstract' instead of 'summary'."""
        paper_data = {
            "id": "2401.00002",
            "title": "Test Paper",
            "authors": ["Author A"],
            "abstract": "Abstract text here",
            "published": "2024-01-01T00:00:00Z",
            "categories": ["cs.AI"]
        }
        paper = mcp_client._parse_mcp_paper(paper_data)
        assert paper.abstract == "Abstract text here"

    def test_parse_mcp_paper_missing_pdf_url(self, mcp_client):
        """Test parsing generates PDF URL if missing."""
        paper_data = {
            "id": "2401.00003",
            "title": "Test",
            "authors": ["Author"],
            "summary": "Summary",
            "published": "2024-01-01T00:00:00Z",
            "categories": ["cs.AI"]
        }
        paper = mcp_client._parse_mcp_paper(paper_data)
        assert paper.pdf_url == "https://arxiv.org/pdf/2401.00003.pdf"

    @pytest.mark.asyncio
    async def test_search_papers_async_success(self, mcp_client, sample_mcp_paper_data):
        """Test successful async paper search."""
        with patch.object(mcp_client, '_call_tool') as mock_call_tool:
            mock_call_tool.return_value = {
                "papers": [sample_mcp_paper_data]
            }

            papers = await mcp_client.search_papers_async("deep learning", max_results=5)

            assert len(papers) == 1
            assert papers[0].arxiv_id == "2401.00001"
            assert papers[0].title == "Deep Learning for Image Classification"

            # Verify tool was called correctly
            mock_call_tool.assert_called_once()
            call_args = mock_call_tool.call_args[0]
            assert call_args[0] == "search_papers"
            assert call_args[1]["query"] == "deep learning"
            assert call_args[1]["max_results"] == 5

    @pytest.mark.asyncio
    async def test_search_papers_async_with_category(self, mcp_client, sample_mcp_paper_data):
        """Test search with category filter."""
        with patch.object(mcp_client, '_call_tool') as mock_call_tool:
            mock_call_tool.return_value = {"papers": [sample_mcp_paper_data]}

            papers = await mcp_client.search_papers_async(
                "machine learning",
                max_results=3,
                category="cs.AI"
            )

            call_args = mock_call_tool.call_args[0]
            assert call_args[1]["category"] == "cs.AI"
            assert call_args[1]["max_results"] == 3

    @pytest.mark.asyncio
    async def test_search_papers_async_list_response(self, mcp_client, sample_mcp_paper_data):
        """Test handling MCP response as list instead of dict."""
        with patch.object(mcp_client, '_call_tool') as mock_call_tool:
            mock_call_tool.return_value = [sample_mcp_paper_data]

            papers = await mcp_client.search_papers_async("test")

            assert len(papers) == 1
            assert papers[0].arxiv_id == "2401.00001"

    @pytest.mark.asyncio
    async def test_search_papers_async_no_results(self, mcp_client):
        """Test search with no results."""
        with patch.object(mcp_client, '_call_tool') as mock_call_tool:
            mock_call_tool.return_value = {"papers": []}

            papers = await mcp_client.search_papers_async("nonexistent query")

            assert len(papers) == 0

    def test_search_papers_sync(self, mcp_client, sample_mcp_paper_data):
        """Test synchronous search_papers wrapper."""
        with patch.object(mcp_client, '_call_tool') as mock_call_tool:
            mock_call_tool.return_value = {"papers": [sample_mcp_paper_data]}

            papers = mcp_client.search_papers("test query")

            assert len(papers) == 1
            assert papers[0].arxiv_id == "2401.00001"

    @pytest.mark.asyncio
    async def test_download_paper_async_success(self, mcp_client, sample_paper, tmp_path):
        """Test successful paper download."""
        pdf_path = tmp_path / "2401.00001.pdf"

        with patch.object(mcp_client, '_call_tool') as mock_call_tool:
            mock_call_tool.return_value = {"status": "success"}

            # Create mock PDF file after the tool call (simulating MCP server behavior)
            def create_pdf(*args, **kwargs):
                pdf_path.write_text("mock pdf content")
                return {"status": "success"}

            mock_call_tool.side_effect = create_pdf

            result = await mcp_client.download_paper_async(sample_paper)

            assert result == pdf_path
            assert pdf_path.exists()

            # Verify tool was called
            mock_call_tool.assert_called_once()
            call_args = mock_call_tool.call_args[0]
            assert call_args[0] == "download_paper"
            assert call_args[1]["paper_id"] == "2401.00001"

    @pytest.mark.asyncio
    async def test_download_paper_async_already_cached(self, mcp_client, sample_paper, tmp_path):
        """Test downloading already cached paper."""
        # Create existing PDF
        pdf_path = tmp_path / "2401.00001.pdf"
        pdf_path.write_text("existing pdf")

        with patch.object(mcp_client, '_call_tool') as mock_call_tool:
            result = await mcp_client.download_paper_async(sample_paper)

            assert result == pdf_path
            # Should not call MCP tool if already cached
            mock_call_tool.assert_not_called()

    @pytest.mark.asyncio
    async def test_download_paper_async_failure(self, mcp_client, sample_paper):
        """Test download failure handling."""
        with patch.object(mcp_client, '_call_tool') as mock_call_tool:
            mock_call_tool.side_effect = Exception("Download failed")

            result = await mcp_client.download_paper_async(sample_paper)

            assert result is None

    def test_download_paper_sync(self, mcp_client, sample_paper, tmp_path):
        """Test synchronous download_paper wrapper."""
        pdf_path = tmp_path / "2401.00001.pdf"
        pdf_path.write_text("mock pdf")

        with patch.object(mcp_client, '_call_tool') as mock_call_tool:
            mock_call_tool.return_value = {"status": "success"}

            result = mcp_client.download_paper(sample_paper)

            assert result == pdf_path

    def test_download_papers_multiple(self, mcp_client, tmp_path):
        """Test downloading multiple papers."""
        papers = [
            Paper(
                arxiv_id=f"2401.0000{i}",
                title=f"Paper {i}",
                authors=["Author"],
                abstract="Abstract",
                pdf_url=f"https://arxiv.org/pdf/2401.0000{i}.pdf",
                published=datetime(2024, 1, 1),
                categories=["cs.AI"]
            )
            for i in range(1, 4)
        ]

        # Create mock PDFs
        for paper in papers:
            pdf_path = tmp_path / f"{paper.arxiv_id}.pdf"
            pdf_path.write_text("mock content")

        with patch.object(mcp_client, '_call_tool') as mock_call_tool:
            mock_call_tool.return_value = {"status": "success"}

            paths = mcp_client.download_papers(papers)

            assert len(paths) == 3
            assert all(isinstance(p, Path) for p in paths)

    @pytest.mark.asyncio
    async def test_get_cached_papers_async_success(self, mcp_client, tmp_path):
        """Test listing cached papers via MCP."""
        # Create mock cached PDFs
        pdf1 = tmp_path / "2401.00001.pdf"
        pdf2 = tmp_path / "2401.00002.pdf"
        pdf1.write_text("pdf1")
        pdf2.write_text("pdf2")

        with patch.object(mcp_client, '_call_tool') as mock_call_tool:
            mock_call_tool.return_value = {
                "papers": ["2401.00001", "2401.00002"]
            }

            paths = await mcp_client.get_cached_papers_async()

            assert len(paths) == 2
            assert pdf1 in paths
            assert pdf2 in paths

    @pytest.mark.asyncio
    async def test_get_cached_papers_async_fallback(self, mcp_client, tmp_path):
        """Test fallback to filesystem listing on MCP error."""
        # Create mock PDFs
        pdf1 = tmp_path / "2401.00001.pdf"
        pdf1.write_text("pdf1")

        with patch.object(mcp_client, '_call_tool') as mock_call_tool:
            mock_call_tool.side_effect = Exception("MCP error")

            paths = await mcp_client.get_cached_papers_async()

            # Should fall back to filesystem glob
            assert len(paths) == 1
            assert pdf1 in paths

    def test_get_cached_papers_sync(self, mcp_client, tmp_path):
        """Test synchronous get_cached_papers wrapper."""
        pdf1 = tmp_path / "2401.00001.pdf"
        pdf1.write_text("pdf")

        with patch.object(mcp_client, '_call_tool') as mock_call_tool:
            mock_call_tool.return_value = {"papers": ["2401.00001"]}

            paths = mcp_client.get_cached_papers()

            assert len(paths) == 1
            assert pdf1 in paths

    @pytest.mark.asyncio
    async def test_call_tool_error_handling(self, mcp_client):
        """Test error handling in _call_tool."""
        with patch.object(mcp_client, '_get_session') as mock_get_session:
            mock_session = AsyncMock()
            mock_session.call_tool.side_effect = Exception("Tool call failed")
            mock_get_session.return_value = mock_session

            with pytest.raises(Exception, match="Tool call failed"):
                await mcp_client._call_tool("test_tool", {})

    @pytest.mark.asyncio
    async def test_close_session(self, mcp_client):
        """Test closing MCP session."""
        mcp_client._session = AsyncMock()
        await mcp_client.close()
        assert mcp_client._session is None

    @pytest.mark.asyncio
    async def test_call_tool_with_calltoolresult(self, mcp_client, sample_mcp_paper_data):
        """Test _call_tool properly extracts data from CallToolResult."""
        with patch.object(mcp_client, '_get_session') as mock_get_session:
            # Create a mock CallToolResult like what real MCP server returns
            mock_result_data = {"papers": [sample_mcp_paper_data]}
            mock_text_content = TextContent(
                type="text",
                text=json.dumps(mock_result_data)
            )
            mock_call_tool_result = CallToolResult(
                content=[mock_text_content]
            )

            # Mock session to return CallToolResult
            mock_session = AsyncMock()
            mock_session.call_tool.return_value = mock_call_tool_result
            mock_get_session.return_value = mock_session

            # Call the tool
            result = await mcp_client._call_tool("search_papers", {"query": "test"})

            # Verify result was properly extracted and parsed
            assert isinstance(result, dict)
            assert "papers" in result
            assert len(result["papers"]) == 1
            assert result["papers"][0]["id"] == "2401.00001"

    @pytest.mark.asyncio
    async def test_search_papers_with_calltools_result(self, mcp_client, sample_mcp_paper_data):
        """Test search_papers_async works with CallToolResult from real MCP server."""
        with patch.object(mcp_client, '_get_session') as mock_get_session:
            # Simulate real MCP server returning CallToolResult
            mock_result_data = {"papers": [sample_mcp_paper_data]}
            mock_text_content = TextContent(
                type="text",
                text=json.dumps(mock_result_data)
            )
            mock_call_tool_result = CallToolResult(
                content=[mock_text_content]
            )

            mock_session = AsyncMock()
            mock_session.call_tool.return_value = mock_call_tool_result
            mock_get_session.return_value = mock_session

            # Search for papers
            papers = await mcp_client.search_papers_async("deep learning")

            # Verify papers were properly parsed
            assert len(papers) == 1
            assert papers[0].arxiv_id == "2401.00001"
            assert papers[0].title == "Deep Learning for Image Classification"
