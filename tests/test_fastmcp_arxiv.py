"""
Unit tests for FastMCP arXiv Server and Client.
"""
import os
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, AsyncMock, call
from typing import Dict, Any
import json

# Import components to test
from utils.schemas import Paper

# These imports may fail if fastmcp is not installed
pytest.importorskip("fastmcp", reason="fastmcp not installed")

from utils.fastmcp_arxiv_client import FastMCPArxivClient
from utils.fastmcp_arxiv_server import ArxivFastMCPServer


@pytest.fixture
def mock_fastmcp_client():
    """Create a mock FastMCP client."""
    mock_client = AsyncMock()
    mock_client.call_tool = AsyncMock()
    mock_client.close = AsyncMock()
    return mock_client


@pytest.fixture
def fastmcp_client(tmp_path):
    """Create FastMCPArxivClient with temporary storage."""
    client = FastMCPArxivClient(
        storage_path=str(tmp_path),
        server_host="localhost",
        server_port=5555
    )
    return client


@pytest.fixture
def sample_mcp_paper_data():
    """Sample paper data as returned by FastMCP tools."""
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


class TestFastMCPArxivClient:
    """Test suite for FastMCPArxivClient."""

    def test_init(self, tmp_path):
        """Test client initialization."""
        client = FastMCPArxivClient(
            storage_path=str(tmp_path),
            server_host="localhost",
            server_port=5555
        )
        assert client.storage_path == tmp_path
        assert client.server_host == "localhost"
        assert client.server_port == 5555
        assert client.server_url == "http://localhost:5555"
        assert tmp_path.exists()

    def test_init_default_path(self):
        """Test initialization with default storage path."""
        with patch.dict(os.environ, {"MCP_ARXIV_STORAGE_PATH": "data/test_mcp"}):
            client = FastMCPArxivClient()
            assert client.storage_path == Path("data/test_mcp")

    def test_parse_mcp_paper_success(self, fastmcp_client, sample_mcp_paper_data):
        """Test parsing MCP paper data into Paper object."""
        paper = fastmcp_client._parse_mcp_paper(sample_mcp_paper_data)

        assert isinstance(paper, Paper)
        assert paper.arxiv_id == "2401.00001"
        assert paper.title == "Deep Learning for Image Classification"
        assert paper.authors == ["John Doe", "Jane Smith"]
        assert paper.abstract == "This paper presents a novel approach to image classification."
        assert paper.pdf_url == "https://arxiv.org/pdf/2401.00001.pdf"
        assert paper.categories == ["cs.CV", "cs.AI"]

    def test_parse_mcp_paper_with_abstract_field(self, fastmcp_client):
        """Test parsing when MCP returns 'abstract' instead of 'summary'."""
        paper_data = {
            "id": "2401.00002",
            "title": "Test Paper",
            "authors": ["Author A"],
            "abstract": "Abstract text here",
            "published": "2024-01-01T00:00:00Z",
            "categories": ["cs.AI"]
        }
        paper = fastmcp_client._parse_mcp_paper(paper_data)
        assert paper.abstract == "Abstract text here"

    def test_parse_mcp_paper_missing_pdf_url(self, fastmcp_client):
        """Test parsing generates PDF URL if missing."""
        paper_data = {
            "id": "2401.00003",
            "title": "Test Paper",
            "authors": ["Author A"],
            "summary": "Test abstract",
            "published": "2024-01-01T00:00:00Z",
            "categories": ["cs.AI"]
        }
        paper = fastmcp_client._parse_mcp_paper(paper_data)
        assert paper.pdf_url == "https://arxiv.org/pdf/2401.00003.pdf"

    def test_parse_mcp_paper_dict_authors(self, fastmcp_client):
        """Test parsing when authors is a dict (edge case)."""
        paper_data = {
            "id": "2401.00004",
            "title": "Test Paper",
            "authors": {"names": ["Author A", "Author B"]},
            "summary": "Test abstract",
            "published": "2024-01-01T00:00:00Z",
            "categories": ["cs.AI"]
        }
        paper = fastmcp_client._parse_mcp_paper(paper_data)
        assert paper.authors == ["Author A", "Author B"]

    def test_parse_mcp_paper_string_authors(self, fastmcp_client):
        """Test parsing when authors is a string (edge case)."""
        paper_data = {
            "id": "2401.00005",
            "title": "Test Paper",
            "authors": "Single Author",
            "summary": "Test abstract",
            "published": "2024-01-01T00:00:00Z",
            "categories": ["cs.AI"]
        }
        paper = fastmcp_client._parse_mcp_paper(paper_data)
        assert paper.authors == ["Single Author"]

    def test_parse_mcp_paper_invalid_published_date(self, fastmcp_client):
        """Test parsing with invalid published date (should use current time)."""
        paper_data = {
            "id": "2401.00006",
            "title": "Test Paper",
            "authors": ["Author A"],
            "summary": "Test abstract",
            "published": "invalid-date",
            "categories": ["cs.AI"]
        }
        paper = fastmcp_client._parse_mcp_paper(paper_data)
        # Should default to current time without raising exception
        assert isinstance(paper.published, datetime)

    @pytest.mark.asyncio
    async def test_search_papers_async_success(self, fastmcp_client, mock_fastmcp_client, sample_mcp_paper_data):
        """Test successful async search."""
        # Mock client response
        mock_fastmcp_client.call_tool.return_value = {
            "papers": [sample_mcp_paper_data],
            "count": 1
        }

        # Patch _get_client to return mock
        with patch.object(fastmcp_client, '_get_client', return_value=mock_fastmcp_client):
            papers = await fastmcp_client.search_papers_async(
                query="machine learning",
                max_results=5,
                category="cs.AI"
            )

            assert len(papers) == 1
            assert papers[0].arxiv_id == "2401.00001"
            assert papers[0].title == "Deep Learning for Image Classification"

            # Verify tool was called with correct arguments
            mock_fastmcp_client.call_tool.assert_called_once()
            call_args = mock_fastmcp_client.call_tool.call_args
            assert call_args[0][0] == "search_papers"
            assert call_args[0][1]["query"] == "machine learning"
            assert call_args[0][1]["max_results"] == 5
            assert call_args[0][1]["categories"] == ["cs.AI"]

    @pytest.mark.asyncio
    async def test_search_papers_async_empty_results(self, fastmcp_client, mock_fastmcp_client):
        """Test search with no results."""
        mock_fastmcp_client.call_tool.return_value = {"papers": [], "count": 0}

        with patch.object(fastmcp_client, '_get_client', return_value=mock_fastmcp_client):
            papers = await fastmcp_client.search_papers_async(
                query="nonexistent topic",
                max_results=5
            )

            assert len(papers) == 0

    @pytest.mark.asyncio
    async def test_search_papers_async_malformed_response(self, fastmcp_client, mock_fastmcp_client):
        """Test search with malformed response."""
        mock_fastmcp_client.call_tool.return_value = "unexpected string response"

        with patch.object(fastmcp_client, '_get_client', return_value=mock_fastmcp_client):
            papers = await fastmcp_client.search_papers_async(
                query="test query",
                max_results=5
            )

            # Should handle gracefully and return empty list
            assert len(papers) == 0

    def test_search_papers_sync(self, fastmcp_client, sample_mcp_paper_data):
        """Test synchronous search wrapper."""
        # Mock the async method
        async def mock_search(*args, **kwargs):
            return [Paper(
                arxiv_id="2401.00001",
                title="Test Paper",
                authors=["Author A"],
                abstract="Test abstract",
                pdf_url="https://arxiv.org/pdf/2401.00001.pdf",
                published=datetime(2024, 1, 1),
                categories=["cs.AI"]
            )]

        with patch.object(fastmcp_client, 'search_papers_async', side_effect=mock_search):
            papers = fastmcp_client.search_papers(
                query="test query",
                max_results=5
            )

            assert len(papers) == 1
            assert papers[0].arxiv_id == "2401.00001"

    @pytest.mark.asyncio
    async def test_download_paper_async_cached(self, fastmcp_client, sample_paper, tmp_path):
        """Test downloading paper that's already cached."""
        # Create cached file
        pdf_path = tmp_path / f"{sample_paper.arxiv_id}.pdf"
        pdf_path.write_bytes(b"fake pdf content")

        path = await fastmcp_client.download_paper_async(sample_paper)
        assert path == pdf_path
        assert path.exists()

    @pytest.mark.asyncio
    async def test_download_paper_async_success(self, fastmcp_client, mock_fastmcp_client, sample_paper, tmp_path):
        """Test successful async download."""
        # Mock successful download response
        mock_fastmcp_client.call_tool.return_value = {
            "status": "success",
            "paper_id": sample_paper.arxiv_id,
            "path": str(tmp_path / f"{sample_paper.arxiv_id}.pdf")
        }

        # Create the file that FastMCP would create
        pdf_path = tmp_path / f"{sample_paper.arxiv_id}.pdf"
        pdf_path.write_bytes(b"downloaded pdf content")

        with patch.object(fastmcp_client, '_get_client', return_value=mock_fastmcp_client):
            path = await fastmcp_client.download_paper_async(sample_paper)

            assert path == pdf_path
            assert path.exists()

    @pytest.mark.asyncio
    async def test_download_paper_async_error_fallback(self, fastmcp_client, mock_fastmcp_client, sample_paper):
        """Test download with error triggers fallback."""
        # Mock error response
        mock_fastmcp_client.call_tool.return_value = {
            "status": "error",
            "message": "Paper not found"
        }

        # Mock direct download fallback
        with patch.object(fastmcp_client, '_get_client', return_value=mock_fastmcp_client), \
             patch.object(fastmcp_client, '_download_from_arxiv_direct', return_value=Path("fake.pdf")) as mock_fallback:

            path = await fastmcp_client.download_paper_async(sample_paper)

            # Verify fallback was called
            mock_fallback.assert_called_once_with(sample_paper)
            assert path == Path("fake.pdf")

    @pytest.mark.asyncio
    async def test_download_paper_async_file_not_found_fallback(self, fastmcp_client, mock_fastmcp_client, sample_paper, tmp_path):
        """Test download succeeds but file not found triggers fallback."""
        # Mock successful response but file doesn't exist
        mock_fastmcp_client.call_tool.return_value = {
            "status": "success",
            "paper_id": sample_paper.arxiv_id,
            "path": str(tmp_path / f"{sample_paper.arxiv_id}.pdf")
        }

        # Don't create the file

        # Mock direct download fallback
        with patch.object(fastmcp_client, '_get_client', return_value=mock_fastmcp_client), \
             patch.object(fastmcp_client, '_download_from_arxiv_direct', return_value=Path("fallback.pdf")) as mock_fallback:

            path = await fastmcp_client.download_paper_async(sample_paper)

            # Verify fallback was called
            mock_fallback.assert_called_once_with(sample_paper)

    def test_download_paper_sync(self, fastmcp_client, sample_paper):
        """Test synchronous download wrapper."""
        # Mock the async method
        async def mock_download(paper):
            return Path("test.pdf")

        with patch.object(fastmcp_client, 'download_paper_async', side_effect=mock_download):
            path = fastmcp_client.download_paper(sample_paper)
            assert path == Path("test.pdf")

    def test_download_papers(self, fastmcp_client):
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
            ) for i in range(1, 4)
        ]

        # Mock download_paper to return paths
        with patch.object(fastmcp_client, 'download_paper', side_effect=[
            Path("paper1.pdf"),
            None,  # Second download fails
            Path("paper3.pdf")
        ]):
            paths = fastmcp_client.download_papers(papers)

            # Should return only successful downloads
            assert len(paths) == 2
            assert paths[0] == Path("paper1.pdf")
            assert paths[1] == Path("paper3.pdf")

    @pytest.mark.asyncio
    async def test_get_cached_papers_async_success(self, fastmcp_client, mock_fastmcp_client, tmp_path):
        """Test getting cached papers list."""
        # Create some fake cached papers
        (tmp_path / "2401.00001.pdf").write_bytes(b"pdf1")
        (tmp_path / "2401.00002.pdf").write_bytes(b"pdf2")

        # Mock list_papers response
        mock_fastmcp_client.call_tool.return_value = {
            "papers": ["2401.00001", "2401.00002"],
            "count": 2
        }

        with patch.object(fastmcp_client, '_get_client', return_value=mock_fastmcp_client):
            paths = await fastmcp_client.get_cached_papers_async()

            assert len(paths) == 2
            assert all(p.exists() for p in paths)
            assert all(p.suffix == ".pdf" for p in paths)

    @pytest.mark.asyncio
    async def test_get_cached_papers_async_fallback(self, fastmcp_client, tmp_path):
        """Test get cached papers falls back to filesystem on error."""
        # Create some fake cached papers
        (tmp_path / "2401.00001.pdf").write_bytes(b"pdf1")
        (tmp_path / "2401.00002.pdf").write_bytes(b"pdf2")

        # Mock client to raise exception
        mock_client = AsyncMock()
        mock_client.call_tool.side_effect = Exception("Connection error")

        with patch.object(fastmcp_client, '_get_client', return_value=mock_client):
            paths = await fastmcp_client.get_cached_papers_async()

            # Should fall back to filesystem listing
            assert len(paths) == 2

    def test_get_cached_papers_sync(self, fastmcp_client):
        """Test synchronous get cached papers wrapper."""
        # Mock the async method
        async def mock_get_cached():
            return [Path("paper1.pdf"), Path("paper2.pdf")]

        with patch.object(fastmcp_client, 'get_cached_papers_async', side_effect=mock_get_cached):
            paths = fastmcp_client.get_cached_papers()
            assert len(paths) == 2

    def test_direct_download_fallback_success(self, fastmcp_client, sample_paper, tmp_path):
        """Test direct arXiv download fallback."""
        # Mock urllib download
        fake_pdf_content = b"PDF content from arXiv"

        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_response = MagicMock()
            mock_response.__enter__ = Mock(return_value=mock_response)
            mock_response.__exit__ = Mock(return_value=False)
            mock_response.read.return_value = fake_pdf_content
            mock_urlopen.return_value = mock_response

            path = fastmcp_client._download_from_arxiv_direct(sample_paper)

            assert path is not None
            assert path.exists()
            assert path.read_bytes() == fake_pdf_content

    def test_direct_download_fallback_http_error(self, fastmcp_client, sample_paper):
        """Test direct download fallback handles HTTP errors."""
        import urllib.error

        with patch('urllib.request.urlopen', side_effect=urllib.error.HTTPError(
            None, 404, "Not Found", None, None
        )):
            path = fastmcp_client._download_from_arxiv_direct(sample_paper)
            assert path is None

    @pytest.mark.asyncio
    async def test_close_async(self, fastmcp_client, mock_fastmcp_client):
        """Test async client cleanup."""
        fastmcp_client._client = mock_fastmcp_client
        fastmcp_client._client_initialized = True

        await fastmcp_client.close_async()

        mock_fastmcp_client.close.assert_called_once()
        assert fastmcp_client._client is None
        assert not fastmcp_client._client_initialized


class TestArxivFastMCPServer:
    """Test suite for ArxivFastMCPServer."""

    def test_server_init(self, tmp_path):
        """Test server initialization without auto-start."""
        with patch('utils.fastmcp_arxiv_server.FastMCP') as mock_fastmcp:
            server = ArxivFastMCPServer(
                storage_path=str(tmp_path),
                server_port=5555,
                auto_start=False
            )

            assert server.storage_path == tmp_path
            assert server.server_port == 5555
            assert not server._running
            mock_fastmcp.assert_called_once_with("arxiv-server")

    def test_server_register_tools(self, tmp_path):
        """Test that server registers tools on init."""
        with patch('utils.fastmcp_arxiv_server.FastMCP') as mock_fastmcp:
            mock_mcp_instance = MagicMock()
            mock_fastmcp.return_value = mock_mcp_instance

            server = ArxivFastMCPServer(
                storage_path=str(tmp_path),
                server_port=5555,
                auto_start=False
            )

            # Verify tool decorator was called (tools registered)
            assert mock_mcp_instance.tool.called

    def test_server_context_manager(self, tmp_path):
        """Test server as context manager."""
        with patch('utils.fastmcp_arxiv_server.FastMCP'):
            server = ArxivFastMCPServer(
                storage_path=str(tmp_path),
                server_port=5555,
                auto_start=False
            )

            with patch.object(server, 'start') as mock_start, \
                 patch.object(server, 'stop') as mock_stop:

                with server:
                    mock_start.assert_called_once()

                mock_stop.assert_called_once()


class TestFastMCPIntegration:
    """Integration tests for FastMCP components."""

    def test_client_server_compatibility(self, tmp_path):
        """Test that client and server have compatible interfaces."""
        # Create client
        client = FastMCPArxivClient(
            storage_path=str(tmp_path),
            server_host="localhost",
            server_port=5555
        )

        # Verify client has required methods
        assert hasattr(client, 'search_papers')
        assert hasattr(client, 'download_paper')
        assert hasattr(client, 'download_papers')
        assert hasattr(client, 'get_cached_papers')

        # Verify client implements ArxivClient interface
        assert callable(client.search_papers)
        assert callable(client.download_paper)
        assert callable(client.download_papers)
        assert callable(client.get_cached_papers)
