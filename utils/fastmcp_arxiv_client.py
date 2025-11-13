"""
FastMCP client for accessing arXiv papers via FastMCP protocol.
Implements same interface as ArxivClient for drop-in compatibility.
"""
import os
import logging
from typing import List, Optional, Any, Dict
from pathlib import Path
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential
import json
import asyncio
import nest_asyncio
import urllib.request
import urllib.error

from utils.schemas import Paper

# Import FastMCP client
try:
    from fastmcp import Client
    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False
    logging.warning("FastMCP not available. Install with: pip install fastmcp")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FastMCPArxivClient:
    """FastMCP client for arXiv operations with ArxivClient-compatible interface."""

    def __init__(
        self,
        storage_path: Optional[str] = None,
        server_host: str = "localhost",
        server_port: int = 5555
    ):
        """
        Initialize FastMCP arXiv client.

        Args:
            storage_path: Path where papers are stored (for local file access)
            server_host: FastMCP server host
            server_port: FastMCP server port
        """
        if not FASTMCP_AVAILABLE:
            raise ImportError("FastMCP not installed. Run: pip install fastmcp")

        self.storage_path = Path(storage_path or os.getenv("MCP_ARXIV_STORAGE_PATH", "data/mcp_papers"))
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.server_host = server_host
        self.server_port = server_port
        # FastMCP SSE server uses /sse endpoint by default
        self.server_url = f"http://{server_host}:{server_port}/sse"

        logger.info(f"FastMCPArxivClient initialized")
        logger.info(f"Storage path: {self.storage_path}")
        logger.info(f"Server: {self.server_url}")

    def _parse_mcp_paper(self, paper_data: Dict[str, Any]) -> Paper:
        """
        Convert MCP tool response to Paper object with robust type validation.
        Reused from legacy MCP client for consistency.

        Args:
            paper_data: Paper data from MCP tool

        Returns:
            Paper object with validated and normalized fields

        Raises:
            Exception: If critical fields are missing or invalid
        """
        try:
            # MCP server returns papers with these fields
            arxiv_id = paper_data.get("id") or paper_data.get("arxiv_id", "")
            if not arxiv_id:
                raise ValueError("Missing required field: arxiv_id")

            # Parse published date with robust error handling
            published_str = paper_data.get("published", "")
            if isinstance(published_str, str):
                try:
                    published = datetime.fromisoformat(published_str.replace('Z', '+00:00'))
                except Exception as e:
                    logger.warning(f"Failed to parse published date '{published_str}': {e}, using current time")
                    published = datetime.now()
            elif isinstance(published_str, datetime):
                published = published_str
            else:
                logger.warning(f"Published field has unexpected type: {type(published_str)}, using current time")
                published = datetime.now()

            # Normalize authors field
            authors_raw = paper_data.get("authors", [])
            if isinstance(authors_raw, list):
                authors = [str(author) if not isinstance(author, str) else author for author in authors_raw]
            elif isinstance(authors_raw, dict):
                logger.warning(f"Authors field is dict for paper {arxiv_id}: {authors_raw}")
                if 'names' in authors_raw:
                    authors = authors_raw['names'] if isinstance(authors_raw['names'], list) else [str(authors_raw['names'])]
                else:
                    authors = [str(val) for val in authors_raw.values() if val]
            elif isinstance(authors_raw, str):
                authors = [authors_raw]
            else:
                logger.warning(f"Unexpected authors format for paper {arxiv_id}: {type(authors_raw)}")
                authors = []

            # Normalize categories field
            categories_raw = paper_data.get("categories", [])
            if isinstance(categories_raw, list):
                categories = [str(cat) if not isinstance(cat, str) else cat for cat in categories_raw]
            elif isinstance(categories_raw, dict):
                logger.warning(f"Categories field is dict for paper {arxiv_id}: {categories_raw}")
                if 'categories' in categories_raw:
                    categories = categories_raw['categories'] if isinstance(categories_raw['categories'], list) else [str(categories_raw['categories'])]
                else:
                    categories = [str(val) for val in categories_raw.values() if val]
            elif isinstance(categories_raw, str):
                categories = [categories_raw]
            else:
                logger.warning(f"Unexpected categories format for paper {arxiv_id}: {type(categories_raw)}")
                categories = []

            # Normalize title field
            title_raw = paper_data.get("title", "")
            if isinstance(title_raw, dict):
                logger.warning(f"Title field is dict for paper {arxiv_id}: {title_raw}")
                title = title_raw.get("title") or str(title_raw)
            else:
                title = str(title_raw) if title_raw else ""

            # Normalize abstract field
            abstract_raw = paper_data.get("summary") or paper_data.get("abstract", "")
            if isinstance(abstract_raw, dict):
                logger.warning(f"Abstract field is dict for paper {arxiv_id}: {abstract_raw}")
                abstract = abstract_raw.get("abstract") or abstract_raw.get("summary") or str(abstract_raw)
            else:
                abstract = str(abstract_raw) if abstract_raw else ""

            # Normalize PDF URL field
            pdf_url_raw = paper_data.get("pdf_url")
            if pdf_url_raw:
                if isinstance(pdf_url_raw, dict):
                    logger.warning(f"pdf_url field is dict for paper {arxiv_id}: {pdf_url_raw}")
                    pdf_url = pdf_url_raw.get("url") or pdf_url_raw.get("pdf_url") or f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                else:
                    pdf_url = str(pdf_url_raw)
            else:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

            # Create Paper object (Pydantic validators provide additional validation)
            paper = Paper(
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                abstract=abstract,
                pdf_url=pdf_url,
                published=published,
                categories=categories
            )

            logger.debug(f"Successfully parsed paper {arxiv_id}: {len(authors)} authors, {len(categories)} categories")
            return paper

        except Exception as e:
            logger.error(f"Error parsing MCP paper data: {str(e)}")
            logger.error(f"Raw paper data: {paper_data}")
            raise

    def _download_from_arxiv_direct(self, paper: Paper) -> Optional[Path]:
        """
        Fallback method to download PDF directly from arXiv.
        Used when FastMCP server fails.

        Args:
            paper: Paper object

        Returns:
            Path to downloaded PDF, or None if download fails
        """
        try:
            pdf_path = self.storage_path / f"{paper.arxiv_id}.pdf"

            logger.info(f"Attempting direct download from arXiv for {paper.arxiv_id}")
            logger.debug(f"PDF URL: {paper.pdf_url}")

            # Download with urllib
            headers = {'User-Agent': 'Mozilla/5.0 (Research Paper Analysis System)'}
            request = urllib.request.Request(paper.pdf_url, headers=headers)

            with urllib.request.urlopen(request, timeout=30) as response:
                pdf_content = response.read()

            # Write to storage
            pdf_path.write_bytes(pdf_content)
            logger.info(f"Successfully downloaded {len(pdf_content)} bytes to {pdf_path}")

            return pdf_path

        except urllib.error.HTTPError as e:
            logger.error(f"HTTP error downloading from arXiv: {e.code} {e.reason}")
            return None
        except urllib.error.URLError as e:
            logger.error(f"URL error downloading from arXiv: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in direct arXiv download: {str(e)}", exc_info=True)
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def search_papers_async(
        self,
        query: str,
        max_results: int = 5,
        category: Optional[str] = None,
        sort_by: str = "relevance"
    ) -> List[Paper]:
        """
        Search for papers on arXiv using FastMCP.

        Args:
            query: Search query
            max_results: Maximum number of papers to return
            category: Optional arXiv category filter (e.g., 'cs.AI')
            sort_by: Sort criterion (relevance, lastUpdatedDate, submittedDate)

        Returns:
            List of Paper objects

        Raises:
            Exception: If FastMCP call fails after retries
        """
        try:
            logger.info(f"Searching arXiv via FastMCP for: {query}")

            # Prepare tool arguments
            tool_args = {
                "query": query,
                "max_results": max_results,
                "sort_by": sort_by
            }

            # Add category filter if provided
            if category:
                tool_args["categories"] = [category]

            # Call search_papers tool via FastMCP client context manager
            logger.debug(f"Calling search_papers tool with args: {tool_args}")
            async with Client(self.server_url) as client:
                result = await client.call_tool("search_papers", tool_args)

            # Parse results - FastMCP returns CallToolResult with data attribute
            papers = []
            # Extract data from CallToolResult object
            if hasattr(result, 'data') and result.data:
                result_data = result.data
            else:
                result_data = result

            # Now parse the actual data
            if isinstance(result_data, dict):
                paper_list = result_data.get("papers", [])
            elif isinstance(result_data, list):
                paper_list = result_data
            else:
                logger.warning(f"Unexpected result format: {type(result_data)}")
                paper_list = []

            # Parse each paper
            for paper_data in paper_list:
                try:
                    paper = self._parse_mcp_paper(paper_data)
                    papers.append(paper)
                except Exception as e:
                    logger.warning(f"Failed to parse paper: {str(e)}")
                    continue

            logger.info(f"Found {len(papers)} papers via FastMCP")
            return papers

        except Exception as e:
            logger.error(f"Error searching arXiv via FastMCP: {str(e)}")
            raise

    def search_papers(
        self,
        query: str,
        max_results: int = 5,
        category: Optional[str] = None,
        sort_by: str = "relevance"
    ) -> List[Paper]:
        """
        Synchronous wrapper for search_papers_async.

        Args:
            query: Search query
            max_results: Maximum number of papers to return
            category: Optional arXiv category filter
            sort_by: Sort criterion

        Returns:
            List of Paper objects
        """
        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Enable nested event loops for Gradio compatibility
        nest_asyncio.apply(loop)

        return loop.run_until_complete(
            self.search_papers_async(query, max_results, category, sort_by)
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def download_paper_async(self, paper: Paper) -> Optional[Path]:
        """
        Download paper PDF using FastMCP.

        Args:
            paper: Paper object

        Returns:
            Path to downloaded PDF, or None if download fails
        """
        try:
            # Expected path in storage
            pdf_path = self.storage_path / f"{paper.arxiv_id}.pdf"

            # Check if already exists locally
            if pdf_path.exists():
                logger.info(f"Paper {paper.arxiv_id} already in storage")
                return pdf_path

            logger.info(f"Downloading paper {paper.arxiv_id} via FastMCP")

            # Call download_paper tool via FastMCP client context manager
            async with Client(self.server_url) as client:
                result = await client.call_tool("download_paper", {"paper_id": paper.arxiv_id})

            # Extract data from CallToolResult object
            if hasattr(result, 'data') and result.data:
                result_data = result.data
            else:
                result_data = result

            logger.debug(f"FastMCP download response: {result_data}")

            # Check for error in response
            if isinstance(result_data, dict):
                if result_data.get("status") == "error":
                    error_msg = result_data.get("message", "Unknown error")
                    logger.error(f"FastMCP download failed for {paper.arxiv_id}: {error_msg}")
                    # Fall back to direct download
                    return self._download_from_arxiv_direct(paper)

            # Check if file exists locally now
            if pdf_path.exists():
                logger.info(f"Successfully downloaded paper to {pdf_path}")
                return pdf_path

            # Search for file in storage
            storage_files = list(self.storage_path.glob("*.pdf"))
            matching_files = [f for f in storage_files if paper.arxiv_id in f.name]
            if matching_files:
                found_file = matching_files[0]
                logger.info(f"Found downloaded file: {found_file}")
                return found_file

            # File not found - fall back to direct download
            logger.warning(f"FastMCP download completed but PDF not found for {paper.arxiv_id}")
            logger.warning("Falling back to direct arXiv download...")
            return self._download_from_arxiv_direct(paper)

        except Exception as e:
            logger.error(f"Error downloading paper {paper.arxiv_id} via FastMCP: {str(e)}", exc_info=True)
            logger.warning("Attempting direct arXiv download as fallback...")
            return self._download_from_arxiv_direct(paper)

    def download_paper(self, paper: Paper) -> Optional[Path]:
        """
        Synchronous wrapper for download_paper_async.

        Args:
            paper: Paper object

        Returns:
            Path to downloaded PDF
        """
        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Enable nested event loops for Gradio compatibility
        nest_asyncio.apply(loop)

        return loop.run_until_complete(self.download_paper_async(paper))

    def download_papers(self, papers: List[Paper]) -> List[Path]:
        """
        Download multiple papers.

        Args:
            papers: List of Paper objects

        Returns:
            List of Paths to downloaded PDFs
        """
        paths = []
        for paper in papers:
            path = self.download_paper(paper)
            if path:
                paths.append(path)
        return paths

    async def get_cached_papers_async(self) -> List[Path]:
        """
        Get list of cached paper PDFs using FastMCP.

        Returns:
            List of Paths to cached PDFs
        """
        try:
            # Call list_papers tool via FastMCP client context manager
            async with Client(self.server_url) as client:
                result = await client.call_tool("list_papers", {})

            # Extract data from CallToolResult object
            if hasattr(result, 'data') and result.data:
                result_data = result.data
            else:
                result_data = result

            # Parse result
            if isinstance(result_data, dict):
                paper_ids = result_data.get("papers", [])
            elif isinstance(result_data, list):
                paper_ids = result_data
            else:
                logger.warning("Unexpected format from list_papers")
                paper_ids = []

            # Convert to paths
            paths = [self.storage_path / f"{pid}.pdf" for pid in paper_ids
                    if (self.storage_path / f"{pid}.pdf").exists()]

            return paths

        except Exception as e:
            logger.warning(f"Error listing cached papers via FastMCP: {str(e)}")
            # Fallback to filesystem listing
            return list(self.storage_path.glob("*.pdf"))

    def get_cached_papers(self) -> List[Path]:
        """
        Synchronous wrapper for get_cached_papers_async.

        Returns:
            List of Paths to cached PDFs
        """
        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Enable nested event loops for Gradio compatibility
        nest_asyncio.apply(loop)

        return loop.run_until_complete(self.get_cached_papers_async())

    async def close_async(self):
        """
        Close FastMCP client connection.

        Note: With per-operation context managers, there is no persistent
        connection to close. Each operation manages its own connection lifecycle.
        """
        logger.info("FastMCP client uses per-operation connections - no persistent connection to close")

    def close(self):
        """
        Synchronous wrapper for close_async.

        Note: With per-operation context managers, there is no persistent
        connection to close. Each operation manages its own connection lifecycle.
        """
        logger.info("FastMCP client uses per-operation connections - no persistent connection to close")

    def __del__(self):
        """
        Cleanup on deletion.

        Note: With per-operation context managers, no cleanup is needed.
        Each operation manages its own connection lifecycle.
        """
        pass  # No cleanup needed with per-operation context managers
