"""
FastMCP server for arXiv paper search and download operations.
Provides MCP-compliant tools via FastMCP framework with auto-start capability.
"""
import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import arxiv
import threading
import time
import urllib.request

# Import FastMCP
try:
    from fastmcp import FastMCP
    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False
    logging.warning("FastMCP not available. Install with: pip install fastmcp")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _extract_pdf_url(result: arxiv.Result) -> Optional[str]:
    """
    Extract PDF URL from arxiv.Result, handling arxiv library v2.2.0 breaking change.

    In arxiv v2.2.0+, pdf_url attribute is always None. PDF URL is now in links field.

    Args:
        result: arxiv.Result object

    Returns:
        PDF URL string or None if not found
    """
    # Try legacy pdf_url attribute first (backward compatibility)
    if result.pdf_url:
        return result.pdf_url

    # arxiv v2.2.0+: PDF URL is in links
    # Links typically have format:
    #   [0] abs URL (alternate)
    #   [1] pdf URL (alternate)
    #   [2] DOI URL (related)
    try:
        for link in result.links:
            if 'pdf' in link.href.lower():
                logger.debug(f"Extracted PDF URL from links: {link.href}")
                return link.href
    except (AttributeError, TypeError) as e:
        logger.warning(f"Error extracting PDF URL from links: {e}")

    # Fallback: construct URL from entry_id
    # entry_id format: http://arxiv.org/abs/2102.08370v2
    try:
        paper_id = result.entry_id.split('/')[-1]
        fallback_url = f"https://arxiv.org/pdf/{paper_id}"
        logger.warning(f"Using fallback PDF URL construction: {fallback_url}")
        return fallback_url
    except (AttributeError, IndexError) as e:
        logger.error(f"Failed to construct fallback PDF URL: {e}")
        return None


class ArxivFastMCPServer:
    """FastMCP server for arXiv operations with auto-start capability."""

    def __init__(
        self,
        storage_path: Optional[str] = None,
        server_port: int = 5555,
        auto_start: bool = True
    ):
        """
        Initialize FastMCP arXiv server.

        Args:
            storage_path: Directory to store downloaded papers
            server_port: Port for FastMCP server (default: 5555)
            auto_start: Whether to start server automatically
        """
        if not FASTMCP_AVAILABLE:
            raise ImportError("FastMCP not installed. Run: pip install fastmcp")

        self.storage_path = Path(storage_path or os.getenv("MCP_ARXIV_STORAGE_PATH", "data/mcp_papers"))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.server_port = server_port
        self.arxiv_client = arxiv.Client()

        # Initialize FastMCP server
        self.mcp = FastMCP("arxiv-server")

        # Register tools
        self._register_tools()

        # Server state
        self._server_thread = None
        self._running = False

        logger.info(f"ArxivFastMCPServer initialized with storage: {self.storage_path}")

        if auto_start:
            self.start()

    def _register_tools(self):
        """Register arXiv tools with FastMCP."""

        @self.mcp.tool()
        def search_papers(
            query: str,
            max_results: int = 5,
            categories: Optional[List[str]] = None,
            sort_by: str = "relevance"
        ) -> Dict[str, Any]:
            """
            Search for papers on arXiv.

            Args:
                query: Search query string
                max_results: Maximum number of papers to return (1-50)
                categories: Optional list of arXiv category filters (e.g., ['cs.AI'])
                sort_by: Sort criterion (relevance, lastUpdatedDate, submittedDate)

            Returns:
                Dictionary with 'papers' list containing paper metadata
            """
            try:
                logger.info(f"Searching arXiv: query='{query}', max_results={max_results}")

                # Build search query with category filter
                search_query = query
                if categories:
                    cat_filter = " OR ".join([f"cat:{cat}" for cat in categories])
                    search_query = f"({query}) AND ({cat_filter})"

                # Map sort_by to arxiv.SortCriterion
                sort_map = {
                    "relevance": arxiv.SortCriterion.Relevance,
                    "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
                    "submittedDate": arxiv.SortCriterion.SubmittedDate
                }
                sort_criterion = sort_map.get(sort_by, arxiv.SortCriterion.Relevance)

                # Create and execute search
                search = arxiv.Search(
                    query=search_query,
                    max_results=min(max_results, 50),
                    sort_by=sort_criterion
                )

                # Search.results() is deprecated/removed in newer arxiv
                # releases in favor of Client.results()
                papers = []
                for result in self.arxiv_client.results(search):
                    paper_data = {
                        "id": result.entry_id.split('/')[-1],
                        "title": result.title,
                        "authors": [author.name for author in result.authors],
                        "summary": result.summary,
                        "pdf_url": _extract_pdf_url(result),
                        "published": result.published.isoformat(),
                        "categories": result.categories
                    }
                    papers.append(paper_data)

                logger.info(f"Found {len(papers)} papers")
                return {"papers": papers, "count": len(papers)}

            except Exception as e:
                logger.error(f"Error searching arXiv: {str(e)}")
                return {"status": "error", "message": str(e), "papers": []}

        @self.mcp.tool()
        def download_paper(paper_id: str) -> Dict[str, Any]:
            """
            Download a paper PDF from arXiv.

            Args:
                paper_id: arXiv paper ID (e.g., '2401.00001')

            Returns:
                Dictionary with download status and file path
            """
            try:
                logger.info(f"Downloading paper: {paper_id}")

                # Check if already exists
                pdf_path = self.storage_path / f"{paper_id}.pdf"
                if pdf_path.exists():
                    logger.info(f"Paper {paper_id} already cached")
                    return {
                        "status": "cached",
                        "paper_id": paper_id,
                        "path": str(pdf_path),
                        "message": "Paper already in storage"
                    }

                # Get paper metadata to get PDF URL
                search = arxiv.Search(id_list=[paper_id])
                result = next(self.arxiv_client.results(search))

                # Extract PDF URL using helper (handles arxiv v2.2.0 breaking change)
                pdf_url = _extract_pdf_url(result)
                if not pdf_url:
                    raise ValueError(f"Could not extract PDF URL for paper {paper_id}")

                # Download PDF directly using urllib to avoid Path/str mixing issues
                headers = {'User-Agent': 'Mozilla/5.0 (FastMCP ArXiv Server)'}
                request = urllib.request.Request(pdf_url, headers=headers)

                with urllib.request.urlopen(request, timeout=30) as response:
                    pdf_content = response.read()

                # Write using pathlib to avoid any string/Path mixing
                pdf_path.write_bytes(pdf_content)

                logger.info(f"Successfully downloaded {paper_id} to {pdf_path}")
                return {
                    "status": "success",
                    "paper_id": paper_id,
                    "path": str(pdf_path),
                    "message": f"Downloaded to {pdf_path}"
                }

            except StopIteration:
                error_msg = f"Paper {paper_id} not found on arXiv"
                logger.error(error_msg)
                return {"status": "error", "paper_id": paper_id, "message": error_msg}
            except Exception as e:
                error_msg = f"Error downloading paper {paper_id}: {str(e)}"
                logger.error(error_msg)
                return {"status": "error", "paper_id": paper_id, "message": error_msg}

        @self.mcp.tool()
        def list_papers() -> Dict[str, Any]:
            """
            List all cached papers in storage.

            Returns:
                Dictionary with list of paper IDs in storage
            """
            try:
                pdf_files = list(self.storage_path.glob("*.pdf"))
                paper_ids = [f.stem for f in pdf_files]

                logger.info(f"Found {len(paper_ids)} cached papers")
                return {
                    "papers": paper_ids,
                    "count": len(paper_ids),
                    "storage_path": str(self.storage_path)
                }
            except Exception as e:
                logger.error(f"Error listing papers: {str(e)}")
                return {"status": "error", "message": str(e), "papers": []}

        logger.info("Registered FastMCP tools: search_papers, download_paper, list_papers")

    def start(self):
        """Start FastMCP server in background thread."""
        if self._running:
            logger.warning("Server already running")
            return

        def run_server():
            """Run FastMCP server with asyncio."""
            try:
                logger.info(f"Starting FastMCP arXiv server on port {self.server_port}")
                self._running = True

                # Run FastMCP server with SSE transport using async method
                # FastMCP 2.x provides run_sse_async for SSE servers
                import asyncio
                asyncio.run(self.mcp.run_sse_async(
                    host="localhost",
                    port=self.server_port,
                    log_level="INFO"
                ))

            except Exception as e:
                logger.error(f"Error running FastMCP server: {str(e)}", exc_info=True)
                self._running = False

        # Start server in daemon thread so it doesn't block app shutdown
        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()

        # Give server time to start
        time.sleep(1)
        logger.info("FastMCP arXiv server started in background")

    def stop(self):
        """Stop FastMCP server."""
        if not self._running:
            logger.warning("Server not running")
            return

        logger.info("Stopping FastMCP arXiv server")
        self._running = False

        # FastMCP should provide graceful shutdown
        # Implementation depends on FastMCP API
        if self._server_thread and self._server_thread.is_alive():
            # Wait for thread to finish (with timeout)
            self._server_thread.join(timeout=5)

        logger.info("FastMCP arXiv server stopped")

    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running and self._server_thread and self._server_thread.is_alive()

    def __enter__(self):
        """Context manager entry."""
        if not self._running:
            self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

    def __del__(self):
        """Cleanup on deletion."""
        try:
            if self._running:
                self.stop()
        except Exception:
            pass


# Singleton instance for application-wide use
_server_instance: Optional[ArxivFastMCPServer] = None


def get_server(
    storage_path: Optional[str] = None,
    server_port: int = 5555,
    auto_start: bool = True
) -> ArxivFastMCPServer:
    """
    Get or create singleton FastMCP server instance.

    Args:
        storage_path: Storage directory for papers
        server_port: Port for server
        auto_start: Auto-start server if not running

    Returns:
        ArxivFastMCPServer instance
    """
    global _server_instance

    if _server_instance is None:
        logger.info("Creating new FastMCP server instance")
        _server_instance = ArxivFastMCPServer(
            storage_path=storage_path,
            server_port=server_port,
            auto_start=auto_start
        )
    elif not _server_instance.is_running() and auto_start:
        logger.info("Restarting stopped FastMCP server")
        _server_instance.start()

    return _server_instance


def shutdown_server():
    """Shutdown singleton server instance."""
    global _server_instance

    if _server_instance:
        logger.info("Shutting down FastMCP server")
        _server_instance.stop()
        _server_instance = None


if __name__ == "__main__":
    # Test server in standalone mode
    import sys

    storage = sys.argv[1] if len(sys.argv) > 1 else "data/mcp_papers"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 5555

    logger.info(f"Starting standalone FastMCP arXiv server")
    logger.info(f"Storage: {storage}")
    logger.info(f"Port: {port}")

    server = ArxivFastMCPServer(
        storage_path=storage,
        server_port=port,
        auto_start=True
    )

    try:
        # Keep server running
        logger.info("Server running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.stop()
