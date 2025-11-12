"""
arXiv MCP client wrapper for accessing arXiv papers via Model Context Protocol.
"""
import os
import logging
from typing import List, Optional, Any, Dict
from pathlib import Path
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential
import json

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult
import urllib.request
import urllib.error

from utils.schemas import Paper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MCPArxivClient:
    """Wrapper for arXiv MCP server with error handling and caching."""

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize MCP arXiv client.

        Args:
            storage_path: Path where MCP server stores papers (reads from env if not provided)
        """
        self.storage_path = Path(storage_path or os.getenv("MCP_ARXIV_STORAGE_PATH", "data/mcp_papers"))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._session: Optional[ClientSession] = None
        self._stdio_context: Optional[Any] = None
        logger.info(f"MCPArxivClient initialized with storage path: {self.storage_path.resolve()}")

        # Log existing files in storage
        existing_files = list(self.storage_path.glob("*.pdf"))
        logger.info(f"Storage directory contains {len(existing_files)} existing PDF files")

    async def _get_session(self) -> ClientSession:
        """
        Get or create MCP client session.

        Returns:
            Active MCP ClientSession
        """
        if self._session is None:
            logger.info("Initializing connection to MCP arXiv server...")

            # Connect to external MCP server via stdio
            # Assumes MCP server is configured in Claude Desktop or running externally
            server_params = StdioServerParameters(
                command="arxiv-mcp-server",
                args=["--storage-path", str(self.storage_path.resolve())],
                env={}
            )

            logger.info(f"MCP server command: {server_params.command}")
            logger.info(f"MCP server args: {server_params.args}")

            # stdio_client returns a context manager that yields (read_stream, write_stream)
            # We need to unpack these streams and use them to create a ClientSession
            self._stdio_context = stdio_client(server_params)
            read_stream, write_stream = await self._stdio_context.__aenter__()

            # Create and initialize the ClientSession with the streams
            self._session = ClientSession(read_stream, write_stream)
            await self._session.__aenter__()

            # Initialize the session with the server (performs capability negotiation)
            await self._session.initialize()

            logger.info("Connected to arXiv MCP server and initialization complete")
            logger.info(f"MCP server will use storage path: {self.storage_path.resolve()}")

            # Discover available tools for debugging
            await self._discover_tools()

        return self._session

    async def _discover_tools(self):
        """
        Discover and log available MCP tools.
        This helps diagnose what capabilities the server provides.
        """
        try:
            if self._session:
                # List available tools from the server
                tools_result = await self._session.list_tools()

                if hasattr(tools_result, 'tools'):
                    available_tools = tools_result.tools
                    logger.info(f"MCP server provides {len(available_tools)} tools:")
                    for tool in available_tools:
                        tool_name = tool.name if hasattr(tool, 'name') else str(tool)
                        tool_desc = tool.description if hasattr(tool, 'description') else "No description"
                        logger.info(f"  - {tool_name}: {tool_desc}")

                        # Log tool schema if available
                        if hasattr(tool, 'inputSchema'):
                            logger.debug(f"    Schema: {tool.inputSchema}")
                else:
                    logger.warning("Could not retrieve tool list from MCP server")
        except Exception as e:
            logger.warning(f"Error discovering MCP tools: {str(e)}")

    def _download_from_arxiv_direct(self, paper: Paper) -> Optional[Path]:
        """
        Fallback method to download PDF directly from arXiv.
        Used when MCP server download fails or file is not accessible.

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

    async def _call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call an MCP tool and return the result.

        Args:
            tool_name: Name of the MCP tool to call
            arguments: Tool arguments as dictionary

        Returns:
            Tool result (parsed from CallToolResult if needed)

        Raises:
            Exception: If tool call fails
        """
        try:
            session = await self._get_session()
            result = await session.call_tool(tool_name, arguments)
            logger.debug(f"MCP tool {tool_name} called successfully")

            # Extract data from CallToolResult if that's what we got
            if isinstance(result, CallToolResult):
                if result.content and len(result.content) > 0:
                    # Extract text from first content item
                    content_item = result.content[0]

                    # Robust extraction - handle different content types
                    try:
                        # Try to get text attribute (standard TextContent)
                        if hasattr(content_item, 'text'):
                            text_content = content_item.text
                        # Fallback to dict access if it's a dict-like object
                        elif isinstance(content_item, dict) and 'text' in content_item:
                            text_content = content_item['text']
                        # If content_item is already a string
                        elif isinstance(content_item, str):
                            text_content = content_item
                        else:
                            logger.error(f"Unexpected content item type: {type(content_item)}, value: {content_item}")
                            return {"error": f"Cannot extract text from content type {type(content_item)}"}

                        # Ensure text_content is a string
                        if not isinstance(text_content, str):
                            logger.error(f"Text content is not a string: {type(text_content)}")
                            return {"error": f"Text content has wrong type: {type(text_content)}"}

                        # Log the raw response for debugging
                        logger.debug(f"Raw MCP response text: {text_content[:200]}...")

                        # Try to parse as JSON
                        try:
                            parsed_data = json.loads(text_content)
                            logger.debug(f"Extracted data from CallToolResult: {type(parsed_data)}")

                            # Check if response contains an error
                            if isinstance(parsed_data, dict) and "error" in parsed_data:
                                logger.error(f"MCP tool {tool_name} returned error: {parsed_data['error']}")

                            return parsed_data
                        except json.JSONDecodeError:
                            logger.warning(f"Could not parse CallToolResult content as JSON: {text_content[:200]}")
                            # Return as plain text if not JSON
                            return text_content

                    except Exception as extraction_error:
                        logger.error(f"Error extracting content from CallToolResult: {str(extraction_error)}")
                        logger.error(f"Content item type: {type(content_item)}, dir: {dir(content_item)}")
                        return {"error": f"Content extraction failed: {str(extraction_error)}"}
                else:
                    logger.warning("CallToolResult has no content")
                    return {}

            # Return as-is if not CallToolResult (for backward compatibility)
            return result
        except Exception as e:
            logger.error(f"Error calling MCP tool {tool_name}: {str(e)}")
            raise

    def _parse_mcp_paper(self, paper_data: Dict[str, Any]) -> Paper:
        """
        Convert MCP tool response to Paper object.

        Args:
            paper_data: Paper data from MCP tool

        Returns:
            Paper object
        """
        try:
            # MCP server returns papers with these fields
            # Handle potential variations in response format
            arxiv_id = paper_data.get("id") or paper_data.get("arxiv_id", "")

            # Parse published date
            published_str = paper_data.get("published", "")
            if isinstance(published_str, str):
                try:
                    published = datetime.fromisoformat(published_str.replace('Z', '+00:00'))
                except:
                    published = datetime.now()
            else:
                published = published_str or datetime.now()

            # Construct PDF URL
            pdf_url = paper_data.get("pdf_url") or f"https://arxiv.org/pdf/{arxiv_id}.pdf"

            paper = Paper(
                arxiv_id=arxiv_id,
                title=paper_data.get("title", ""),
                authors=paper_data.get("authors", []),
                abstract=paper_data.get("summary") or paper_data.get("abstract", ""),
                pdf_url=pdf_url,
                published=published,
                categories=paper_data.get("categories", [])
            )
            return paper
        except Exception as e:
            logger.error(f"Error parsing MCP paper data: {str(e)}")
            raise

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
        Search for papers on arXiv using MCP server.

        Args:
            query: Search query
            max_results: Maximum number of papers to return
            category: Optional arXiv category filter (e.g., 'cs.AI')
            sort_by: Sort criterion (relevance, lastUpdatedDate, submittedDate)

        Returns:
            List of Paper objects

        Raises:
            Exception: If MCP tool call fails after retries
        """
        try:
            logger.info(f"Searching arXiv via MCP for: {query}")

            # Prepare MCP tool arguments
            search_args = {
                "query": query,
                "max_results": max_results,
                "sort_by": sort_by
            }

            if category:
                search_args["category"] = category

            # Call MCP search_papers tool
            result = await self._call_tool("search_papers", search_args)

            # Parse results - _call_tool now handles CallToolResult extraction
            papers = []
            if isinstance(result, dict):
                paper_list = result.get("papers", [])
            elif isinstance(result, list):
                paper_list = result
            else:
                logger.warning(f"Unexpected result format after extraction: {type(result)}")
                paper_list = []

            for paper_data in paper_list:
                try:
                    paper = self._parse_mcp_paper(paper_data)
                    papers.append(paper)
                except Exception as e:
                    logger.warning(f"Failed to parse paper: {str(e)}")
                    continue

            logger.info(f"Found {len(papers)} papers via MCP")
            return papers

        except Exception as e:
            logger.error(f"Error searching arXiv via MCP: {str(e)}")
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
        import asyncio
        import nest_asyncio

        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
            # Check if loop is closed
            if loop.is_closed():
                # Create new loop if closed
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            # Create new event loop if none exists
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Enable nested event loops for compatibility
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
        Download paper PDF using MCP server.

        Args:
            paper: Paper object

        Returns:
            Path to downloaded PDF, or None if download fails
        """
        try:
            # Expected path in MCP server's storage
            pdf_path = self.storage_path / f"{paper.arxiv_id}.pdf"

            # Check if already exists
            if pdf_path.exists():
                logger.info(f"Paper {paper.arxiv_id} already in MCP storage")
                return pdf_path

            logger.info(f"Downloading paper {paper.arxiv_id} via MCP")
            logger.debug(f"Expected download path: {pdf_path}")

            # Call MCP download_paper tool
            result = await self._call_tool("download_paper", {
                "paper_id": paper.arxiv_id
            })

            # Log the MCP response for debugging
            logger.info(f"MCP download_paper response type: {type(result)}")
            if isinstance(result, dict):
                logger.info(f"MCP response keys: {list(result.keys())}")
            logger.debug(f"MCP response content: {result}")

            # Check if MCP returned an error response
            if isinstance(result, dict) and "error" in result:
                error_msg = result.get("error", "Unknown error")
                logger.error(f"MCP download failed for {paper.arxiv_id}: {error_msg}")
                return None
            elif isinstance(result, str) and "error" in result.lower():
                logger.error(f"MCP download failed for {paper.arxiv_id}: {result}")
                return None

            # Check if MCP returned a file path
            returned_path = None
            if isinstance(result, dict):
                # Try various possible field names for file path
                returned_path = result.get("file_path") or result.get("path") or result.get("pdf_path")
                if returned_path:
                    returned_path = Path(returned_path)
                    logger.info(f"MCP returned file path: {returned_path}")

            # Verify file exists at expected path
            if pdf_path.exists():
                logger.info(f"Successfully downloaded paper to {pdf_path}")
                return pdf_path

            # Check if file exists at returned path (if different)
            if returned_path and returned_path != pdf_path and returned_path.exists():
                logger.info(f"File found at MCP-returned path: {returned_path}")
                return returned_path

            # Search for the file in storage directory
            logger.warning(f"File not found at expected path {pdf_path}")
            logger.info(f"Searching storage directory: {self.storage_path}")

            # List all PDFs in storage
            storage_files = list(self.storage_path.glob("*.pdf"))
            logger.info(f"Storage contains {len(storage_files)} PDF files")

            # Try to find file matching arxiv_id
            matching_files = [f for f in storage_files if paper.arxiv_id in f.name]
            if matching_files:
                found_file = matching_files[0]
                logger.info(f"Found matching file: {found_file}")
                return found_file

            # File not found anywhere - try direct download as fallback
            logger.error(f"MCP download call completed but file not found for {paper.arxiv_id}")
            logger.error(f"Checked paths: {pdf_path}, Storage files: {[f.name for f in storage_files[:5]]}")
            logger.warning("Falling back to direct arXiv download...")

            # Fallback to direct download
            return self._download_from_arxiv_direct(paper)

        except Exception as e:
            logger.error(f"Error downloading paper {paper.arxiv_id} via MCP: {str(e)}", exc_info=True)
            logger.warning("Attempting direct arXiv download as fallback...")

            # Try direct download on any MCP error
            try:
                return self._download_from_arxiv_direct(paper)
            except Exception as fallback_error:
                logger.error(f"Direct download fallback also failed: {str(fallback_error)}")
                return None

    def download_paper(self, paper: Paper) -> Optional[Path]:
        """
        Synchronous wrapper for download_paper_async.

        Args:
            paper: Paper object

        Returns:
            Path to downloaded PDF
        """
        import asyncio
        import nest_asyncio

        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
            # Check if loop is closed
            if loop.is_closed():
                # Create new loop if closed
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            # Create new event loop if none exists
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Enable nested event loops for compatibility
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
        Get list of cached paper PDFs from MCP server.

        Returns:
            List of Paths to cached PDFs
        """
        try:
            # Call MCP list_papers tool
            result = await self._call_tool("list_papers", {})

            # Parse result to get paths
            if isinstance(result, dict):
                paper_ids = result.get("papers", [])
            elif isinstance(result, list):
                paper_ids = result
            else:
                logger.warning("Unexpected format from list_papers")
                paper_ids = []

            # Convert to paths
            paths = [self.storage_path / f"{pid}.pdf" for pid in paper_ids
                    if (self.storage_path / f"{pid}.pdf").exists()]

            return paths
        except Exception as e:
            logger.warning(f"Error listing cached papers via MCP: {str(e)}")
            # Fallback to filesystem listing
            return list(self.storage_path.glob("*.pdf"))

    def get_cached_papers(self) -> List[Path]:
        """
        Synchronous wrapper for get_cached_papers_async.

        Returns:
            List of Paths to cached PDFs
        """
        import asyncio
        import nest_asyncio

        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
            # Check if loop is closed
            if loop.is_closed():
                # Create new loop if closed
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            # Create new event loop if none exists
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Enable nested event loops for compatibility
        nest_asyncio.apply(loop)

        return loop.run_until_complete(self.get_cached_papers_async())

    async def close(self):
        """Close MCP session and stdio context."""
        if self._session:
            await self._session.__aexit__(None, None, None)
            self._session = None
        if self._stdio_context:
            await self._stdio_context.__aexit__(None, None, None)
            self._stdio_context = None
        logger.info("Closed MCP session")

    def __del__(self):
        """Cleanup on deletion."""
        if self._session:
            import asyncio
            import nest_asyncio
            try:
                loop = asyncio.get_event_loop()
                # Check if loop is closed
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                if loop.is_running():
                    loop.create_task(self.close())
                else:
                    nest_asyncio.apply(loop)
                    loop.run_until_complete(self.close())
            except Exception as e:
                logger.warning(f"Error closing MCP session in __del__: {str(e)}")
