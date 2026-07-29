"""
arXiv API client wrapper with error handling and caching.
"""
import os
import logging
from typing import List, Optional
from pathlib import Path
import arxiv
from tenacity import retry, stop_after_attempt, wait_exponential

from utils.schemas import Paper

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


class ArxivClient:
    """Wrapper for arXiv API with error handling and caching."""

    def __init__(self, cache_dir: str = "data/papers"):
        """
        Initialize arXiv client.

        Args:
            cache_dir: Directory to cache downloaded papers
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.client = arxiv.Client()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def search_papers(
        self,
        query: str,
        max_results: int = 5,
        category: Optional[str] = None,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance
    ) -> List[Paper]:
        """
        Search for papers on arXiv.

        Args:
            query: Search query
            max_results: Maximum number of papers to return
            category: Optional arXiv category filter (e.g., 'cs.AI')
            sort_by: Sort criterion

        Returns:
            List of Paper objects

        Raises:
            Exception: If arXiv API fails after retries
        """
        try:
            # Build search query
            search_query = query
            if category:
                search_query = f"{query} AND cat:{category}"

            logger.info(f"Searching arXiv for: {search_query}")

            # Create search
            search = arxiv.Search(
                query=search_query,
                max_results=max_results,
                sort_by=sort_by
            )

            # Fetch results (Search.results() is deprecated/removed in newer
            # arxiv releases in favor of Client.results())
            papers = []
            for result in self.client.results(search):
                paper = Paper(
                    arxiv_id=result.entry_id.split('/')[-1],
                    title=result.title,
                    authors=[author.name for author in result.authors],
                    abstract=result.summary,
                    pdf_url=_extract_pdf_url(result),
                    published=result.published,
                    categories=result.categories
                )
                papers.append(paper)

            logger.info(f"Found {len(papers)} papers")
            return papers

        except Exception as e:
            logger.error(f"Error searching arXiv: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def download_paper(self, paper: Paper) -> Optional[Path]:
        """
        Download paper PDF if not already cached.

        Args:
            paper: Paper object

        Returns:
            Path to downloaded PDF, or None if download fails
        """
        try:
            # Check if already cached
            pdf_path = self.cache_dir / f"{paper.arxiv_id}.pdf"
            if pdf_path.exists():
                logger.info(f"Paper {paper.arxiv_id} already cached")
                return pdf_path

            logger.info(f"Downloading paper {paper.arxiv_id}")

            # Download using arxiv library
            search = arxiv.Search(id_list=[paper.arxiv_id])
            result = next(self.client.results(search))
            result.download_pdf(dirpath=str(self.cache_dir), filename=f"{paper.arxiv_id}.pdf")

            logger.info(f"Downloaded paper to {pdf_path}")
            return pdf_path

        except Exception as e:
            logger.error(f"Error downloading paper {paper.arxiv_id}: {str(e)}")
            return None

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

    def get_cached_papers(self) -> List[Path]:
        """
        Get list of cached paper PDFs.

        Returns:
            List of Paths to cached PDFs
        """
        return list(self.cache_dir.glob("*.pdf"))
