"""
PDF processing and text extraction with chunking.
"""
import logging
from pathlib import Path
from typing import List, Optional
import hashlib
import tiktoken
from pypdf import PdfReader

from utils.schemas import PaperChunk, Paper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """Process PDFs and extract text with intelligent chunking."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        encoding_name: str = "cl100k_base"
    ):
        """
        Initialize PDF processor.

        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            encoding_name: Tiktoken encoding name
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)

    def extract_text(self, pdf_path: Path) -> Optional[str]:
        """
        Extract text from PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text or None if extraction fails
        """
        try:
            reader = PdfReader(str(pdf_path))
            text_parts = []

            for page_num, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text()
                    if text.strip():
                        text_parts.append(f"[Page {page_num}]\n{text}")
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num}: {str(e)}")
                    continue

            if not text_parts:
                logger.error(f"No text extracted from {pdf_path}")
                return None

            full_text = "\n\n".join(text_parts)
            logger.info(f"Extracted {len(full_text)} characters from {pdf_path.name}")
            return full_text

        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {str(e)}")
            return None

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))

    def _generate_chunk_id(self, paper_id: str, chunk_index: int) -> str:
        """Generate unique chunk ID."""
        content = f"{paper_id}_{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()

    def chunk_text(
        self,
        text: str,
        paper: Paper
    ) -> List[PaperChunk]:
        """
        Chunk text into overlapping segments.

        Args:
            text: Full text to chunk
            paper: Paper metadata

        Returns:
            List of PaperChunk objects
        """
        chunks = []
        tokens = self.encoding.encode(text)

        # Extract page information from text
        page_markers = []
        lines = text.split('\n')
        current_char = 0
        for line in lines:
            if line.startswith('[Page ') and line.endswith(']'):
                try:
                    page_num = int(line[6:-1])
                    page_markers.append((current_char, page_num))
                except ValueError:
                    pass
            current_char += len(line) + 1

        chunk_index = 0
        start_idx = 0

        while start_idx < len(tokens):
            # Calculate end index
            end_idx = min(start_idx + self.chunk_size, len(tokens))

            # Get chunk tokens and decode
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.encoding.decode(chunk_tokens)

            # Determine page number
            chunk_start_char = len(self.encoding.decode(tokens[:start_idx]))
            page_number = self._get_page_number(chunk_start_char, page_markers)

            # Extract section if possible (simple heuristic)
            section = self._extract_section(chunk_text)

            # Create chunk
            chunk = PaperChunk(
                chunk_id=self._generate_chunk_id(paper.arxiv_id, chunk_index),
                paper_id=paper.arxiv_id,
                content=chunk_text.strip(),
                section=section,
                page_number=page_number,
                arxiv_url=paper.pdf_url,
                metadata={
                    "title": paper.title,
                    "authors": paper.authors,
                    "chunk_index": chunk_index,
                    "token_count": len(chunk_tokens)
                }
            )
            chunks.append(chunk)

            # Move to next chunk with overlap
            start_idx += self.chunk_size - self.chunk_overlap
            chunk_index += 1

        logger.info(f"Created {len(chunks)} chunks for paper {paper.arxiv_id}")
        return chunks

    def _get_page_number(
        self,
        char_position: int,
        page_markers: List[tuple]
    ) -> Optional[int]:
        """Determine page number for character position."""
        if not page_markers:
            return None

        for i, (marker_pos, page_num) in enumerate(page_markers):
            if char_position < marker_pos:
                return page_markers[i - 1][1] if i > 0 else None
        return page_markers[-1][1]

    def _extract_section(self, text: str) -> Optional[str]:
        """
        Extract section name from chunk (simple heuristic).

        Looks for common section headers.
        """
        section_keywords = [
            'abstract', 'introduction', 'related work', 'methodology',
            'method', 'experiments', 'results', 'discussion',
            'conclusion', 'references', 'appendix'
        ]

        lines = text.split('\n')[:5]  # Check first 5 lines
        for line in lines:
            line_lower = line.lower().strip()
            for keyword in section_keywords:
                if keyword in line_lower and len(line.split()) < 10:
                    return line.strip()
        return None

    def process_paper(
        self,
        pdf_path: Path,
        paper: Paper
    ) -> List[PaperChunk]:
        """
        Process a paper PDF into chunks.

        Args:
            pdf_path: Path to PDF file
            paper: Paper metadata

        Returns:
            List of PaperChunk objects
        """
        # Extract text
        text = self.extract_text(pdf_path)
        if not text:
            logger.error(f"Failed to extract text from {pdf_path}")
            return []

        # Chunk text
        chunks = self.chunk_text(text, paper)
        return chunks
