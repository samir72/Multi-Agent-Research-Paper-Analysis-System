"""
Pydantic schemas for type safety and validation.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator, field_validator
import logging

logger = logging.getLogger(__name__)


class Paper(BaseModel):
    """Schema for arXiv paper metadata."""
    arxiv_id: str = Field(..., description="arXiv paper ID")
    title: str = Field(..., description="Paper title")
    authors: List[str] = Field(..., description="List of author names")
    abstract: str = Field(..., description="Paper abstract")
    pdf_url: str = Field(..., description="URL to PDF")
    published: datetime = Field(..., description="Publication date")
    categories: List[str] = Field(default_factory=list, description="arXiv categories")

    @validator('authors', pre=True)
    def normalize_authors(cls, v):
        """Ensure authors is always a List[str], handling various input formats."""
        if isinstance(v, list):
            # Already a list, ensure all elements are strings
            return [str(author) if not isinstance(author, str) else author for author in v]
        elif isinstance(v, dict):
            # Dict format - extract values or keys as list
            logger.warning(f"Authors field is dict, extracting values: {v}")
            if 'names' in v:
                return v['names'] if isinstance(v['names'], list) else [str(v['names'])]
            elif 'authors' in v:
                return v['authors'] if isinstance(v['authors'], list) else [str(v['authors'])]
            else:
                # Extract all values from dict
                return [str(val) for val in v.values() if val]
        elif isinstance(v, str):
            # Single author as string
            return [v]
        else:
            logger.warning(f"Unexpected authors format: {type(v)}, returning empty list")
            return []

    @validator('categories', pre=True)
    def normalize_categories(cls, v):
        """Ensure categories is always a List[str], handling various input formats."""
        if isinstance(v, list):
            # Already a list, ensure all elements are strings
            return [str(cat) if not isinstance(cat, str) else cat for cat in v]
        elif isinstance(v, dict):
            # Dict format - extract values or keys as list
            logger.warning(f"Categories field is dict, extracting values: {v}")
            if 'categories' in v:
                return v['categories'] if isinstance(v['categories'], list) else [str(v['categories'])]
            else:
                # Extract all values from dict
                return [str(val) for val in v.values() if val]
        elif isinstance(v, str):
            # Single category as string
            return [v]
        else:
            logger.warning(f"Unexpected categories format: {type(v)}, returning empty list")
            return []

    @validator('pdf_url', pre=True)
    def normalize_pdf_url(cls, v):
        """Ensure pdf_url is always a string."""
        if isinstance(v, dict):
            logger.warning(f"pdf_url is dict, extracting url value: {v}")
            return v.get('url') or v.get('pdf_url') or str(v)
        return str(v) if v else ""

    @validator('title', pre=True)
    def normalize_title(cls, v):
        """Ensure title is always a string."""
        if isinstance(v, dict):
            logger.warning(f"title is dict, extracting title value: {v}")
            return v.get('title') or str(v)
        return str(v) if v else ""

    @validator('abstract', pre=True)
    def normalize_abstract(cls, v):
        """Ensure abstract is always a string."""
        if isinstance(v, dict):
            logger.warning(f"abstract is dict, extracting abstract value: {v}")
            return v.get('abstract') or v.get('summary') or str(v)
        return str(v) if v else ""

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PaperChunk(BaseModel):
    """Schema for chunked paper content."""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    paper_id: str = Field(..., description="arXiv paper ID")
    content: str = Field(..., description="Chunk text content")
    section: Optional[str] = Field(None, description="Section name if available")
    page_number: Optional[int] = Field(None, description="Page number")
    arxiv_url: str = Field(..., description="arXiv URL for citation")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class Analysis(BaseModel):
    """Schema for individual paper analysis."""
    paper_id: str = Field(..., description="arXiv paper ID")
    methodology: str = Field(..., description="Research methodology description")
    key_findings: List[str] = Field(..., description="Main findings from the paper")
    conclusions: str = Field(..., description="Paper conclusions")
    limitations: List[str] = Field(..., description="Study limitations")
    citations: List[str] = Field(..., description="Source locations for claims")
    main_contributions: List[str] = Field(default_factory=list, description="Key contributions")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Analysis confidence")

    @field_validator('key_findings', 'limitations', 'citations', 'main_contributions', mode='before')
    @classmethod
    def normalize_string_lists(cls, v, info):
        """
        Normalize list fields to ensure they contain only strings.
        Handles nested lists, None values, and mixed types.
        """
        def flatten_and_clean(value):
            """Recursively flatten nested lists and clean values."""
            if isinstance(value, str):
                return [value.strip()] if value.strip() else []
            elif isinstance(value, list):
                cleaned = []
                for item in value:
                    if isinstance(item, str):
                        if item.strip():
                            cleaned.append(item.strip())
                    elif isinstance(item, list):
                        # Recursively flatten nested lists
                        cleaned.extend(flatten_and_clean(item))
                    elif item is not None and str(item).strip():
                        cleaned.append(str(item).strip())
                return cleaned
            elif value is not None:
                str_value = str(value).strip()
                return [str_value] if str_value else []
            else:
                return []

        result = flatten_and_clean(v)
        if v != result:
            logger.warning(f"Normalized '{info.field_name}' in Analysis: cleaned nested/invalid values")
        return result


class ConsensusPoint(BaseModel):
    """Schema for consensus findings across papers."""
    statement: str = Field(..., description="Consensus statement")
    supporting_papers: List[str] = Field(..., description="Paper IDs supporting this claim")
    citations: List[str] = Field(..., description="Specific citations")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in consensus")

    @field_validator('supporting_papers', 'citations', mode='before')
    @classmethod
    def normalize_string_lists(cls, v, info):
        """Normalize list fields to ensure they contain only strings."""
        def flatten_and_clean(value):
            if isinstance(value, str):
                return [value.strip()] if value.strip() else []
            elif isinstance(value, list):
                cleaned = []
                for item in value:
                    if isinstance(item, str) and item.strip():
                        cleaned.append(item.strip())
                    elif isinstance(item, list):
                        cleaned.extend(flatten_and_clean(item))
                    elif item is not None and str(item).strip():
                        cleaned.append(str(item).strip())
                return cleaned
            elif value is not None:
                str_value = str(value).strip()
                return [str_value] if str_value else []
            else:
                return []

        result = flatten_and_clean(v)
        if v != result:
            logger.warning(f"Normalized '{info.field_name}' in ConsensusPoint: cleaned nested/invalid values")
        return result


class Contradiction(BaseModel):
    """Schema for contradictory findings."""
    topic: str = Field(..., description="Topic of contradiction")
    viewpoint_a: str = Field(..., description="First viewpoint")
    papers_a: List[str] = Field(..., description="Papers supporting viewpoint A")
    viewpoint_b: str = Field(..., description="Second viewpoint")
    papers_b: List[str] = Field(..., description="Papers supporting viewpoint B")
    citations: List[str] = Field(..., description="Specific citations for both sides")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in contradiction")

    @field_validator('papers_a', 'papers_b', 'citations', mode='before')
    @classmethod
    def normalize_string_lists(cls, v, info):
        """Normalize list fields to ensure they contain only strings."""
        def flatten_and_clean(value):
            if isinstance(value, str):
                return [value.strip()] if value.strip() else []
            elif isinstance(value, list):
                cleaned = []
                for item in value:
                    if isinstance(item, str) and item.strip():
                        cleaned.append(item.strip())
                    elif isinstance(item, list):
                        cleaned.extend(flatten_and_clean(item))
                    elif item is not None and str(item).strip():
                        cleaned.append(str(item).strip())
                return cleaned
            elif value is not None:
                str_value = str(value).strip()
                return [str_value] if str_value else []
            else:
                return []

        result = flatten_and_clean(v)
        if v != result:
            logger.warning(f"Normalized '{info.field_name}' in Contradiction: cleaned nested/invalid values")
        return result


class SynthesisResult(BaseModel):
    """Schema for synthesis across multiple papers."""
    consensus_points: List[ConsensusPoint] = Field(..., description="Areas of agreement")
    contradictions: List[Contradiction] = Field(..., description="Areas of disagreement")
    research_gaps: List[str] = Field(..., description="Identified research gaps")
    summary: str = Field(..., description="Executive summary")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    papers_analyzed: List[str] = Field(..., description="List of paper IDs analyzed")

    @field_validator('research_gaps', 'papers_analyzed', mode='before')
    @classmethod
    def normalize_string_lists(cls, v, info):
        """Normalize list fields to ensure they contain only strings."""
        def flatten_and_clean(value):
            if isinstance(value, str):
                return [value.strip()] if value.strip() else []
            elif isinstance(value, list):
                cleaned = []
                for item in value:
                    if isinstance(item, str) and item.strip():
                        cleaned.append(item.strip())
                    elif isinstance(item, list):
                        cleaned.extend(flatten_and_clean(item))
                    elif item is not None and str(item).strip():
                        cleaned.append(str(item).strip())
                return cleaned
            elif value is not None:
                str_value = str(value).strip()
                return [str_value] if str_value else []
            else:
                return []

        result = flatten_and_clean(v)
        if v != result:
            logger.warning(f"Normalized '{info.field_name}' in SynthesisResult: cleaned nested/invalid values")
        return result


class Citation(BaseModel):
    """Schema for properly formatted citations."""
    paper_id: str = Field(..., description="arXiv paper ID")
    authors: List[str] = Field(..., description="Paper authors")
    year: int = Field(..., description="Publication year")
    title: str = Field(..., description="Paper title")
    source: str = Field(..., description="Publication source (arXiv)")
    apa_format: str = Field(..., description="Full APA formatted citation")
    url: str = Field(..., description="arXiv URL")


class ValidatedOutput(BaseModel):
    """Schema for final validated output with citations."""
    synthesis: SynthesisResult = Field(..., description="Synthesis results")
    citations: List[Citation] = Field(..., description="All citations used")
    retrieved_chunks: List[str] = Field(..., description="Chunk IDs used for grounding")
    token_usage: Dict[str, int] = Field(default_factory=dict, description="Token usage stats")
    model_desc: Dict[str, str] = Field(default_factory=dict, description="Model descriptions")
    cost_estimate: float = Field(..., description="Estimated cost in USD")
    processing_time: float = Field(..., description="Processing time in seconds")


class AgentState(BaseModel):
    """
    Schema for LangGraph state management.

    Note: This Pydantic model serves as type documentation and validation reference.
    The actual LangGraph workflow in app.py uses Dict[str, Any] for state to maintain
    compatibility with Gradio progress tracking and dynamic state updates during execution.

    All fields in this schema correspond to keys in the workflow state dictionary.
    """
    query: str = Field(..., description="User research question")
    category: Optional[str] = Field(None, description="arXiv category filter")
    num_papers: int = Field(default=5, ge=1, le=20, description="Number of papers to retrieve")
    papers: List[Paper] = Field(default_factory=list, description="Retrieved papers")
    chunks: List[PaperChunk] = Field(default_factory=list, description="Chunked content")
    analyses: List[Analysis] = Field(default_factory=list, description="Individual analyses")
    synthesis: Optional[SynthesisResult] = Field(None, description="Synthesis result")
    validated_output: Optional[ValidatedOutput] = Field(None, description="Final output")
    errors: List[str] = Field(default_factory=list, description="Error messages")

    class Config:
        arbitrary_types_allowed = True
