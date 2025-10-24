"""
Pydantic schemas for type safety and validation.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Paper(BaseModel):
    """Schema for arXiv paper metadata."""
    arxiv_id: str = Field(..., description="arXiv paper ID")
    title: str = Field(..., description="Paper title")
    authors: List[str] = Field(..., description="List of author names")
    abstract: str = Field(..., description="Paper abstract")
    pdf_url: str = Field(..., description="URL to PDF")
    published: datetime = Field(..., description="Publication date")
    categories: List[str] = Field(default_factory=list, description="arXiv categories")

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


class ConsensusPoint(BaseModel):
    """Schema for consensus findings across papers."""
    statement: str = Field(..., description="Consensus statement")
    supporting_papers: List[str] = Field(..., description="Paper IDs supporting this claim")
    citations: List[str] = Field(..., description="Specific citations")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in consensus")


class Contradiction(BaseModel):
    """Schema for contradictory findings."""
    topic: str = Field(..., description="Topic of contradiction")
    viewpoint_a: str = Field(..., description="First viewpoint")
    papers_a: List[str] = Field(..., description="Papers supporting viewpoint A")
    viewpoint_b: str = Field(..., description="Second viewpoint")
    papers_b: List[str] = Field(..., description="Papers supporting viewpoint B")
    citations: List[str] = Field(..., description="Specific citations for both sides")


class SynthesisResult(BaseModel):
    """Schema for synthesis across multiple papers."""
    consensus_points: List[ConsensusPoint] = Field(..., description="Areas of agreement")
    contradictions: List[Contradiction] = Field(..., description="Areas of disagreement")
    research_gaps: List[str] = Field(..., description="Identified research gaps")
    summary: str = Field(..., description="Executive summary")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    papers_analyzed: List[str] = Field(..., description="List of paper IDs analyzed")


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
    cost_estimate: float = Field(..., description="Estimated cost in USD")
    processing_time: float = Field(..., description="Processing time in seconds")


class AgentState(BaseModel):
    """Schema for LangGraph state management."""
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
