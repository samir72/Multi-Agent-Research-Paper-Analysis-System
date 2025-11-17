"""
Main Gradio application with LangGraph agent orchestration.
"""
# Fix MCP dependency conflict on Hugging Face Spaces startup
# This must run before any other imports that depend on mcp
import subprocess
import sys
import os

# Only run the fix if we detect we're in a fresh environment
if os.getenv("SPACE_ID"):  # Running on Hugging Face Spaces
    try:
        print("🔧 Fixing MCP dependency conflict for Hugging Face Spaces...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", "mcp==1.17.0"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("✅ MCP dependency fixed!")
    except Exception as e:
        print(f"⚠️  Warning: Could not fix MCP dependency: {e}")
        print("   App may still work if dependencies are correctly installed")

import time
import logging
import copy
from typing import Dict, Any, Tuple
from pathlib import Path
from dotenv import load_dotenv
import gradio as gr
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Validate required environment variables
def validate_environment():
    """Validate that all required environment variables are set."""
    required_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_DEPLOYMENT_NAME",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"
    ]

    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if not value or value.strip() == "":
            missing_vars.append(var)

    if missing_vars:
        error_msg = (
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            f"Please set them in your .env file or HuggingFace Spaces secrets.\n"
            f"See .env.example for reference."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Log configuration (masked)
    logger.info(f"Azure OpenAI Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
    logger.info(f"LLM Deployment: {os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')}")
    logger.info(f"Embedding Deployment: {os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME')}")
    logger.info(f"API Version: {os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-01')}")

# Validate environment before importing other modules
validate_environment()

# Import utilities
from utils.arxiv_client import ArxivClient
from utils.pdf_processor import PDFProcessor
from utils.cache import SemanticCache

# Import MCP clients if available
try:
    from utils.mcp_arxiv_client import MCPArxivClient
    LEGACY_MCP_AVAILABLE = True
except ImportError:
    LEGACY_MCP_AVAILABLE = False
    logger.warning("Legacy MCP client not available")

try:
    from utils.fastmcp_arxiv_client import FastMCPArxivClient
    from utils.fastmcp_arxiv_server import get_server, shutdown_server
    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False
    logger.warning("FastMCP not available - install with: pip install fastmcp")

# Import RAG components
from rag.embeddings import EmbeddingGenerator
from rag.vector_store import VectorStore
from rag.retrieval import RAGRetriever

# Import agents
from agents.retriever import RetrieverAgent
from agents.analyzer import AnalyzerAgent
from agents.synthesis import SynthesisAgent
from agents.citation import CitationAgent

# Import LangGraph orchestration
from orchestration.workflow_graph import create_workflow_graph, run_workflow
from utils.langgraph_state import create_initial_state

# Import LangFuse observability
from utils.langfuse_client import initialize_langfuse, instrument_openai, flush_langfuse, shutdown_langfuse



class ResearchPaperAnalyzer:
    """Main application class for research paper analysis."""

    def __init__(self):
        """Initialize the analyzer with all components."""
        logger.info("Initializing Research Paper Analyzer...")

        # Initialize LangFuse observability
        initialize_langfuse()
        instrument_openai()  # Auto-trace all OpenAI calls
        logger.info("LangFuse observability initialized")

        # Configuration
        storage_path = os.getenv("MCP_ARXIV_STORAGE_PATH", "data/mcp_papers")
        server_port = int(os.getenv("FASTMCP_SERVER_PORT", "5555"))
        use_mcp = os.getenv("USE_MCP_ARXIV", "false").lower() == "true"
        use_legacy_mcp = os.getenv("USE_LEGACY_MCP", "false").lower() == "true"

        # Initialize arXiv clients with intelligent selection
        self.fastmcp_server = None
        primary_client = None
        fallback_client = None

        if use_mcp:
            if use_legacy_mcp and LEGACY_MCP_AVAILABLE:
                # Use legacy MCP as primary
                logger.info("Using legacy MCP arXiv client (USE_LEGACY_MCP=true)")
                primary_client = MCPArxivClient(storage_path=storage_path)
                fallback_client = ArxivClient()  # Direct API as fallback
            elif FASTMCP_AVAILABLE:
                # Use FastMCP as primary (default MCP mode)
                logger.info("Using FastMCP arXiv client (default MCP mode)")

                # Start FastMCP server with auto-start
                try:
                    self.fastmcp_server = get_server(
                        storage_path=storage_path,
                        server_port=server_port,
                        auto_start=True
                    )
                    logger.info(f"FastMCP server started on port {server_port}")

                    # Create FastMCP client
                    primary_client = FastMCPArxivClient(
                        storage_path=storage_path,
                        server_host="localhost",
                        server_port=server_port
                    )
                    fallback_client = ArxivClient()  # Direct API as fallback

                except Exception as e:
                    logger.error(f"Failed to start FastMCP: {str(e)}")
                    logger.warning("Falling back to legacy MCP or direct API")

                    if LEGACY_MCP_AVAILABLE:
                        logger.info("Using legacy MCP as fallback")
                        primary_client = MCPArxivClient(storage_path=storage_path)
                    else:
                        logger.info("Using direct arXiv API")
                        primary_client = ArxivClient()
                    fallback_client = None
            elif LEGACY_MCP_AVAILABLE:
                # FastMCP not available, use legacy MCP
                logger.warning("FastMCP not available, using legacy MCP")
                primary_client = MCPArxivClient(storage_path=storage_path)
                fallback_client = ArxivClient()
            else:
                # No MCP available
                logger.warning("MCP requested but not available - using direct arXiv API")
                primary_client = ArxivClient()
                fallback_client = None
        else:
            # Direct API mode (default)
            logger.info("Using direct arXiv API client (USE_MCP_ARXIV=false)")
            primary_client = ArxivClient()
            fallback_client = None

        # Store primary client for reference
        self.arxiv_client = primary_client

        # Initialize other components
        self.pdf_processor = PDFProcessor()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore()
        self.rag_retriever = RAGRetriever(
            vector_store=self.vector_store,
            embedding_generator=self.embedding_generator
        )
        self.cache = SemanticCache()

        # Initialize agents with fallback support
        self.retriever_agent = RetrieverAgent(
            arxiv_client=primary_client,
            pdf_processor=self.pdf_processor,
            vector_store=self.vector_store,
            embedding_generator=self.embedding_generator,
            fallback_client=fallback_client  # Enable fallback
        )
        self.analyzer_agent = AnalyzerAgent(rag_retriever=self.rag_retriever)
        self.synthesis_agent = SynthesisAgent(rag_retriever=self.rag_retriever)
        self.citation_agent = CitationAgent(rag_retriever=self.rag_retriever)

        # Create LangGraph workflow
        self.workflow_app = create_workflow_graph(
            retriever_agent=self.retriever_agent,
            analyzer_agent=self.analyzer_agent,
            synthesis_agent=self.synthesis_agent,
            citation_agent=self.citation_agent,
            use_checkpointing=True,
        )
        logger.info("LangGraph workflow created with checkpointing")

        logger.info("Initialization complete")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            # Flush and shutdown LangFuse
            logger.info("Shutting down LangFuse observability")
            shutdown_langfuse()

            # Shutdown FastMCP server if running
            if self.fastmcp_server:
                logger.info("Shutting down FastMCP server")
                shutdown_server()
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")

    def _create_empty_outputs(self) -> Tuple[pd.DataFrame, str, str, str, str]:
        """Create empty outputs for initial state."""
        empty_df = pd.DataFrame({"Status": ["⏳ Initializing..."]})
        empty_html = "<p>Processing...</p>"
        return empty_df, empty_html, empty_html, empty_html, empty_html

    def _format_papers_partial(
        self,
        papers: list,
        analyses: list,
        completed_count: int
    ) -> pd.DataFrame:
        """Format papers table with partial analysis results."""
        papers_data = []
        for i, paper in enumerate(papers):
            if i < completed_count and i < len(analyses):
                # Analysis completed
                analysis = analyses[i]
                if analysis.confidence_score == 0.0:
                    status = "⚠️ Failed"
                else:
                    status = "✅ Complete"
                confidence = f"{analysis.confidence_score:.1%}"
            elif i < completed_count:
                # Analysis in progress (submitted but not yet in analyses list)
                status = "⏳ Analyzing"
                confidence = "-"
            else:
                # Not started
                status = "⏸️ Pending"
                confidence = "-"

            papers_data.append({
                "Title": paper.title,
                "Authors": ", ".join(paper.authors[:3]) + ("..." if len(paper.authors) > 3 else ""),
                "Date": paper.published.strftime("%Y-%m-%d"),
                "arXiv ID": paper.arxiv_id,
                "Status": status,
                "Confidence": confidence,
                "Link": f"[View PDF]({paper.pdf_url})"
            })
        return pd.DataFrame(papers_data)

    def _format_analysis_partial(self, papers: list, analyses: list) -> str:
        """Format analysis HTML with partial results."""
        if not analyses:
            return "<h2>Paper Analyses</h2><p>Analyzing papers...</p>"

        analysis_html = "<h2>Paper Analyses</h2>"
        analysis_html += f"<p><em>Analyzed {len(analyses)}/{len(papers)} papers</em></p>"

        for paper, analysis in zip(papers[:len(analyses)], analyses):
            # Skip failed analyses
            if analysis.confidence_score == 0.0:
                continue

            analysis_html += f"""
            <details style="margin-bottom: 20px; border: 1px solid #ddd; padding: 10px; border-radius: 5px;">
                <summary style="cursor: pointer; font-weight: bold; font-size: 1.1em;">
                    {paper.title}
                </summary>
                <div style="margin-top: 10px;">
                    <p><strong>Confidence:</strong> {analysis.confidence_score:.2%}</p>
                    <h4>Methodology</h4>
                    <p>{analysis.methodology}</p>
                    <h4>Key Findings</h4>
                    <ul>
                        {"".join(f"<li>{f}</li>" for f in analysis.key_findings)}
                    </ul>
                    <h4>Main Contributions</h4>
                    <ul>
                        {"".join(f"<li>{c}</li>" for c in analysis.main_contributions)}
                    </ul>
                    <h4>Conclusions</h4>
                    <p>{analysis.conclusions}</p>
                    <h4>Limitations</h4>
                    <ul>
                        {"".join(f"<li>{l}</li>" for l in analysis.limitations)}
                    </ul>
                </div>
            </details>
            """
        return analysis_html

    def _format_synthesis_output(self, papers: list, validated_output) -> str:
        """Format synthesis section HTML."""
        synthesis = validated_output.synthesis
        synthesis_html = f"""
        <div style="background-color: #f0f8ff; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h2>Executive Summary</h2>
            <p><strong>Confidence Score:</strong> {synthesis.confidence_score:.2%}</p>
            <p style="font-size: 1.1em; line-height: 1.6;">{synthesis.summary}</p>
        </div>

        <div style="margin-bottom: 30px;">
            <h3 style="color: #2e7d32;">Consensus Findings</h3>
            {"".join(f'''
            <div style="background-color: #e8f5e9; padding: 15px; margin-bottom: 10px; border-radius: 5px; border-left: 4px solid #4caf50;">
                <p style="font-weight: bold;">{cp.statement}</p>
                <p><strong>Supporting Papers:</strong>{self._format_paper_references(cp.supporting_papers, papers)}</p>
                <p><strong>Confidence:</strong> {cp.confidence:.2%}</p>
            </div>
            ''' for cp in synthesis.consensus_points)}
        </div>

        <div style="margin-bottom: 30px;">
            <h3 style="color: #f57c00;">Contradictions</h3>
            {"".join(f'''
            <div style="background-color: #fff8e1; padding: 15px; margin-bottom: 10px; border-radius: 5px; border-left: 4px solid #ffa726;">
                <p style="font-weight: bold;">Topic: {c.topic}</p>
                <p><strong>Confidence:</strong> {c.confidence:.2%}</p>
                <p><strong>Viewpoint A:</strong> {c.viewpoint_a}</p>
                <p style="margin-left: 20px; color: #555; margin-top: 5px;"><em>Papers:</em>{self._format_paper_references(c.papers_a, papers)}</p>
                <p style="margin-top: 10px;"><strong>Viewpoint B:</strong> {c.viewpoint_b}</p>
                <p style="margin-left: 20px; color: #555; margin-top: 5px;"><em>Papers:</em>{self._format_paper_references(c.papers_b, papers)}</p>
            </div>
            ''' for c in synthesis.contradictions)}
        </div>

        <div>
            <h3 style="color: #1976d2;">Research Gaps</h3>
            <ul>
                {"".join(f"<li style='margin-bottom: 8px;'>{gap}</li>" for gap in synthesis.research_gaps)}
            </ul>
        </div>
        """
        return synthesis_html

    def run_workflow(
        self,
        query: str,
        category: str,
        num_papers: int,
        progress=gr.Progress()
    ):
        """
        Execute the complete research paper analysis workflow using LangGraph.

        This is a generator function that yields progressive UI updates as the workflow executes.

        Args:
            query: Research question
            category: arXiv category
            num_papers: Number of papers to analyze
            progress: Gradio progress tracker

        Yields:
            Tuple of (papers_df, analysis_html, synthesis_html, citations_html, stats)
            after each significant workflow update
        """
        try:
            start_time = time.time()

            # Yield initial empty state
            yield self._create_empty_outputs()

            # Check cache first
            progress(0.0, desc="Checking cache...")
            query_embedding = self.embedding_generator.generate_embedding(query)
            cached_result = self.cache.get(query, query_embedding, category)

            if cached_result:
                logger.info("Using cached result")
                # Make a deep copy to avoid mutating the cache
                cached_result = copy.deepcopy(cached_result)

                # Convert dicts back to Pydantic models
                from utils.schemas import Paper, Analysis, ValidatedOutput
                cached_result["papers"] = [Paper(**p) for p in cached_result["papers"]]
                cached_result["analyses"] = [Analysis(**a) for a in cached_result["analyses"]]
                cached_result["validated_output"] = ValidatedOutput(**cached_result["validated_output"])
                yield self._format_output(cached_result)
                return

            # Create initial state using LangGraph state schema
            import uuid
            session_id = f"session-{uuid.uuid4().hex[:8]}"

            initial_state = create_initial_state(
                query=query,
                category=category if category != "All" else None,
                num_papers=num_papers,
                model_desc={
                    "llm_model": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini"),
                    "embedding_model": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-small")
                },
                start_time=start_time,
                session_id=session_id,
            )
            # Note: Progress object is NOT added to state to avoid msgpack serialization issues

            logger.info(f"Starting LangGraph workflow execution (session: {session_id})")

            # Execute LangGraph workflow (non-streaming for simplicity)
            # The workflow internally handles progress updates via the progress callback
            progress(0.1, desc="Executing workflow...")

            # Execute LangGraph workflow
            final_state = run_workflow(
                app=self.workflow_app,
                initial_state=initial_state,
                thread_id=session_id,
                use_streaming=False,  # Set to True for streaming in future
            )

            logger.info("LangGraph workflow execution complete")

            # Flush LangFuse traces
            flush_langfuse()

            # Check workflow results
            if not final_state.get("papers"):
                logger.warning("No papers found, terminating workflow")
                progress(1.0, desc="No papers found")
                yield self._format_error(final_state.get("errors", ["No papers found"]))
                return

            # Check for validated output
            if not final_state.get("validated_output"):
                logger.warning("Workflow completed but no validated output")
                yield self._format_error(final_state.get("errors", ["Unknown error occurred"]))
                return

            # Processing time is now calculated in finalize_node
            progress(1.0, desc="Complete!")

            # Cache the result
            cache_data = {
                "papers": [p.model_dump(mode='json') for p in final_state["papers"]],
                "analyses": [a.model_dump(mode='json') for a in final_state["analyses"]],
                "validated_output": final_state["validated_output"].model_dump(mode='json')
            }
            self.cache.set(query, query_embedding, cache_data, category)

            # Format final output
            result = {
                "papers": final_state["papers"],
                "analyses": final_state["analyses"],
                "validated_output": final_state["validated_output"]
            }
            yield self._format_output(result)

        except Exception as e:
            logger.error(f"Workflow error: {str(e)}")
            yield self._format_error([str(e)])

    def _format_paper_references(self, paper_ids: list, papers: list) -> str:
        """
        Format paper references with title and arXiv ID.

        Args:
            paper_ids: List of arXiv IDs
            papers: List of Paper objects

        Returns:
            Formatted HTML string with paper titles and IDs
        """
        # Create a lookup dictionary
        paper_map = {p.arxiv_id: p for p in papers}

        formatted_refs = []
        for paper_id in paper_ids:
            paper = paper_map.get(paper_id)
            if paper:
                # Truncate long titles
                title = paper.title if len(paper.title) <= 60 else paper.title[:57] + "..."
                formatted_refs.append(f"{title} ({paper_id})")
            else:
                # Fallback if paper not found
                formatted_refs.append(paper_id)

        return "<br>• " + "<br>• ".join(formatted_refs) if formatted_refs else ""

    def _format_output(
        self,
        result: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, str, str, str, str]:
        """Format the workflow output for Gradio."""
        papers = result["papers"]
        analyses = result["analyses"]
        validated_output = result["validated_output"]

        # Format papers table
        papers_data = []
        for paper, analysis in zip(papers, analyses):
            # Determine status based on confidence
            if analysis.confidence_score == 0.0:
                status = "⚠️ Failed"
            else:
                status = "✅ Complete"

            papers_data.append({
                "Title": paper.title,
                "Authors": ", ".join(paper.authors[:3]) + ("..." if len(paper.authors) > 3 else ""),
                "Date": paper.published.strftime("%Y-%m-%d"),
                "arXiv ID": paper.arxiv_id,
                "Status": status,
                "Confidence": f"{analysis.confidence_score:.1%}",
                "Link": f"[View PDF]({paper.pdf_url})"  # Markdown link format
            })
        papers_df = pd.DataFrame(papers_data)

        # Format analysis - only show successful analyses (confidence > 0%)
        analysis_html = "<h2>Paper Analyses</h2>"
        successful_count = sum(1 for a in analyses if a.confidence_score > 0.0)
        failed_count = len(analyses) - successful_count

        if failed_count > 0:
            analysis_html += f"""
            <div style="background-color: #fff3cd; padding: 10px; margin-bottom: 20px; border-radius: 5px; border-left: 4px solid #ffc107;">
                <p><strong>Note:</strong> {failed_count} paper(s) failed analysis and are excluded from this view.
                Check the Papers tab for complete status information.</p>
            </div>
            """

        for paper, analysis in zip(papers, analyses):
            # Only show successful analyses
            if analysis.confidence_score == 0.0:
                continue

            analysis_html += f"""
            <details style="margin-bottom: 20px; border: 1px solid #ddd; padding: 10px; border-radius: 5px;">
                <summary style="cursor: pointer; font-weight: bold; font-size: 1.1em;">
                    {paper.title}
                </summary>
                <div style="margin-top: 10px;">
                    <p><strong>Confidence:</strong> {analysis.confidence_score:.2%}</p>
                    <h4>Methodology</h4>
                    <p>{analysis.methodology}</p>
                    <h4>Key Findings</h4>
                    <ul>
                        {"".join(f"<li>{f}</li>" for f in analysis.key_findings)}
                    </ul>
                    <h4>Main Contributions</h4>
                    <ul>
                        {"".join(f"<li>{c}</li>" for c in analysis.main_contributions)}
                    </ul>
                    <h4>Conclusions</h4>
                    <p>{analysis.conclusions}</p>
                    <h4>Limitations</h4>
                    <ul>
                        {"".join(f"<li>{l}</li>" for l in analysis.limitations)}
                    </ul>
                </div>
            </details>
            """

        # Format synthesis
        synthesis = validated_output.synthesis
        synthesis_html = f"""
        <div style="background-color: #f0f8ff; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h2>Executive Summary</h2>
            <p><strong>Confidence Score:</strong> {synthesis.confidence_score:.2%}</p>
            <p style="font-size: 1.1em; line-height: 1.6;">{synthesis.summary}</p>
        </div>

        <div style="margin-bottom: 30px;">
            <h3 style="color: #2e7d32;">Consensus Findings</h3>
            {"".join(f'''
            <div style="background-color: #e8f5e9; padding: 15px; margin-bottom: 10px; border-radius: 5px; border-left: 4px solid #4caf50;">
                <p style="font-weight: bold;">{cp.statement}</p>
                <p><strong>Supporting Papers:</strong>{self._format_paper_references(cp.supporting_papers, papers)}</p>
                <p><strong>Confidence:</strong> {cp.confidence:.2%}</p>
            </div>
            ''' for cp in synthesis.consensus_points)}
        </div>

        <div style="margin-bottom: 30px;">
            <h3 style="color: #f57c00;">Contradictions</h3>
            {"".join(f'''
            <div style="background-color: #fff8e1; padding: 15px; margin-bottom: 10px; border-radius: 5px; border-left: 4px solid #ffa726;">
                <p style="font-weight: bold;">Topic: {c.topic}</p>
                <p><strong>Confidence:</strong> {c.confidence:.2%}</p>
                <p><strong>Viewpoint A:</strong> {c.viewpoint_a}</p>
                <p style="margin-left: 20px; color: #555; margin-top: 5px;"><em>Papers:</em>{self._format_paper_references(c.papers_a, papers)}</p>
                <p style="margin-top: 10px;"><strong>Viewpoint B:</strong> {c.viewpoint_b}</p>
                <p style="margin-left: 20px; color: #555; margin-top: 5px;"><em>Papers:</em>{self._format_paper_references(c.papers_b, papers)}</p>
            </div>
            ''' for c in synthesis.contradictions)}
        </div>

        <div>
            <h3 style="color: #1976d2;">Research Gaps</h3>
            <ul>
                {"".join(f"<li style='margin-bottom: 8px;'>{gap}</li>" for gap in synthesis.research_gaps)}
            </ul>
        </div>
        """

        # Format citations
        citations_html = "<h2>References (APA Style)</h2><ol>"
        for citation in validated_output.citations:
            citations_html += f"""
            <li style="margin-bottom: 15px;">
                {citation.apa_format}
            </li>
            """
        citations_html += "</ol>"

        # Format stats
        stats = f"""
        <h3>Processing Statistics</h3>
        <ul>
            <li>Papers Analyzed: {len(validated_output.synthesis.papers_analyzed)}</li>
            <li>Processing Time: {validated_output.processing_time:.1f} seconds</li>
            <li>Estimated Cost: ${validated_output.cost_estimate:.4f}</li>
            <li>Chunks Used: {len(validated_output.retrieved_chunks)}</li>
            <li>Token Usage:</li>
            <ul>
                <li>Input: {validated_output.token_usage.get('input_tokens', 0):,}</li>
                <li>Output: {validated_output.token_usage.get('output_tokens', 0):,}</li>
                <li>Embeddings: {validated_output.token_usage.get('embedding_tokens', 0):,}</li>
            </ul>
        </ul>
        """

        return papers_df, analysis_html, synthesis_html, citations_html, stats

    def _format_error(self, errors: list) -> Tuple[pd.DataFrame, str, str, str, str]:
        """Format error message with graceful display on Papers tab."""
        error_text = " ".join(errors)

        if "No papers found" in error_text:
            # Create a friendly message DataFrame for Papers tab
            message_df = pd.DataFrame({
                "Status": ["🔍 No Papers Found"],
                "Message": ["We couldn't find any papers matching your search query."],
                "Suggestions": [
                    "Try different keywords • Broaden your search • "
                    "Check spelling • Try another category • Simplify your query"
                ]
            })

            # All other tabs should be empty
            return message_df, "", "", "", ""
        else:
            # For other errors, show simple message in Papers tab
            error_df = pd.DataFrame({
                "Error": [f"⚠️ {error_text}"]
            })

            return error_df, "", "", "", ""


# Initialize the analyzer
analyzer = ResearchPaperAnalyzer()

# Define arXiv categories
ARXIV_CATEGORIES = [
    "All",
    "cs.AI - Artificial Intelligence",
    "cs.CL - Computation and Language",
    "cs.CV - Computer Vision",
    "cs.LG - Machine Learning",
    "cs.NE - Neural and Evolutionary Computing",
    "cs.RO - Robotics",
    "stat.ML - Machine Learning (Statistics)"
]


def analyze_research(query, category, num_papers, progress=gr.Progress()):
    """Gradio interface function."""
    # Extract category code
    cat_code = category.split(" - ")[0] if category != "All" else "All"
    yield from analyzer.run_workflow(query, cat_code, num_papers, progress)


# Create Gradio interface
with gr.Blocks(title="Research Paper Analyzer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Research Paper Analyzer
    ### Multi-Agent System for Analyzing Academic Papers from arXiv

    This tool uses AI agents to search arXiv, analyze papers, synthesize findings, and provide citation-backed insights.
    """)

    with gr.Row():
        with gr.Column(scale=2):
            query_input = gr.Textbox(
                label="Research Question",
                placeholder="What are the latest advances in multi-agent reinforcement learning?",
                lines=3
            )
        with gr.Column(scale=1):
            category_input = gr.Dropdown(
                choices=ARXIV_CATEGORIES,
                label="arXiv Category",
                value="All"
            )
            num_papers_input = gr.Slider(
                minimum=1,
                maximum=20,
                value=5,
                step=1,
                label="Number of Papers"
            )

    analyze_btn = gr.Button("Analyze Papers", variant="primary", size="lg")

    with gr.Tabs() as tabs:
        with gr.Tab("Papers"):
            papers_output = gr.Dataframe(
                label="Retrieved Papers",
                wrap=True,
                datatype=["str", "str", "str", "str", "str", "str", "markdown"],  # Last column is markdown for clickable links
                column_widths=["25%", "20%", "8%", "10%", "8%", "10%", "19%"]
            )

        with gr.Tab("Analysis"):
            analysis_output = gr.HTML(label="Paper Analyses")

        with gr.Tab("Synthesis"):
            synthesis_output = gr.HTML(label="Synthesis Report")

        with gr.Tab("Citations"):
            citations_output = gr.HTML(label="Citations")

        with gr.Tab("Stats"):
            stats_output = gr.HTML(label="Processing Statistics")

    analyze_btn.click(
        fn=analyze_research,
        inputs=[query_input, category_input, num_papers_input],
        outputs=[papers_output, analysis_output, synthesis_output, citations_output, stats_output]
    )

    gr.Markdown("""
    ---
    ### How it works:
    1. **Retriever Agent**: Searches arXiv and downloads papers
    2. **Analyzer Agent**: Extracts key information from each paper using RAG
    3. **Synthesis Agent**: Compares findings and identifies patterns
    4. **Citation Agent**: Validates claims and generates proper citations

    **Note**: Requires Azure OpenAI credentials. Results are cached for efficiency.
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
