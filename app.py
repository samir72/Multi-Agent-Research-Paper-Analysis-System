"""
Main Gradio application with LangGraph agent orchestration.
"""
import os
import time
import logging
from typing import Dict, Any, Tuple
from pathlib import Path
from dotenv import load_dotenv
import gradio as gr
import pandas as pd

# Load environment variables
load_dotenv()

# Import utilities
from utils.arxiv_client import ArxivClient
from utils.pdf_processor import PDFProcessor
from utils.cache import SemanticCache
from utils.schemas import AgentState

# Import RAG components
from rag.embeddings import EmbeddingGenerator
from rag.vector_store import VectorStore
from rag.retrieval import RAGRetriever

# Import agents
from agents.retriever import RetrieverAgent
from agents.analyzer import AnalyzerAgent
from agents.synthesis import SynthesisAgent
from agents.citation import CitationAgent


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ResearchPaperAnalyzer:
    """Main application class for research paper analysis."""

    def __init__(self):
        """Initialize the analyzer with all components."""
        logger.info("Initializing Research Paper Analyzer...")

        # Initialize components
        self.arxiv_client = ArxivClient()
        self.pdf_processor = PDFProcessor()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore()
        self.rag_retriever = RAGRetriever(
            vector_store=self.vector_store,
            embedding_generator=self.embedding_generator
        )
        self.cache = SemanticCache()

        # Initialize agents
        self.retriever_agent = RetrieverAgent(
            arxiv_client=self.arxiv_client,
            pdf_processor=self.pdf_processor,
            vector_store=self.vector_store,
            embedding_generator=self.embedding_generator
        )
        self.analyzer_agent = AnalyzerAgent(rag_retriever=self.rag_retriever)
        self.synthesis_agent = SynthesisAgent(rag_retriever=self.rag_retriever)
        self.citation_agent = CitationAgent(rag_retriever=self.rag_retriever)

        logger.info("Initialization complete")

    def run_workflow(
        self,
        query: str,
        category: str,
        num_papers: int,
        progress=gr.Progress()
    ) -> Tuple[pd.DataFrame, str, str, str, str]:
        """
        Execute the complete research paper analysis workflow.

        Args:
            query: Research question
            category: arXiv category
            num_papers: Number of papers to analyze
            progress: Gradio progress tracker

        Returns:
            Tuple of (papers_df, analysis_html, synthesis_html, citations_html, stats)
        """
        try:
            start_time = time.time()

            # Check cache first
            progress(0.0, desc="Checking cache...")
            query_embedding = self.embedding_generator.generate_embedding(query)
            cached_result = self.cache.get(query, query_embedding, category)

            if cached_result:
                logger.info("Using cached result")
                return self._format_output(cached_result)

            # Initialize state
            state = {
                "query": query,
                "category": category if category != "All" else None,
                "num_papers": num_papers,
                "papers": [],
                "chunks": [],
                "analyses": [],
                "synthesis": None,
                "validated_output": None,
                "errors": [],
                "token_usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "embedding_tokens": 0
                }
            }

            # Step 1: Retriever Agent
            progress(0.1, desc="Searching and downloading papers...")
            state = self.retriever_agent.run(state)
            if state.get("errors") and not state.get("papers"):
                return self._format_error(state["errors"])

            # Step 2: Analyzer Agent
            progress(0.4, desc="Analyzing individual papers...")
            state = self.analyzer_agent.run(state)
            if state.get("errors") and not state.get("analyses"):
                return self._format_error(state["errors"])

            # Step 3: Synthesis Agent
            progress(0.7, desc="Synthesizing findings across papers...")
            state = self.synthesis_agent.run(state)
            if state.get("errors") and not state.get("synthesis"):
                return self._format_error(state["errors"])

            # Step 4: Citation Agent
            progress(0.9, desc="Validating and generating citations...")
            processing_time = time.time() - start_time
            state["processing_time"] = processing_time
            state = self.citation_agent.run(state)

            progress(1.0, desc="Complete!")

            # Cache the result
            result = {
                "papers": state["papers"],
                "analyses": state["analyses"],
                "validated_output": state["validated_output"]
            }
            self.cache.set(query, query_embedding, result, category)

            # Format output
            return self._format_output(result)

        except Exception as e:
            logger.error(f"Workflow error: {str(e)}")
            return self._format_error([str(e)])

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
        for paper in papers:
            papers_data.append({
                "Title": paper.title,
                "Authors": ", ".join(paper.authors[:3]) + ("..." if len(paper.authors) > 3 else ""),
                "Date": paper.published.strftime("%Y-%m-%d"),
                "arXiv ID": paper.arxiv_id,
                "Link": paper.pdf_url
            })
        papers_df = pd.DataFrame(papers_data)

        # Format analysis
        analysis_html = "<h2>Paper Analyses</h2>"
        for paper, analysis in zip(papers, analyses):
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
                <p><strong>Supporting Papers:</strong> {", ".join(cp.supporting_papers)}</p>
                <p><strong>Confidence:</strong> {cp.confidence:.2%}</p>
            </div>
            ''' for cp in synthesis.consensus_points)}
        </div>

        <div style="margin-bottom: 30px;">
            <h3 style="color: #f57c00;">Contradictions</h3>
            {"".join(f'''
            <div style="background-color: #fff8e1; padding: 15px; margin-bottom: 10px; border-radius: 5px; border-left: 4px solid #ffa726;">
                <p style="font-weight: bold;">Topic: {c.topic}</p>
                <p><strong>Viewpoint A:</strong> {c.viewpoint_a} (Papers: {", ".join(c.papers_a)})</p>
                <p><strong>Viewpoint B:</strong> {c.viewpoint_b} (Papers: {", ".join(c.papers_b)})</p>
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
            <li>Papers Analyzed: {len(papers)}</li>
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
        """Format error message."""
        error_html = f"""
        <div style="background-color: #ffebee; padding: 20px; border-radius: 10px; border-left: 4px solid #f44336;">
            <h3 style="color: #c62828;">Error</h3>
            <p>{" ".join(errors)}</p>
        </div>
        """
        return pd.DataFrame(), error_html, "", "", ""


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
    return analyzer.run_workflow(query, cat_code, num_papers, progress)


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
                wrap=True
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
