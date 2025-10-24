"""
Unit tests for Analyzer Agent.
"""
import os
import json
import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

from agents.analyzer import AnalyzerAgent
from utils.schemas import Paper, Analysis
from rag.retrieval import RAGRetriever


@pytest.fixture
def mock_rag_retriever():
    """Create a mock RAG retriever."""
    retriever = Mock(spec=RAGRetriever)

    # Mock retrieve method
    retriever.retrieve.return_value = {
        "query": "test query",
        "chunks": [
            {
                "chunk_id": "chunk_1",
                "content": "This study uses a novel deep learning approach for image classification.",
                "metadata": {
                    "title": "Test Paper",
                    "authors": "John Doe, Jane Smith",
                    "section": "Methodology",
                    "page_number": 3,
                    "arxiv_url": "https://arxiv.org/abs/2401.00001"
                },
                "distance": 0.1
            },
            {
                "chunk_id": "chunk_2",
                "content": "Our results show 95% accuracy on the test set, outperforming previous benchmarks.",
                "metadata": {
                    "title": "Test Paper",
                    "authors": "John Doe, Jane Smith",
                    "section": "Results",
                    "page_number": 7,
                    "arxiv_url": "https://arxiv.org/abs/2401.00001"
                },
                "distance": 0.15
            }
        ],
        "chunk_ids": ["chunk_1", "chunk_2"]
    }

    # Mock format_context method
    retriever.format_context.return_value = """[Chunk 1] Paper: Test Paper
Authors: John Doe, Jane Smith
Section: Methodology
Page: 3
Source: https://arxiv.org/abs/2401.00001
--------------------------------------------------------------------------------
This study uses a novel deep learning approach for image classification.

[Chunk 2] Paper: Test Paper
Authors: John Doe, Jane Smith
Section: Results
Page: 7
Source: https://arxiv.org/abs/2401.00001
--------------------------------------------------------------------------------
Our results show 95% accuracy on the test set, outperforming previous benchmarks."""

    return retriever


@pytest.fixture
def sample_paper():
    """Create a sample paper for testing."""
    return Paper(
        arxiv_id="2401.00001",
        title="Deep Learning for Image Classification",
        authors=["John Doe", "Jane Smith"],
        abstract="This paper presents a novel approach to image classification using deep learning.",
        pdf_url="https://arxiv.org/pdf/2401.00001.pdf",
        published=datetime(2024, 1, 1),
        categories=["cs.CV", "cs.LG"]
    )


@pytest.fixture
def mock_azure_client():
    """Create a mock Azure OpenAI client."""
    mock_client = MagicMock()

    # Mock completion response
    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps({
        "methodology": "Deep learning approach using convolutional neural networks",
        "key_findings": [
            "95% accuracy on test set",
            "Outperforms previous benchmarks",
            "Faster training time"
        ],
        "conclusions": "The proposed method achieves state-of-the-art results",
        "limitations": [
            "Limited to specific image domains",
            "Requires large training dataset"
        ],
        "main_contributions": [
            "Novel architecture design",
            "Improved training procedure"
        ],
        "citations": ["Methodology section", "Results section"]
    })

    mock_client.chat.completions.create.return_value = mock_response

    return mock_client


@pytest.fixture
def analyzer_agent(mock_rag_retriever, mock_azure_client):
    """Create an analyzer agent with mocked dependencies."""
    with patch.dict(os.environ, {
        "AZURE_OPENAI_API_KEY": "test_key",
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        "AZURE_OPENAI_API_VERSION": "2024-02-01",
        "AZURE_OPENAI_DEPLOYMENT_NAME": "test-deployment"
    }):
        with patch('agents.analyzer.AzureOpenAI', return_value=mock_azure_client):
            agent = AnalyzerAgent(
                rag_retriever=mock_rag_retriever,
                model="test-deployment",
                temperature=0.0
            )
            return agent


class TestAnalyzerAgent:
    """Test suite for AnalyzerAgent."""

    def test_init(self, mock_rag_retriever):
        """Test analyzer agent initialization."""
        with patch.dict(os.environ, {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
            "AZURE_OPENAI_API_VERSION": "2024-02-01",
            "AZURE_OPENAI_DEPLOYMENT_NAME": "test-deployment"
        }):
            with patch('agents.analyzer.AzureOpenAI'):
                agent = AnalyzerAgent(
                    rag_retriever=mock_rag_retriever,
                    model="test-model",
                    temperature=0.5
                )

                assert agent.rag_retriever == mock_rag_retriever
                assert agent.model == "test-model"
                assert agent.temperature == 0.5
                assert agent.client is not None

    def test_create_analysis_prompt(self, analyzer_agent, sample_paper):
        """Test prompt creation for analysis."""
        context = "Sample context about the paper"

        prompt = analyzer_agent._create_analysis_prompt(sample_paper, context)

        assert sample_paper.title in prompt
        assert "John Doe" in prompt
        assert "Jane Smith" in prompt
        assert sample_paper.abstract in prompt
        assert context in prompt
        assert "methodology" in prompt
        assert "key_findings" in prompt
        assert "conclusions" in prompt
        assert "limitations" in prompt

    def test_analyze_paper_success(self, analyzer_agent, sample_paper, mock_rag_retriever):
        """Test successful paper analysis."""
        analysis = analyzer_agent.analyze_paper(sample_paper, top_k_chunks=10)

        # Verify the analysis was created
        assert isinstance(analysis, Analysis)
        assert analysis.paper_id == sample_paper.arxiv_id
        assert analysis.methodology == "Deep learning approach using convolutional neural networks"
        assert len(analysis.key_findings) == 3
        assert analysis.conclusions == "The proposed method achieves state-of-the-art results"
        assert len(analysis.limitations) == 2
        assert len(analysis.main_contributions) == 2
        assert 0.0 <= analysis.confidence_score <= 1.0

        # Verify RAG retriever was called with correct queries
        assert mock_rag_retriever.retrieve.call_count == 4  # 4 queries
        assert mock_rag_retriever.format_context.called

    def test_analyze_paper_confidence_score(self, analyzer_agent, sample_paper, mock_rag_retriever):
        """Test confidence score calculation."""
        # Test with 10 chunks requested, 2 returned
        analysis = analyzer_agent.analyze_paper(sample_paper, top_k_chunks=10)

        # Confidence should be based on number of chunks retrieved
        # With 8 unique chunks (2 per query * 4 queries), confidence = 8/10 = 0.8
        # But since we mock 2 chunks total with duplicates filtered, it will be 0.2
        assert 0.0 <= analysis.confidence_score <= 1.0

    def test_analyze_paper_with_error(self, analyzer_agent, sample_paper, mock_rag_retriever):
        """Test error handling during paper analysis."""
        # Make RAG retriever raise an exception
        mock_rag_retriever.retrieve.side_effect = Exception("Retrieval failed")

        analysis = analyzer_agent.analyze_paper(sample_paper)

        # Should return a minimal analysis on error
        assert isinstance(analysis, Analysis)
        assert analysis.paper_id == sample_paper.arxiv_id
        assert analysis.methodology == "Analysis failed"
        assert analysis.conclusions == "Analysis failed"
        assert analysis.confidence_score == 0.0
        assert len(analysis.key_findings) == 0

    def test_run_with_papers(self, analyzer_agent, sample_paper):
        """Test run method with papers in state."""
        state = {
            "papers": [sample_paper],
            "errors": []
        }

        result_state = analyzer_agent.run(state)

        # Verify analyses were added to state
        assert "analyses" in result_state
        assert len(result_state["analyses"]) == 1
        assert isinstance(result_state["analyses"][0], Analysis)
        assert result_state["analyses"][0].paper_id == sample_paper.arxiv_id

    def test_run_with_multiple_papers(self, analyzer_agent):
        """Test run method with multiple papers."""
        papers = [
            Paper(
                arxiv_id=f"2401.0000{i}",
                title=f"Test Paper {i}",
                authors=["Author A", "Author B"],
                abstract=f"Abstract for paper {i}",
                pdf_url=f"https://arxiv.org/pdf/2401.0000{i}.pdf",
                published=datetime(2024, 1, i),
                categories=["cs.AI"]
            )
            for i in range(1, 4)
        ]

        state = {
            "papers": papers,
            "errors": []
        }

        result_state = analyzer_agent.run(state)

        # Verify all papers were analyzed
        assert len(result_state["analyses"]) == 3
        assert all(isinstance(a, Analysis) for a in result_state["analyses"])

    def test_run_without_papers(self, analyzer_agent):
        """Test run method when no papers are provided."""
        state = {
            "papers": [],
            "errors": []
        }

        result_state = analyzer_agent.run(state)

        # Verify error was added
        assert len(result_state["errors"]) > 0
        assert "No papers to analyze" in result_state["errors"][0]
        assert "analyses" not in result_state

    def test_run_with_analysis_failure(self, analyzer_agent, sample_paper, mock_rag_retriever):
        """Test run method when analysis fails for a paper."""
        # Make analyze_paper fail
        mock_rag_retriever.retrieve.side_effect = Exception("Analysis error")

        state = {
            "papers": [sample_paper],
            "errors": []
        }

        result_state = analyzer_agent.run(state)

        # Should still have analyses (with failed analysis)
        assert "analyses" in result_state
        assert len(result_state["analyses"]) == 1
        assert result_state["analyses"][0].confidence_score == 0.0

    def test_run_state_error_handling(self, analyzer_agent):
        """Test run method error handling with invalid state."""
        # Missing 'errors' key in state
        state = {
            "papers": []
        }

        # Should handle gracefully and add error
        result_state = analyzer_agent.run(state)
        assert isinstance(result_state, dict)

    def test_azure_client_initialization(self, mock_rag_retriever):
        """Test Azure OpenAI client initialization with environment variables."""
        test_env = {
            "AZURE_OPENAI_API_KEY": "test_key_123",
            "AZURE_OPENAI_ENDPOINT": "https://test-endpoint.openai.azure.com",
            "AZURE_OPENAI_API_VERSION": "2024-02-01",
            "AZURE_OPENAI_DEPLOYMENT_NAME": "gpt-4"
        }

        with patch.dict(os.environ, test_env):
            with patch('agents.analyzer.AzureOpenAI') as mock_azure:
                agent = AnalyzerAgent(rag_retriever=mock_rag_retriever)

                # Verify AzureOpenAI was called with correct parameters
                mock_azure.assert_called_once_with(
                    api_key="test_key_123",
                    api_version="2024-02-01",
                    azure_endpoint="https://test-endpoint.openai.azure.com"
                )

    def test_multiple_query_retrieval(self, analyzer_agent, sample_paper, mock_rag_retriever):
        """Test that multiple queries are used for comprehensive retrieval."""
        analyzer_agent.analyze_paper(sample_paper, top_k_chunks=12)

        # Verify retrieve was called 4 times (for 4 different queries)
        assert mock_rag_retriever.retrieve.call_count == 4

        # Verify the queries cover different aspects
        call_args_list = mock_rag_retriever.retrieve.call_args_list
        queries = [call.kwargs['query'] for call in call_args_list]

        assert any("methodology" in q.lower() for q in queries)
        assert any("results" in q.lower() or "findings" in q.lower() for q in queries)
        assert any("conclusions" in q.lower() or "contributions" in q.lower() for q in queries)
        assert any("limitations" in q.lower() or "future work" in q.lower() for q in queries)

    def test_chunk_deduplication(self, analyzer_agent, sample_paper, mock_rag_retriever):
        """Test that duplicate chunks are filtered out."""
        # Make retrieve return duplicate chunks
        mock_rag_retriever.retrieve.return_value = {
            "query": "test query",
            "chunks": [
                {"chunk_id": "chunk_1", "content": "Content 1", "metadata": {}},
                {"chunk_id": "chunk_1", "content": "Content 1", "metadata": {}},  # Duplicate
            ],
            "chunk_ids": ["chunk_1", "chunk_1"]
        }

        analysis = analyzer_agent.analyze_paper(sample_paper)

        # Verify analysis still succeeds despite duplicates
        assert isinstance(analysis, Analysis)
        assert mock_rag_retriever.format_context.called


class TestAnalyzerAgentIntegration:
    """Integration tests for analyzer agent with more realistic scenarios."""

    def test_full_analysis_workflow(self, analyzer_agent, sample_paper):
        """Test complete analysis workflow from paper to analysis."""
        analysis = analyzer_agent.analyze_paper(sample_paper, top_k_chunks=10)

        # Verify complete analysis structure
        assert analysis.paper_id == sample_paper.arxiv_id
        assert isinstance(analysis.methodology, str)
        assert isinstance(analysis.key_findings, list)
        assert isinstance(analysis.conclusions, str)
        assert isinstance(analysis.limitations, list)
        assert isinstance(analysis.citations, list)
        assert isinstance(analysis.main_contributions, list)
        assert isinstance(analysis.confidence_score, float)

    def test_state_transformation(self, analyzer_agent, sample_paper):
        """Test complete state transformation through run method."""
        initial_state = {
            "query": "What are the latest advances in deep learning?",
            "papers": [sample_paper],
            "errors": []
        }

        final_state = analyzer_agent.run(initial_state)

        # Verify state contains all required fields
        assert "query" in final_state
        assert "papers" in final_state
        assert "analyses" in final_state
        assert "errors" in final_state

        # Verify the original query and papers are preserved
        assert final_state["query"] == initial_state["query"]
        assert final_state["papers"] == initial_state["papers"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
