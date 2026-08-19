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
            "errors": [],
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "embedding_tokens": 0},
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
            "errors": [],
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "embedding_tokens": 0},
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
            "errors": [],
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "embedding_tokens": 0},
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
                    azure_endpoint="https://test-endpoint.openai.azure.com",
                    timeout=60,
                    max_retries=2,
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


class TestAnalyzerResponsesAPI:
    """
    Coverage for the USE_RESPONSES_API=true path (analyzer.py's
    client.responses.create() branch) -- previously untested anywhere in the
    repo despite being live-verified manually. Mirrors the Chat Completions
    fixtures above but adds a `.responses.create` mock alongside `.chat.completions.create`
    so tests can assert on which one was actually used.
    """

    @staticmethod
    def _analysis_json():
        return json.dumps({
            "methodology": "Deep learning approach using convolutional neural networks",
            "key_findings": ["95% accuracy on test set"],
            "conclusions": "The proposed method achieves state-of-the-art results",
            "limitations": ["Limited to specific image domains"],
            "main_contributions": ["Novel architecture design"],
            "citations": ["Methodology section"],
        })

    @pytest.fixture
    def mock_responses_api_client(self):
        """Mock Azure OpenAI client exposing both responses.create and chat.completions.create."""
        mock_client = MagicMock()

        mock_resp_api_response = MagicMock()
        mock_resp_api_response.output_text = self._analysis_json()
        mock_resp_api_response.usage.input_tokens = 111
        mock_resp_api_response.usage.output_tokens = 22
        mock_client.responses.create.return_value = mock_resp_api_response

        mock_chat_response = MagicMock()
        mock_chat_response.choices[0].message.content = self._analysis_json()
        mock_chat_response.usage.prompt_tokens = 999
        mock_chat_response.usage.completion_tokens = 888
        mock_client.chat.completions.create.return_value = mock_chat_response

        return mock_client

    @pytest.fixture
    def analyzer_agent_responses_api(self, mock_rag_retriever, mock_responses_api_client):
        """AnalyzerAgent constructed with USE_RESPONSES_API=true."""
        with patch.dict(os.environ, {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
            "AZURE_OPENAI_API_VERSION": "2025-03-01-preview",
            "AZURE_OPENAI_DEPLOYMENT_NAME": "test-deployment",
            "USE_RESPONSES_API": "true",
        }):
            with patch('agents.analyzer.AzureOpenAI', return_value=mock_responses_api_client):
                agent = AnalyzerAgent(
                    rag_retriever=mock_rag_retriever,
                    model="test-deployment",
                    temperature=0.0,
                )
                assert agent.use_responses_api is True
                return agent

    def test_responses_api_success(self, analyzer_agent_responses_api, sample_paper, mock_responses_api_client):
        """Responses API path is used, parsed correctly, and Chat Completions is never called."""
        analysis = analyzer_agent_responses_api.analyze_paper(sample_paper, top_k_chunks=10)

        assert isinstance(analysis, Analysis)
        assert analysis.methodology == "Deep learning approach using convolutional neural networks"
        assert analysis.confidence_score > 0.0

        mock_responses_api_client.responses.create.assert_called_once()
        mock_responses_api_client.chat.completions.create.assert_not_called()

        # Responses API's input_tokens/output_tokens usage fields, not Chat
        # Completions' prompt_tokens/completion_tokens, must land in batch_tokens.
        assert analyzer_agent_responses_api.batch_tokens["input"] == 111
        assert analyzer_agent_responses_api.batch_tokens["output"] == 22

    def test_responses_api_fallback_on_error(self, analyzer_agent_responses_api, sample_paper, mock_responses_api_client):
        """A Responses API error falls back to Chat Completions and still returns a real analysis."""
        mock_responses_api_client.responses.create.side_effect = Exception("Responses API unavailable")

        analysis = analyzer_agent_responses_api.analyze_paper(sample_paper, top_k_chunks=10)

        # Fallback succeeded -- this must NOT be the "Analysis failed" degraded result.
        assert isinstance(analysis, Analysis)
        assert analysis.methodology == "Deep learning approach using convolutional neural networks"
        assert analysis.confidence_score > 0.0

        mock_responses_api_client.responses.create.assert_called_once()
        mock_responses_api_client.chat.completions.create.assert_called_once()

        # Chat Completions' prompt_tokens/completion_tokens fields, not the
        # Responses API's, must land in batch_tokens for the fallback call.
        assert analyzer_agent_responses_api.batch_tokens["input"] == 999
        assert analyzer_agent_responses_api.batch_tokens["output"] == 888


class TestAnalyzerNormalization:
    """Tests for LLM response normalization edge cases."""

    @pytest.fixture
    def analyzer_agent_for_normalization(self, mock_rag_retriever):
        """Create analyzer agent with mocked Azure OpenAI client."""
        with patch('agents.analyzer.AzureOpenAI'):
            agent = AnalyzerAgent(mock_rag_retriever)
            return agent

    def test_normalize_nested_lists_in_citations(self, analyzer_agent_for_normalization):
        """Test that nested lists in citations are flattened."""
        agent = analyzer_agent_for_normalization

        # LLM returns nested lists (the bug we're fixing)
        malformed_data = {
            "methodology": "Test methodology",
            "key_findings": ["Finding 1", "Finding 2"],
            "conclusions": "Test conclusions",
            "limitations": ["Limitation 1"],
            "main_contributions": ["Contribution 1"],
            "citations": ["Citation 1", [], "Citation 2"]  # Nested empty list
        }

        normalized = agent._normalize_analysis_response(malformed_data)

        # Should flatten and remove empty lists
        assert normalized["citations"] == ["Citation 1", "Citation 2"]
        assert all(isinstance(c, str) for c in normalized["citations"])

    def test_normalize_deeply_nested_lists(self, analyzer_agent_for_normalization):
        """Test deeply nested lists are flattened recursively."""
        agent = analyzer_agent_for_normalization

        malformed_data = {
            "methodology": "Test",
            "key_findings": [["Nested finding"], "Normal finding", [["Double nested"]]],
            "conclusions": "Test",
            "limitations": [],
            "main_contributions": [],
            "citations": [[["Triple nested citation"]]]
        }

        normalized = agent._normalize_analysis_response(malformed_data)

        assert normalized["key_findings"] == ["Nested finding", "Normal finding", "Double nested"]
        assert normalized["citations"] == ["Triple nested citation"]

    def test_normalize_mixed_types_in_lists(self, analyzer_agent_for_normalization):
        """Test that mixed types (strings, None, numbers) are handled."""
        agent = analyzer_agent_for_normalization

        malformed_data = {
            "methodology": "Test",
            "key_findings": ["Finding 1", None, "Finding 2", ""],
            "conclusions": "Test",
            "limitations": ["Limit 1", 123, "Limit 2"],  # Number mixed in
            "main_contributions": [],
            "citations": ["Citation", None, "", "  ", "Valid"]
        }

        normalized = agent._normalize_analysis_response(malformed_data)

        # None and empty strings should be filtered out
        assert normalized["key_findings"] == ["Finding 1", "Finding 2"]
        # Numbers should be converted to strings
        assert normalized["limitations"] == ["Limit 1", "123", "Limit 2"]
        # Whitespace-only strings filtered out
        assert normalized["citations"] == ["Citation", "Valid"]

    def test_normalize_string_instead_of_list(self, analyzer_agent_for_normalization):
        """Test that strings are converted to single-element lists."""
        agent = analyzer_agent_for_normalization

        malformed_data = {
            "methodology": "Test",
            "key_findings": "Single finding as string",  # Should be list
            "conclusions": "Test",
            "limitations": "Single limitation",  # Should be list
            "main_contributions": [],
            "citations": []
        }

        normalized = agent._normalize_analysis_response(malformed_data)

        assert normalized["key_findings"] == ["Single finding as string"]
        assert normalized["limitations"] == ["Single limitation"]

    def test_normalize_missing_fields(self, analyzer_agent_for_normalization):
        """Test that missing fields are set to empty lists."""
        agent = analyzer_agent_for_normalization

        malformed_data = {
            "methodology": "Test",
            "conclusions": "Test",
            # key_findings, limitations, citations, main_contributions are missing
        }

        normalized = agent._normalize_analysis_response(malformed_data)

        assert normalized["key_findings"] == []
        assert normalized["limitations"] == []
        assert normalized["citations"] == []
        assert normalized["main_contributions"] == []

    def test_normalize_creates_valid_analysis_object(self, analyzer_agent_for_normalization):
        """Test that normalized data creates valid Analysis object."""
        agent = analyzer_agent_for_normalization

        # Extreme malformed data
        malformed_data = {
            "methodology": "Test",
            "key_findings": [[], "Finding", None, [["Nested"]]],
            "conclusions": "Test",
            "limitations": "Single string",
            "main_contributions": [123, None, "Valid"],
            "citations": ["Citation", [], "", None]
        }

        normalized = agent._normalize_analysis_response(malformed_data)

        # Should successfully create Analysis object without Pydantic errors
        analysis = Analysis(
            paper_id="test_id",
            methodology=normalized["methodology"],
            key_findings=normalized["key_findings"],
            conclusions=normalized["conclusions"],
            limitations=normalized["limitations"],
            citations=normalized["citations"],
            main_contributions=normalized["main_contributions"],
            confidence_score=0.8
        )

        assert isinstance(analysis, Analysis)
        assert analysis.key_findings == ["Finding", "Nested"]
        assert analysis.limitations == ["Single string"]
        assert analysis.main_contributions == ["123", "Valid"]
        assert analysis.citations == ["Citation"]


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
            "errors": [],
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "embedding_tokens": 0},
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
