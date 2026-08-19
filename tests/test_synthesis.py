"""
Unit tests for Synthesis Agent.

Minimal coverage -- this module had zero tests before. Mirrors the shape of
tests/test_analyzer.py (same mock/fixture style) rather than re-testing
identical ground; focuses on synthesize()/run() behavior and the
USE_RESPONSES_API=true path (previously untested anywhere, despite being
live-verified manually).
"""
import os
import json
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from agents.synthesis import SynthesisAgent
from utils.schemas import Paper, Analysis, SynthesisResult
from rag.retrieval import RAGRetriever


def _synthesis_json():
    return json.dumps({
        "consensus_points": [
            {
                "statement": "Deep learning outperforms classical methods",
                "supporting_papers": ["2401.00001"],
                "citations": ["Results section"],
                "confidence": 0.9,
            }
        ],
        "contradictions": [],
        "research_gaps": ["Generalization to other domains is untested"],
        "summary": "Both papers support deep learning approaches for the query.",
        "confidence_score": 0.85,
    })


@pytest.fixture
def mock_rag_retriever():
    return MagicMock(spec=RAGRetriever)


@pytest.fixture
def sample_papers():
    return [
        Paper(
            arxiv_id="2401.00001",
            title="Deep Learning for Image Classification",
            authors=["John Doe"],
            abstract="Abstract A",
            pdf_url="https://arxiv.org/pdf/2401.00001.pdf",
            published=datetime(2024, 1, 1),
            categories=["cs.CV"],
        ),
        Paper(
            arxiv_id="2401.00002",
            title="Transformers for Vision",
            authors=["Jane Smith"],
            abstract="Abstract B",
            pdf_url="https://arxiv.org/pdf/2401.00002.pdf",
            published=datetime(2024, 1, 2),
            categories=["cs.CV"],
        ),
    ]


@pytest.fixture
def sample_analyses():
    return [
        Analysis(
            paper_id="2401.00001",
            methodology="CNN-based approach",
            key_findings=["95% accuracy"],
            conclusions="Strong results",
            limitations=["Small dataset"],
            citations=["Results"],
            main_contributions=["Novel architecture"],
            confidence_score=0.8,
        ),
        Analysis(
            paper_id="2401.00002",
            methodology="Transformer-based approach",
            key_findings=["97% accuracy"],
            conclusions="State of the art",
            limitations=["Compute-heavy"],
            citations=["Results"],
            main_contributions=["Attention mechanism"],
            confidence_score=0.85,
        ),
    ]


@pytest.fixture
def mock_azure_client():
    """Mock Azure OpenAI client for the default Chat Completions path."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = _synthesis_json()
    mock_response.usage.prompt_tokens = 500
    mock_response.usage.completion_tokens = 200
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def synthesis_agent(mock_rag_retriever, mock_azure_client):
    with patch.dict(os.environ, {
        "AZURE_OPENAI_API_KEY": "test_key",
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        "AZURE_OPENAI_API_VERSION": "2024-02-01",
        "AZURE_OPENAI_DEPLOYMENT_NAME": "test-deployment",
    }, clear=False):
        with patch('agents.synthesis.AzureOpenAI', return_value=mock_azure_client):
            return SynthesisAgent(rag_retriever=mock_rag_retriever, model="test-deployment", temperature=0.0)


class TestSynthesisAgentInit:
    def test_init(self, mock_rag_retriever):
        with patch.dict(os.environ, {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
            "AZURE_OPENAI_API_VERSION": "2024-02-01",
            "AZURE_OPENAI_DEPLOYMENT_NAME": "test-deployment",
        }):
            with patch('agents.synthesis.AzureOpenAI'):
                agent = SynthesisAgent(rag_retriever=mock_rag_retriever, model="test-model", temperature=0.3)
                assert agent.rag_retriever == mock_rag_retriever
                assert agent.model == "test-model"
                assert agent.temperature == 0.3
                assert agent.use_responses_api is False


class TestSynthesisPrompt:
    def test_create_synthesis_prompt_includes_paper_content(self, synthesis_agent, sample_papers, sample_analyses):
        prompt = synthesis_agent._create_synthesis_prompt(sample_papers, sample_analyses, "What works best?")

        assert "What works best?" in prompt
        assert sample_papers[0].title in prompt
        assert sample_papers[1].title in prompt
        assert "CNN-based approach" in prompt
        assert "Transformer-based approach" in prompt


class TestSynthesizeChatCompletions:
    def test_synthesize_success(self, synthesis_agent, sample_papers, sample_analyses, mock_azure_client):
        state = {"token_usage": {"input_tokens": 0, "output_tokens": 0, "embedding_tokens": 0}}

        result = synthesis_agent.synthesize(sample_papers, sample_analyses, "What works best?", state)

        assert isinstance(result, SynthesisResult)
        assert result.confidence_score == 0.85
        assert len(result.consensus_points) == 1
        assert result.consensus_points[0].statement == "Deep learning outperforms classical methods"
        assert result.papers_analyzed == ["2401.00001", "2401.00002"]

        # Chat Completions' prompt_tokens/completion_tokens must land in state.
        assert state["token_usage"]["input_tokens"] == 500
        assert state["token_usage"]["output_tokens"] == 200

    def test_synthesize_error_returns_minimal_result(self, synthesis_agent, sample_papers, sample_analyses, mock_azure_client):
        mock_azure_client.chat.completions.create.side_effect = Exception("API error")
        state = {"token_usage": {"input_tokens": 0, "output_tokens": 0, "embedding_tokens": 0}}

        result = synthesis_agent.synthesize(sample_papers, sample_analyses, "What works best?", state)

        assert isinstance(result, SynthesisResult)
        assert result.confidence_score == 0.0
        assert result.summary == "Synthesis failed due to an error"
        assert result.papers_analyzed == ["2401.00001", "2401.00002"]


class TestSynthesisAgentRun:
    def test_run_success(self, synthesis_agent, sample_papers, sample_analyses):
        state = {
            "query": "What works best?",
            "papers": sample_papers,
            "analyses": sample_analyses,
            "errors": [],
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "embedding_tokens": 0},
        }

        result_state = synthesis_agent.run(state)

        assert "synthesis" in result_state
        assert isinstance(result_state["synthesis"], SynthesisResult)
        assert result_state["errors"] == []

    def test_run_missing_analyses_adds_error(self, synthesis_agent, sample_papers):
        state = {
            "query": "What works best?",
            "papers": sample_papers,
            "analyses": [],
            "errors": [],
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "embedding_tokens": 0},
        }

        result_state = synthesis_agent.run(state)

        assert "synthesis" not in result_state
        assert len(result_state["errors"]) == 1
        assert "No papers or analyses" in result_state["errors"][0]

    def test_run_mismatched_lengths_truncates_to_min(self, synthesis_agent, sample_papers, sample_analyses):
        state = {
            "query": "What works best?",
            "papers": sample_papers,  # 2 papers
            "analyses": sample_analyses[:1],  # 1 analysis
            "errors": [],
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "embedding_tokens": 0},
        }

        result_state = synthesis_agent.run(state)

        assert "synthesis" in result_state
        assert result_state["synthesis"].papers_analyzed == ["2401.00001"]


class TestSynthesisResponsesAPI:
    """Coverage for the USE_RESPONSES_API=true path (previously untested)."""

    @pytest.fixture
    def mock_responses_api_client(self):
        mock_client = MagicMock()

        mock_resp_api_response = MagicMock()
        mock_resp_api_response.output_text = _synthesis_json()
        mock_resp_api_response.usage.input_tokens = 321
        mock_resp_api_response.usage.output_tokens = 123
        mock_client.responses.create.return_value = mock_resp_api_response

        mock_chat_response = MagicMock()
        mock_chat_response.choices[0].message.content = _synthesis_json()
        mock_chat_response.usage.prompt_tokens = 500
        mock_chat_response.usage.completion_tokens = 200
        mock_client.chat.completions.create.return_value = mock_chat_response

        return mock_client

    @pytest.fixture
    def synthesis_agent_responses_api(self, mock_rag_retriever, mock_responses_api_client):
        with patch.dict(os.environ, {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
            "AZURE_OPENAI_API_VERSION": "2025-03-01-preview",
            "AZURE_OPENAI_DEPLOYMENT_NAME": "test-deployment",
            "USE_RESPONSES_API": "true",
        }):
            with patch('agents.synthesis.AzureOpenAI', return_value=mock_responses_api_client):
                agent = SynthesisAgent(rag_retriever=mock_rag_retriever, model="test-deployment", temperature=0.0)
                assert agent.use_responses_api is True
                return agent

    def test_responses_api_success(self, synthesis_agent_responses_api, sample_papers, sample_analyses, mock_responses_api_client):
        state = {"token_usage": {"input_tokens": 0, "output_tokens": 0, "embedding_tokens": 0}}

        result = synthesis_agent_responses_api.synthesize(sample_papers, sample_analyses, "What works best?", state)

        assert isinstance(result, SynthesisResult)
        assert result.confidence_score == 0.85
        mock_responses_api_client.responses.create.assert_called_once()
        mock_responses_api_client.chat.completions.create.assert_not_called()

        # Responses API's input_tokens/output_tokens, not Chat Completions'
        # prompt_tokens/completion_tokens, must land in state.
        assert state["token_usage"]["input_tokens"] == 321
        assert state["token_usage"]["output_tokens"] == 123

    def test_responses_api_fallback_on_error(self, synthesis_agent_responses_api, sample_papers, sample_analyses, mock_responses_api_client):
        mock_responses_api_client.responses.create.side_effect = Exception("Responses API unavailable")
        state = {"token_usage": {"input_tokens": 0, "output_tokens": 0, "embedding_tokens": 0}}

        result = synthesis_agent_responses_api.synthesize(sample_papers, sample_analyses, "What works best?", state)

        # Fallback succeeded -- must NOT be the "Synthesis failed" degraded result.
        assert isinstance(result, SynthesisResult)
        assert result.confidence_score == 0.85
        assert result.summary != "Synthesis failed due to an error"

        mock_responses_api_client.responses.create.assert_called_once()
        mock_responses_api_client.chat.completions.create.assert_called_once()

        assert state["token_usage"]["input_tokens"] == 500
        assert state["token_usage"]["output_tokens"] == 200


class TestSynthesisNormalization:
    """Sanity check for _normalize_synthesis_response's nested-list handling."""

    def test_normalize_nested_lists_in_research_gaps(self, synthesis_agent):
        malformed_data = {
            "consensus_points": [],
            "contradictions": [],
            "research_gaps": ["Gap 1", [], ["Gap 2"], None],
            "summary": "Test",
            "confidence_score": 0.5,
        }

        normalized = synthesis_agent._normalize_synthesis_response(malformed_data)

        assert normalized["research_gaps"] == ["Gap 1", "Gap 2"]

    def test_normalize_nested_lists_in_consensus_point_fields(self, synthesis_agent):
        malformed_data = {
            "consensus_points": [
                {"statement": "X", "supporting_papers": ["a", ["b"]], "citations": [[], "c"], "confidence": 0.5}
            ],
            "contradictions": [],
            "research_gaps": [],
            "summary": "Test",
            "confidence_score": 0.5,
        }

        normalized = synthesis_agent._normalize_synthesis_response(malformed_data)

        assert normalized["consensus_points"][0]["supporting_papers"] == ["a", "b"]
        assert normalized["consensus_points"][0]["citations"] == ["c"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
