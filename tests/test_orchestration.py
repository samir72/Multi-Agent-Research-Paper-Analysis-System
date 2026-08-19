"""
Tests for LangGraph orchestration: Send-based analyzer fan-out, the
partial-delta node-return contract, and reducer correctness.

This module (orchestration/workflow_graph.py, orchestration/nodes.py) had
zero test coverage before the Responses API + LangGraph Send migration.
These tests specifically guard against the failure mode identified during
that migration: AgentState.analyses/errors/token_usage use LangGraph
reducers (operator.add / merge_token_usage) to support the analyzer's
Send-based parallel fan-out, and a missing or misconfigured reducer would
silently collapse fanned-out results down to a single paper's contribution
instead of raising -- exactly the kind of bug that ships undetected without
a test like the ones below.
"""
import os
import json
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch

import pytest

from agents.analyzer import AnalyzerAgent
from utils.schemas import Analysis, Paper, SynthesisResult, ValidatedOutput
from utils.langgraph_state import create_initial_state, merge_token_usage
from rag.retrieval import RAGRetriever
from orchestration.workflow_graph import (
    create_workflow_graph,
    run_workflow,
    ANALYZER_MAX_CONCURRENCY,
)
from orchestration.nodes import analyzer_paper_node


def make_paper(arxiv_id: str) -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title=f"Paper {arxiv_id}",
        authors=["Author A"],
        abstract="Abstract text",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        published=datetime(2024, 1, 1),
        categories=["cs.AI"],
    )


def make_analysis(paper_id: str, confidence: float = 0.8) -> Analysis:
    return Analysis(
        paper_id=paper_id,
        methodology="m",
        key_findings=["f"],
        conclusions="c",
        limitations=["l"],
        citations=["cit"],
        main_contributions=["mc"],
        confidence_score=confidence,
    )


# ---------------------------------------------------------------------------
# Fake agents implementing the same run(state) -> state contract as the real
# retriever/synthesis/citation agents (unaffected by this migration), so the
# full graph can be exercised end-to-end without hitting real Azure OpenAI.
# ---------------------------------------------------------------------------

class FakeRetrieverAgent:
    def __init__(self, papers):
        self.papers = papers

    def run(self, state):
        state["papers"] = self.papers
        state["chunks"] = []
        return state


class FakeSynthesisAgent:
    def __init__(self):
        self.call_count = 0

    def run(self, state):
        self.call_count += 1
        state["synthesis"] = SynthesisResult(
            consensus_points=[],
            contradictions=[],
            research_gaps=[],
            summary="ok",
            confidence_score=0.9,
            papers_analyzed=[a.paper_id for a in state.get("filtered_analyses", [])],
        )
        return state


class FakeCitationAgent:
    def __init__(self):
        self.call_count = 0

    def run(self, state):
        self.call_count += 1
        state["validated_output"] = ValidatedOutput(
            synthesis=state["synthesis"],
            citations=[],
            retrieved_chunks=[],
            token_usage=state.get("token_usage", {}),
            cost_estimate=0.0,
            processing_time=0.0,
        )
        return state


@pytest.fixture
def mock_analyzer_agent():
    """A MagicMock analyzer agent: analyze_paper() raises for 'bad.0001',
    succeeds for everything else. Used to test the graph's fan-out/fan-in
    wiring in isolation from the real Azure OpenAI call path (that path is
    covered separately by tests/test_analyzer.py and the circuit-breaker
    test below, which uses the real AnalyzerAgent)."""
    agent = MagicMock()
    agent.batch_tokens = {"input": 0, "output": 0}
    agent.consecutive_failures = 0

    def analyze_paper(paper, **kwargs):
        if paper.arxiv_id == "bad.0001":
            raise Exception("Simulated analyzer failure")
        return make_analysis(paper.arxiv_id)

    agent.analyze_paper.side_effect = analyze_paper
    return agent


def _run_graph(analyzer_agent, papers):
    retriever_agent = FakeRetrieverAgent(papers)
    synthesis_agent = FakeSynthesisAgent()
    citation_agent = FakeCitationAgent()

    app = create_workflow_graph(
        retriever_agent=retriever_agent,
        analyzer_agent=analyzer_agent,
        synthesis_agent=synthesis_agent,
        citation_agent=citation_agent,
        use_checkpointing=False,
    )

    initial_state = create_initial_state(
        query="test query",
        category=None,
        num_papers=len(papers),
        model_desc={"llm_model": "test", "embedding_model": "test"},
        start_time=0.0,
    )
    config = {"configurable": {"thread_id": "test-thread"}, "max_concurrency": ANALYZER_MAX_CONCURRENCY}
    result = app.invoke(initial_state, config=config)
    return result, synthesis_agent, citation_agent


class TestSendFanOutReducerCorrectness:
    """Guards against the main risk identified during the Send migration:
    a missing/misconfigured Annotated[..., operator.add] reducer would
    silently collapse N fanned-out analyses down to 1 (last-write-wins)
    instead of merging them."""

    def test_fan_out_produces_one_analysis_per_paper(self, mock_analyzer_agent):
        papers = [make_paper(f"good.000{i}") for i in range(4)]
        result, _, _ = _run_graph(mock_analyzer_agent, papers)

        assert len(result["analyses"]) == 4, (
            "Expected one merged Analysis per paper -- a length of 1 here "
            "would indicate the operator.add reducer on AgentState.analyses "
            "is missing or not being applied (last-write-wins collapse)."
        )
        assert {a.paper_id for a in result["analyses"]} == {p.arxiv_id for p in papers}

    def test_partial_failure_omits_paper_and_reports_error(self, mock_analyzer_agent):
        papers = [make_paper("good.0001"), make_paper("bad.0001"), make_paper("good.0002")]
        result, _, _ = _run_graph(mock_analyzer_agent, papers)

        # The failed paper contributes zero analyses (matches the pre-Send
        # behavior: a raised exception meant that paper was never added to
        # `analyses` at all, not added as a degraded confidence=0.0 entry).
        assert len(result["analyses"]) == 2
        assert {a.paper_id for a in result["analyses"]} == {"good.0001", "good.0002"}

        assert len(result["errors"]) == 1
        assert "bad.0001" in result["errors"][0]

        # filter_node should retain both successful analyses (confidence 0.8 > 0)
        assert len(result["filtered_analyses"]) == 2


class TestBarrierSemantics:
    """Guards against fan-out/fan-in wiring bugs: the node(s) downstream of
    the fanned-out analyzer must run exactly once per graph execution, after
    all Send branches complete -- not once per branch."""

    def test_downstream_nodes_run_exactly_once(self, mock_analyzer_agent):
        papers = [make_paper(f"good.000{i}") for i in range(5)]
        result, synthesis_agent, citation_agent = _run_graph(mock_analyzer_agent, papers)

        assert synthesis_agent.call_count == 1
        assert citation_agent.call_count == 1
        assert result["validated_output"] is not None


class TestTokenUsageReducer:
    def test_merge_token_usage_sums_matching_keys(self):
        a = {"input_tokens": 10, "output_tokens": 5}
        b = {"input_tokens": 3, "embedding_tokens": 7}
        merged = merge_token_usage(a, b)
        assert merged == {"input_tokens": 13, "output_tokens": 5, "embedding_tokens": 7}

    def test_filter_node_aggregates_analyzer_batch_tokens(self, mock_analyzer_agent):
        # should_continue_after_retriever resets batch_tokens to {0, 0} at the
        # start of each new batch (mirrors the old AnalyzerAgent.run()'s
        # per-batch reset), so the mock's analyze_paper must accumulate into
        # it itself to realistically simulate the real AnalyzerAgent's
        # lock-protected `self.batch_tokens["input"] += ...` behavior.
        def analyze_paper(paper, **kwargs):
            mock_analyzer_agent.batch_tokens["input"] += 42
            mock_analyzer_agent.batch_tokens["output"] += 17
            return make_analysis(paper.arxiv_id)

        mock_analyzer_agent.analyze_paper.side_effect = analyze_paper

        papers = [make_paper("good.0001")]
        result, _, _ = _run_graph(mock_analyzer_agent, papers)

        assert result["token_usage"]["input_tokens"] == 42
        assert result["token_usage"]["output_tokens"] == 17


class TestMaxConcurrencyConfig:
    def test_run_workflow_passes_max_concurrency(self):
        """max_concurrency must be explicitly set to preserve the old hard
        cap of 4 concurrent papers -- omitting it silently raises the
        ceiling to the executor's default (min(32, os.cpu_count()+4))."""
        mock_app = MagicMock()
        mock_app.invoke.return_value = {"errors": [], "papers": [], "user_id": None}

        run_workflow(app=mock_app, initial_state={"errors": [], "user_id": None}, thread_id="t")

        _, kwargs = mock_app.invoke.call_args
        assert kwargs["config"]["max_concurrency"] == 4 == ANALYZER_MAX_CONCURRENCY


class TestAnalyzerCircuitBreakerUnderNewDispatch:
    """Uses the real AnalyzerAgent (mocked Azure client only) to confirm the
    circuit breaker still trips correctly when analyzer_paper_node calls
    analyze_paper() one paper at a time, mirroring how Send dispatches it."""

    @pytest.fixture
    def failing_analyzer_agent(self):
        with patch.dict(os.environ, {
            "AZURE_OPENAI_API_KEY": "test_key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
            "AZURE_OPENAI_API_VERSION": "2024-02-01",
            "AZURE_OPENAI_DEPLOYMENT_NAME": "test-deployment",
        }):
            client = MagicMock()
            client.chat.completions.create.side_effect = Exception("Simulated API failure")
            with patch("agents.analyzer.AzureOpenAI", return_value=client):
                retriever = Mock(spec=RAGRetriever)
                retriever.retrieve.return_value = {"query": "q", "chunks": [], "chunk_ids": []}
                retriever.format_context.return_value = "context"
                agent = AnalyzerAgent(rag_retriever=retriever, model="test-deployment")
                agent.max_consecutive_failures = 2
                return agent

    def test_circuit_breaker_trips_after_consecutive_failures(self, failing_analyzer_agent):
        papers = [make_paper(f"210{i}.0000{i}") for i in range(3)]
        results = [analyzer_paper_node({"paper": p}, failing_analyzer_agent) for p in papers]

        # First two calls: the mocked LLM call raises, caught by
        # analyze_paper()'s own try/except -> degraded Analysis
        # (confidence_score=0.0), not an error -- matches existing
        # pre-Send behavior for ordinary per-call failures.
        for r in results[:2]:
            assert len(r["analyses"]) == 1
            assert r["analyses"][0].confidence_score == 0.0
            assert "errors" not in r

        # Third call: circuit breaker has now tripped (2 consecutive
        # failures) -- analyze_paper() raises *before* its own try/except,
        # caught by analyzer_paper_node instead: no analysis contributed,
        # error surfaced. Matches pre-Send behavior where a circuit-breaker
        # trip meant that paper was omitted from `analyses` entirely.
        assert results[2]["analyses"] == []
        assert "errors" in results[2]
        assert "circuit breaker" in results[2]["errors"][0].lower()


class TestNoPapersFound:
    def test_workflow_ends_when_no_papers_found(self, mock_analyzer_agent):
        result, synthesis_agent, citation_agent = _run_graph(mock_analyzer_agent, [])

        assert result.get("analyses", []) == []
        assert synthesis_agent.call_count == 0
        assert citation_agent.call_count == 0
