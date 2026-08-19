"""
Unit tests for observability/trace_reader.py.

Covers the v3 SDK fix: TraceReader must call client.api.trace.get/list and
client.api.observations.get_many (not the nonexistent v2-era
client.get_trace()/get_traces()/get_observations()), with the correct kwarg
names (from_start_time/to_start_time for observations, from_timestamp/
to_timestamp for traces), and must prefer the non-deprecated usage_details/
cost_details fields when extracting usage/cost.
"""
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from observability.trace_reader import TraceReader


def _make_trace(
    id="trace-1",
    name="research_workflow_run",
    user_id="user-1",
    session_id="session-1",
    total_cost=0.0118901,
    latency=12.5,
    usage_details=None,
    cost_details=None,
):
    trace = Mock()
    trace.id = id
    trace.name = name
    trace.user_id = user_id
    trace.session_id = session_id
    trace.timestamp = datetime.now(timezone.utc)
    trace.metadata = {}
    trace.input = None
    trace.output = None
    trace.total_cost = total_cost
    trace.latency = latency
    # Explicit None (not just omitted) so Mock's auto-attribute-vivification
    # doesn't make getattr(trace, "usage_details", None) return a truthy Mock.
    trace.usage_details = usage_details
    trace.cost_details = cost_details
    trace.usage = None
    return trace


def _make_observation(
    id="obs-1",
    trace_id="trace-1",
    name="analyzer_agent_run",
    obs_type="GENERATION",
    model="gpt-4o-mini-2024-07-18",
    usage_details=None,
    cost_details=None,
    calculated_total_cost=None,
    latency=1.5,
):
    obs = Mock()
    obs.id = id
    obs.trace_id = trace_id
    obs.name = name
    obs.type = obs_type
    obs.model = model
    obs.input = "prompt"
    obs.output = "completion"
    obs.metadata = {}
    obs.level = "DEFAULT"
    obs.start_time = datetime.now(timezone.utc)
    obs.end_time = datetime.now(timezone.utc)
    obs.latency = latency
    obs.usage_details = usage_details
    obs.cost_details = cost_details
    obs.calculated_total_cost = calculated_total_cost
    # Explicit None so the deprecated-field fallback path in
    # _extract_token_usage() doesn't pick up a truthy auto-vivified Mock.
    obs.usage = None
    return obs


def _reader_with_mock_client(mock_client):
    with patch("observability.trace_reader.is_langfuse_enabled", return_value=True), \
         patch("observability.trace_reader.get_langfuse_client", return_value=mock_client):
        return TraceReader()


class TestTraceReaderDisabled:
    def test_all_methods_degrade_gracefully_when_disabled(self):
        with patch("observability.trace_reader.is_langfuse_enabled", return_value=False):
            reader = TraceReader()

        assert reader.client is None
        assert reader.get_traces() == []
        assert reader.get_trace_by_id("trace-1") is None
        assert reader.filter_by_agent("analyzer_agent") == []
        assert reader.get_generations() == []


class TestGetTraces:
    def test_calls_api_trace_list_not_v2_method(self):
        mock_client = Mock(spec=["api"])
        mock_client.api = Mock(spec=["trace"])
        mock_client.api.trace = Mock(spec=["list", "get"])
        mock_client.api.trace.list.return_value = Mock(data=[_make_trace()])
        assert not hasattr(mock_client, "get_traces")

        reader = _reader_with_mock_client(mock_client)
        traces = reader.get_traces(limit=10, user_id="user-1", session_id="session-1")

        mock_client.api.trace.list.assert_called_once_with(
            limit=10, user_id="user-1", session_id="session-1"
        )
        assert len(traces) == 1
        assert traces[0].id == "trace-1"
        assert traces[0].total_cost == 0.0118901
        # Trace duration comes from the pre-computed `latency` (seconds) field,
        # not a start_time/end_time diff -- Trace objects have no such fields.
        assert traces[0].duration_ms == 12.5 * 1000

    def test_returns_empty_list_on_client_error(self):
        mock_client = Mock(spec=["api"])
        mock_client.api = Mock(spec=["trace"])
        mock_client.api.trace = Mock(spec=["list"])
        mock_client.api.trace.list.side_effect = Exception("boom")

        reader = _reader_with_mock_client(mock_client)
        assert reader.get_traces() == []


class TestGetTraceById:
    def test_calls_api_trace_get_not_v2_method(self):
        mock_client = Mock(spec=["api"])
        mock_client.api = Mock(spec=["trace"])
        mock_client.api.trace = Mock(spec=["get"])
        mock_client.api.trace.get.return_value = _make_trace(id="trace-42", total_cost=0.05)
        assert not hasattr(mock_client, "get_trace")

        reader = _reader_with_mock_client(mock_client)
        trace = reader.get_trace_by_id("trace-42")

        mock_client.api.trace.get.assert_called_once_with("trace-42")
        assert trace is not None
        assert trace.id == "trace-42"
        assert trace.total_cost == 0.05

    def test_returns_none_when_not_found(self):
        mock_client = Mock(spec=["api"])
        mock_client.api = Mock(spec=["trace"])
        mock_client.api.trace = Mock(spec=["get"])
        mock_client.api.trace.get.return_value = None

        reader = _reader_with_mock_client(mock_client)
        assert reader.get_trace_by_id("missing") is None


class TestFilterByAgent:
    def test_calls_observations_get_many_with_from_start_time(self):
        mock_client = Mock(spec=["api"])
        mock_client.api = Mock(spec=["observations"])
        mock_client.api.observations = Mock(spec=["get_many"])
        mock_client.api.observations.get_many.return_value = Mock(
            data=[_make_observation(obs_type="SPAN", name="analyzer_agent")]
        )
        assert not hasattr(mock_client, "get_observations")

        reader = _reader_with_mock_client(mock_client)
        from_date = datetime.now(timezone.utc)
        spans = reader.filter_by_agent("analyzer_agent", limit=25, from_timestamp=from_date)

        # observations.get_many's real kwarg is from_start_time, not from_timestamp.
        # request_options widens the SDK's default timeout -- an unscoped query by a
        # high-volume agent name can otherwise time out against real usage history.
        mock_client.api.observations.get_many.assert_called_once_with(
            request_options={"timeout_in_seconds": 30},
            limit=25, name="analyzer_agent", type="SPAN", from_start_time=from_date
        )
        assert len(spans) == 1
        assert spans[0].name == "analyzer_agent"

    def test_trace_id_scopes_the_query(self):
        # trace_id narrows the server-side query, which is what actually fixes the
        # timeout in practice -- verified live, see CLAUDE.md's Trace Querying section.
        mock_client = Mock(spec=["api"])
        mock_client.api = Mock(spec=["observations"])
        mock_client.api.observations = Mock(spec=["get_many"])
        mock_client.api.observations.get_many.return_value = Mock(
            data=[_make_observation(obs_type="SPAN", name="analyzer_agent")]
        )

        reader = _reader_with_mock_client(mock_client)
        spans = reader.filter_by_agent("analyzer_agent", trace_id="trace-123")

        mock_client.api.observations.get_many.assert_called_once_with(
            request_options={"timeout_in_seconds": 30},
            limit=50, name="analyzer_agent", type="SPAN", trace_id="trace-123"
        )
        assert len(spans) == 1


class TestGetGenerations:
    def test_calls_observations_get_many_with_generation_type(self):
        mock_client = Mock(spec=["api"])
        mock_client.api = Mock(spec=["observations"])
        mock_client.api.observations = Mock(spec=["get_many"])
        mock_client.api.observations.get_many.return_value = Mock(
            data=[_make_observation()]
        )

        reader = _reader_with_mock_client(mock_client)
        generations = reader.get_generations(trace_id="trace-1", name="analyzer_agent_run", limit=50)

        mock_client.api.observations.get_many.assert_called_once_with(
            limit=50, type="GENERATION", trace_id="trace-1", name="analyzer_agent_run"
        )
        assert len(generations) == 1
        assert generations[0].name == "analyzer_agent_run"

    def test_prefers_usage_details_and_cost_details_over_deprecated_fields(self):
        mock_client = Mock(spec=["api"])
        mock_client.api = Mock(spec=["observations"])
        mock_client.api.observations = Mock(spec=["get_many"])
        mock_client.api.observations.get_many.return_value = Mock(
            data=[
                _make_observation(
                    usage_details={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                    cost_details={"total": 0.0009},
                    calculated_total_cost=999.0,  # deliberately wrong/stale to prove it's ignored
                )
            ]
        )

        reader = _reader_with_mock_client(mock_client)
        generations = reader.get_generations()

        gen = generations[0]
        assert gen.usage == {"input": 100, "output": 50, "total": 150}
        assert gen.cost == 0.0009

    def test_falls_back_to_deprecated_fields_when_usage_details_absent(self):
        mock_client = Mock(spec=["api"])
        mock_client.api = Mock(spec=["observations"])
        mock_client.api.observations = Mock(spec=["get_many"])
        obs = _make_observation(usage_details=None, cost_details=None, calculated_total_cost=0.002)
        obs.usage = Mock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_client.api.observations.get_many.return_value = Mock(data=[obs])

        reader = _reader_with_mock_client(mock_client)
        generations = reader.get_generations()

        gen = generations[0]
        assert gen.usage == {"input": 10, "output": 5, "total": 15}
        assert gen.cost == 0.002

    def test_returns_empty_list_on_client_error(self):
        mock_client = Mock(spec=["api"])
        mock_client.api = Mock(spec=["observations"])
        mock_client.api.observations = Mock(spec=["get_many"])
        mock_client.api.observations.get_many.side_effect = Exception("boom")

        reader = _reader_with_mock_client(mock_client)
        assert reader.get_generations() == []
