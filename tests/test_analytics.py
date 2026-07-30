"""
Unit tests for observability/analytics.py::get_verified_trace_cost().

Covers the retry/poll semantics: LangFuse computes cost server-side,
asynchronously, after ingestion, so a freshly-flushed trace may briefly
report $0.00 before the real cost is available.
"""
from unittest.mock import Mock

import pytest

from observability.analytics import get_verified_trace_cost


def _trace_reader_returning(costs):
    """Mock TraceReader whose get_trace_by_id() yields total_cost from `costs` in order."""
    reader = Mock()
    trace_sequence = [Mock(total_cost=cost) if cost is not None else None for cost in costs]
    reader.get_trace_by_id.side_effect = trace_sequence
    return reader


class TestGetVerifiedTraceCost:
    def test_returns_immediately_on_nonzero_cost(self):
        reader = _trace_reader_returning([0.0118901])
        cost = get_verified_trace_cost("trace-1", max_attempts=5, delay_seconds=0, trace_reader=reader)
        assert cost == 0.0118901
        assert reader.get_trace_by_id.call_count == 1

    def test_retries_while_cost_is_zero_then_succeeds(self):
        reader = _trace_reader_returning([0.0, 0.0, 0.004])
        cost = get_verified_trace_cost("trace-1", max_attempts=5, delay_seconds=0, trace_reader=reader)
        assert cost == 0.004
        assert reader.get_trace_by_id.call_count == 3

    def test_returns_zero_after_exhausting_attempts(self):
        reader = _trace_reader_returning([0.0, 0.0, 0.0])
        cost = get_verified_trace_cost("trace-1", max_attempts=3, delay_seconds=0, trace_reader=reader)
        assert cost == 0.0
        assert reader.get_trace_by_id.call_count == 3

    def test_returns_none_when_trace_never_found(self):
        reader = _trace_reader_returning([None, None])
        cost = get_verified_trace_cost("trace-1", max_attempts=2, delay_seconds=0, trace_reader=reader)
        assert cost is None
        assert reader.get_trace_by_id.call_count == 2

    def test_never_raises_on_lookup_exception(self):
        reader = Mock()
        reader.get_trace_by_id.side_effect = Exception("network error")
        cost = get_verified_trace_cost("trace-1", max_attempts=2, delay_seconds=0, trace_reader=reader)
        assert cost is None
        assert reader.get_trace_by_id.call_count == 2
