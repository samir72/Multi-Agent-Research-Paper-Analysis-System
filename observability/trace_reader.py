"""
Trace reader for querying LangFuse observability data.

Provides Python API for programmatic access to traces, spans, and generations.
"""
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

from utils.langfuse_client import get_langfuse_client, is_langfuse_enabled

logger = logging.getLogger(__name__)


class TraceInfo(BaseModel):
    """Pydantic model for trace information."""
    id: str
    name: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)
    input: Optional[Any] = None
    output: Optional[Any] = None
    duration_ms: Optional[float] = None
    total_cost: Optional[float] = None
    token_usage: Dict[str, int] = Field(default_factory=dict)


class SpanInfo(BaseModel):
    """Pydantic model for span information."""
    id: str
    trace_id: str
    name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    input: Optional[Any] = None
    output: Optional[Any] = None
    level: str = "DEFAULT"


class GenerationInfo(BaseModel):
    """Pydantic model for LLM generation information."""
    id: str
    trace_id: str
    name: str
    model: Optional[str] = None
    # Optional[Any], not Optional[str]: this covers every GENERATION-type
    # observation, including the outer @observe(as_type="generation")-decorated
    # agent run() methods (e.g. "analyzer_agent_run"), whose captured
    # input/output is the raw state dict, not a plain text prompt/completion.
    prompt: Optional[Any] = None
    completion: Optional[Any] = None
    usage: Dict[str, int] = Field(default_factory=dict)
    cost: Optional[float] = None
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TraceReader:
    """
    Read and query LangFuse traces programmatically.

    Usage:
        reader = TraceReader()
        traces = reader.get_traces(limit=10)
        trace = reader.get_trace_by_id("trace-123")
        agent_traces = reader.filter_by_agent("retriever_agent")
    """

    def __init__(self):
        """Initialize trace reader with LangFuse client."""
        if not is_langfuse_enabled():
            logger.warning("LangFuse is not enabled. TraceReader will return empty results.")
            self.client = None
        else:
            self.client = get_langfuse_client()
            logger.info("TraceReader initialized with LangFuse client")

    def get_traces(
        self,
        limit: int = 50,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        from_timestamp: Optional[datetime] = None,
        to_timestamp: Optional[datetime] = None,
    ) -> List[TraceInfo]:
        """
        Get traces with optional filters.

        Args:
            limit: Maximum number of traces to return
            user_id: Filter by user ID
            session_id: Filter by session ID
            from_timestamp: Filter traces after this timestamp
            to_timestamp: Filter traces before this timestamp

        Returns:
            List of TraceInfo objects
        """
        if not self.client:
            logger.warning("LangFuse client not available")
            return []

        try:
            # Build filter params
            params = {"limit": limit}
            if user_id:
                params["user_id"] = user_id
            if session_id:
                params["session_id"] = session_id
            if from_timestamp:
                params["from_timestamp"] = from_timestamp
            if to_timestamp:
                params["to_timestamp"] = to_timestamp

            # Fetch traces from LangFuse (v3 SDK: query API lives under client.api.*,
            # not directly on the Langfuse client -- there is no client.get_traces())
            traces_data = self.client.api.trace.list(**params)

            # Convert to TraceInfo objects
            traces = []
            for trace in traces_data.data:
                trace_info = TraceInfo(
                    id=trace.id,
                    name=trace.name,
                    user_id=trace.user_id,
                    session_id=trace.session_id,
                    timestamp=trace.timestamp,
                    metadata=trace.metadata or {},
                    input=trace.input,
                    output=trace.output,
                    duration_ms=self._calculate_duration(trace),
                    total_cost=getattr(trace, "total_cost", None),
                    token_usage=self._extract_token_usage(trace),
                )
                traces.append(trace_info)

            logger.info(f"Retrieved {len(traces)} traces")
            return traces

        except Exception as e:
            logger.error(f"Error fetching traces: {e}")
            return []

    def get_trace_by_id(self, trace_id: str) -> Optional[TraceInfo]:
        """
        Get a specific trace by ID.

        Args:
            trace_id: Trace identifier

        Returns:
            TraceInfo object or None if not found
        """
        if not self.client:
            logger.warning("LangFuse client not available")
            return None

        try:
            trace = self.client.api.trace.get(trace_id)

            if not trace:
                logger.warning(f"Trace {trace_id} not found")
                return None

            trace_info = TraceInfo(
                id=trace.id,
                name=trace.name,
                user_id=trace.user_id,
                session_id=trace.session_id,
                timestamp=trace.timestamp,
                metadata=trace.metadata or {},
                input=trace.input,
                output=trace.output,
                duration_ms=self._calculate_duration(trace),
                total_cost=getattr(trace, "total_cost", None),
                token_usage=self._extract_token_usage(trace),
            )

            logger.info(f"Retrieved trace {trace_id}")
            return trace_info

        except Exception as e:
            logger.error(f"Error fetching trace {trace_id}: {e}")
            return None

    def filter_by_agent(
        self,
        agent_name: str,
        limit: int = 50,
        from_timestamp: Optional[datetime] = None,
        trace_id: Optional[str] = None,
    ) -> List[SpanInfo]:
        """
        Filter traces by agent name.

        Args:
            agent_name: Name of the agent (e.g., "retriever_agent", "analyzer_agent")
            limit: Maximum number of results
            from_timestamp: Filter traces after this timestamp
            trace_id: Optional trace ID to scope the query to a single trace.
                Strongly recommended when known -- without it, LangFuse scans
                by name across the *entire* project's history. Verified live:
                on this project's real usage history, unscoped queries for the
                two highest-volume span names ("analyzer_agent", "retriever_agent")
                reliably time out (~5s, reproduced twice) while low-volume names
                ("synthesis_agent", "citation_agent") return in 3-4s -- so add
                trace_id whenever the caller already has it (e.g. right after a
                workflow run), and treat the unscoped mode as a slow, best-effort
                "recent activity" query, not something to poll routinely.

        Returns:
            List of SpanInfo objects for the specified agent
        """
        if not self.client:
            logger.warning("LangFuse client not available")
            return []

        try:
            # Get observations filtered by name. Note: observations.get_many()'s
            # real kwarg is from_start_time, not from_timestamp (that's only
            # valid on trace.list()) -- the public param name here stays
            # from_timestamp for backward compatibility with existing callers.
            params = {"limit": limit, "name": agent_name, "type": "SPAN"}
            if from_timestamp:
                params["from_start_time"] = from_timestamp
            if trace_id:
                params["trace_id"] = trace_id

            # Widen past the SDK's default request timeout: verified live that
            # the default is too short for an unscoped scan over a high-volume
            # agent name in a project with real usage history (see trace_id
            # docstring above) -- 30s comfortably covers that case without
            # masking a genuinely dead connection.
            observations = self.client.api.observations.get_many(
                request_options={"timeout_in_seconds": 30},
                **params,
            )

            spans = []
            for obs in observations.data:
                span_info = SpanInfo(
                    id=obs.id,
                    trace_id=obs.trace_id,
                    name=obs.name,
                    start_time=obs.start_time,
                    end_time=obs.end_time,
                    duration_ms=self._calculate_duration(obs),
                    metadata=obs.metadata or {},
                    input=obs.input,
                    output=obs.output,
                    level=getattr(obs, "level", "DEFAULT"),
                )
                spans.append(span_info)

            logger.info(f"Retrieved {len(spans)} spans for agent '{agent_name}'")
            return spans

        except Exception as e:
            logger.error(f"Error filtering by agent {agent_name}: {e}")
            return []

    def filter_by_date_range(
        self,
        from_date: datetime,
        to_date: datetime,
        limit: int = 100,
    ) -> List[TraceInfo]:
        """
        Filter traces by date range.

        Args:
            from_date: Start date
            to_date: End date
            limit: Maximum number of traces

        Returns:
            List of TraceInfo objects within date range
        """
        return self.get_traces(
            limit=limit,
            from_timestamp=from_date,
            to_timestamp=to_date,
        )

    def get_generations(
        self,
        trace_id: Optional[str] = None,
        name: Optional[str] = None,
        from_timestamp: Optional[datetime] = None,
        limit: int = 50,
    ) -> List[GenerationInfo]:
        """
        Get LLM generations (optionally filtered by trace, name, and/or date).

        Args:
            trace_id: Optional trace ID to filter generations
            name: Optional generation name filter (e.g. "analyzer_agent_run")
            from_timestamp: Optional lower bound on generation start time
            limit: Maximum number of generations

        Returns:
            List of GenerationInfo objects
        """
        if not self.client:
            logger.warning("LangFuse client not available")
            return []

        try:
            params = {"limit": limit, "type": "GENERATION"}
            if trace_id:
                params["trace_id"] = trace_id
            if name:
                params["name"] = name
            if from_timestamp:
                params["from_start_time"] = from_timestamp

            observations = self.client.api.observations.get_many(**params)

            generations = []
            for obs in observations.data:
                gen_info = GenerationInfo(
                    id=obs.id,
                    trace_id=obs.trace_id,
                    name=obs.name,
                    model=getattr(obs, "model", None),
                    prompt=getattr(obs, "input", None),
                    completion=getattr(obs, "output", None),
                    usage=self._extract_token_usage(obs),
                    cost=self._extract_cost(obs),
                    start_time=obs.start_time,
                    end_time=obs.end_time,
                    duration_ms=self._calculate_duration(obs),
                    metadata=obs.metadata or {},
                )
                generations.append(gen_info)

            logger.info(f"Retrieved {len(generations)} generations")
            return generations

        except Exception as e:
            logger.error(f"Error fetching generations: {e}")
            return []

    def export_traces_to_json(
        self,
        traces: List[TraceInfo],
        output_file: str,
    ) -> bool:
        """
        Export traces to JSON file.

        Args:
            traces: List of TraceInfo objects
            output_file: Path to output JSON file

        Returns:
            True if successful, False otherwise
        """
        try:
            import json

            data = [trace.model_dump() for trace in traces]

            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(f"Exported {len(traces)} traces to {output_file}")
            return True

        except Exception as e:
            logger.error(f"Error exporting traces: {e}")
            return False

    def export_traces_to_csv(
        self,
        traces: List[TraceInfo],
        output_file: str,
    ) -> bool:
        """
        Export traces to CSV file.

        Args:
            traces: List of TraceInfo objects
            output_file: Path to output CSV file

        Returns:
            True if successful, False otherwise
        """
        try:
            import csv

            if not traces:
                logger.warning("No traces to export")
                return False

            # Define CSV columns
            fieldnames = [
                "id", "name", "user_id", "session_id", "timestamp",
                "duration_ms", "total_cost", "input_tokens", "output_tokens"
            ]

            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for trace in traces:
                    row = {
                        "id": trace.id,
                        "name": trace.name,
                        "user_id": trace.user_id or "",
                        "session_id": trace.session_id or "",
                        "timestamp": trace.timestamp.isoformat(),
                        "duration_ms": trace.duration_ms or 0,
                        "total_cost": trace.total_cost or 0,
                        "input_tokens": trace.token_usage.get("input", 0),
                        "output_tokens": trace.token_usage.get("output", 0),
                    }
                    writer.writerow(row)

            logger.info(f"Exported {len(traces)} traces to {output_file}")
            return True

        except Exception as e:
            logger.error(f"Error exporting traces to CSV: {e}")
            return False

    # Helper methods

    def _calculate_duration(self, obj: Any) -> Optional[float]:
        """
        Calculate duration in milliseconds.

        Traces (Trace/TraceWithDetails/TraceWithFullDetails) have no
        start_time/end_time -- only a pre-computed `latency` field (seconds).
        Observations (spans/generations) have both `latency` and real
        start_time/end_time; `latency` is preferred uniformly when present,
        falling back to a start/end diff for older data that lacks it.
        """
        try:
            latency = getattr(obj, "latency", None)
            if latency is not None:
                return latency * 1000
            if hasattr(obj, 'start_time') and hasattr(obj, 'end_time') and obj.end_time:
                return (obj.end_time - obj.start_time).total_seconds() * 1000
            return None
        except Exception:
            return None

    def _extract_token_usage(self, obj: Any) -> Dict[str, int]:
        """
        Extract token usage from an observation.

        Prefers the non-deprecated `usage_details` dict (key names vary by
        provider -- langfuse.openai's wrapper stores raw OpenAI usage keys
        like prompt_tokens/completion_tokens/total_tokens for chat calls, or
        just {"input": N} for embeddings), falling back to the deprecated
        `usage` object for older data that predates usage_details.
        """
        try:
            usage_details = getattr(obj, "usage_details", None)
            if usage_details:
                return {
                    "input": usage_details.get("input", usage_details.get("prompt_tokens", 0)) or 0,
                    "output": usage_details.get("output", usage_details.get("completion_tokens", 0)) or 0,
                    "total": usage_details.get("total", usage_details.get("total_tokens", 0)) or 0,
                }
        except Exception:
            pass

        usage = {}
        try:
            if hasattr(obj, 'usage') and obj.usage:
                usage["input"] = getattr(obj.usage, "prompt_tokens", 0) or getattr(obj.usage, "input", 0)
                usage["output"] = getattr(obj.usage, "completion_tokens", 0) or getattr(obj.usage, "output", 0)
                usage["total"] = getattr(obj.usage, "total_tokens", 0) or getattr(obj.usage, "total", 0)
        except Exception:
            pass
        return usage

    def _extract_cost(self, obj: Any) -> Optional[float]:
        """
        Extract total cost from an observation.

        Prefers the non-deprecated `cost_details` dict's "total" key, falling
        back to the deprecated `calculated_total_cost` field for older data.
        """
        try:
            cost_details = getattr(obj, "cost_details", None)
            if cost_details and "total" in cost_details:
                return cost_details["total"]
        except Exception:
            pass
        return getattr(obj, "calculated_total_cost", None)
