"""
Performance analytics for agent execution and trajectory analysis.

Provides comprehensive metrics, statistics, and visualizations for observability data.
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

from pydantic import BaseModel, Field
from observability.trace_reader import TraceReader, TraceInfo, SpanInfo, GenerationInfo

logger = logging.getLogger(__name__)


class AgentStats(BaseModel):
    """Statistics for a single agent."""
    agent_name: str
    execution_count: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    success_rate: float
    total_cost: float
    avg_input_tokens: float
    avg_output_tokens: float


class WorkflowStats(BaseModel):
    """Statistics for entire workflow execution."""
    total_runs: int
    avg_duration_ms: float
    p50_duration_ms: float
    p95_duration_ms: float
    p99_duration_ms: float
    success_rate: float
    total_cost: float
    avg_cost_per_run: float
    total_tokens: int
    avg_tokens_per_run: float


class AgentTrajectory(BaseModel):
    """Trajectory of agent execution within a workflow."""
    trace_id: str
    session_id: Optional[str]
    start_time: datetime
    total_duration_ms: float
    agent_sequence: List[str] = Field(default_factory=list)
    agent_timings: Dict[str, float] = Field(default_factory=dict)
    agent_costs: Dict[str, float] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    success: bool = True


class AgentPerformanceAnalyzer:
    """
    Analyze agent performance metrics from LangFuse traces.

    Usage:
        analyzer = AgentPerformanceAnalyzer()
        stats = analyzer.agent_latency_stats("retriever_agent", days=7)
        cost_breakdown = analyzer.cost_per_agent(session_id="session-123")
        error_rates = analyzer.error_rates(days=30)
    """

    def __init__(self, trace_reader: Optional[TraceReader] = None):
        """
        Initialize performance analyzer.

        Args:
            trace_reader: Optional TraceReader instance (creates new if None)
        """
        self.trace_reader = trace_reader or TraceReader()
        logger.info("AgentPerformanceAnalyzer initialized")

    def agent_latency_stats(
        self,
        agent_name: str,
        days: int = 7,
        limit: int = 1000,
    ) -> Optional[AgentStats]:
        """
        Calculate latency statistics for a specific agent.

        Args:
            agent_name: Name of the agent
            days: Number of days to analyze
            limit: Maximum number of spans to analyze

        Returns:
            AgentStats object or None if no data
        """
        from_date = datetime.now() - timedelta(days=days)

        spans = self.trace_reader.filter_by_agent(
            agent_name=agent_name,
            limit=limit,
            from_timestamp=from_date,
        )

        if not spans:
            logger.warning(f"No data found for agent '{agent_name}'")
            return None

        # Extract latencies
        latencies = [s.duration_ms for s in spans if s.duration_ms is not None]

        if not latencies:
            logger.warning(f"No latency data for agent '{agent_name}'")
            return None

        # Calculate percentiles
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)

        stats = AgentStats(
            agent_name=agent_name,
            execution_count=len(spans),
            avg_latency_ms=statistics.mean(latencies),
            p50_latency_ms=latencies_sorted[int(n * 0.50)] if n > 0 else 0,
            p95_latency_ms=latencies_sorted[int(n * 0.95)] if n > 1 else 0,
            p99_latency_ms=latencies_sorted[int(n * 0.99)] if n > 1 else 0,
            min_latency_ms=min(latencies),
            max_latency_ms=max(latencies),
            success_rate=self._calculate_success_rate(spans),
            total_cost=0.0,  # Cost tracking requires generation data
            avg_input_tokens=0.0,
            avg_output_tokens=0.0,
        )

        logger.info(f"Calculated stats for '{agent_name}': avg={stats.avg_latency_ms:.2f}ms, "
                   f"p95={stats.p95_latency_ms:.2f}ms")
        return stats

    def token_usage_breakdown(
        self,
        session_id: Optional[str] = None,
        days: int = 7,
        limit: int = 100,
    ) -> Dict[str, Dict[str, int]]:
        """
        Get token usage breakdown by agent.

        Args:
            session_id: Optional session ID filter
            days: Number of days to analyze
            limit: Maximum number of traces

        Returns:
            Dictionary mapping agent names to token usage
        """
        from_date = datetime.now() - timedelta(days=days)

        traces = self.trace_reader.get_traces(
            limit=limit,
            session_id=session_id,
            from_timestamp=from_date,
        )

        if not traces:
            logger.warning("No traces found for token usage analysis")
            return {}

        # Aggregate token usage
        usage_by_agent = defaultdict(lambda: {"input": 0, "output": 0, "total": 0})

        for trace in traces:
            # Get generations for this trace
            generations = self.trace_reader.get_generations(trace_id=trace.id)

            for gen in generations:
                agent_name = gen.name
                usage_by_agent[agent_name]["input"] += gen.usage.get("input", 0)
                usage_by_agent[agent_name]["output"] += gen.usage.get("output", 0)
                usage_by_agent[agent_name]["total"] += gen.usage.get("total", 0)

        logger.info(f"Token usage breakdown calculated for {len(usage_by_agent)} agents")
        return dict(usage_by_agent)

    def cost_per_agent(
        self,
        session_id: Optional[str] = None,
        days: int = 7,
        limit: int = 100,
    ) -> Dict[str, float]:
        """
        Calculate cost breakdown per agent.

        Args:
            session_id: Optional session ID filter
            days: Number of days to analyze
            limit: Maximum number of traces

        Returns:
            Dictionary mapping agent names to total cost
        """
        from_date = datetime.now() - timedelta(days=days)

        traces = self.trace_reader.get_traces(
            limit=limit,
            session_id=session_id,
            from_timestamp=from_date,
        )

        if not traces:
            logger.warning("No traces found for cost analysis")
            return {}

        # Aggregate costs
        cost_by_agent = defaultdict(float)

        for trace in traces:
            generations = self.trace_reader.get_generations(trace_id=trace.id)

            for gen in generations:
                agent_name = gen.name
                cost = gen.cost or 0.0
                cost_by_agent[agent_name] += cost

        logger.info(f"Cost breakdown calculated for {len(cost_by_agent)} agents")
        return dict(cost_by_agent)

    def error_rates(
        self,
        days: int = 7,
        limit: int = 200,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate error rates per agent.

        Args:
            days: Number of days to analyze
            limit: Maximum number of spans per agent

        Returns:
            Dictionary with error rates and counts per agent
        """
        from_date = datetime.now() - timedelta(days=days)

        agent_names = [
            "retriever_agent",
            "analyzer_agent",
            "synthesis_agent",
            "citation_agent",
        ]

        error_stats = {}

        for agent_name in agent_names:
            spans = self.trace_reader.filter_by_agent(
                agent_name=agent_name,
                limit=limit,
                from_timestamp=from_date,
            )

            if not spans:
                continue

            total = len(spans)
            errors = sum(1 for s in spans if s.level == "ERROR" or "error" in s.metadata)
            error_rate = (errors / total) * 100 if total > 0 else 0

            error_stats[agent_name] = {
                "total_executions": total,
                "errors": errors,
                "error_rate_percent": error_rate,
                "success_rate_percent": 100 - error_rate,
            }

        logger.info(f"Error rates calculated for {len(error_stats)} agents")
        return error_stats

    def workflow_performance_summary(
        self,
        days: int = 7,
        limit: int = 100,
    ) -> Optional[WorkflowStats]:
        """
        Generate workflow-level performance summary.

        Args:
            days: Number of days to analyze
            limit: Maximum number of workflow runs

        Returns:
            WorkflowStats object or None if no data
        """
        from_date = datetime.now() - timedelta(days=days)

        traces = self.trace_reader.get_traces(
            limit=limit,
            from_timestamp=from_date,
        )

        if not traces:
            logger.warning("No workflow traces found")
            return None

        # Calculate statistics
        durations = [t.duration_ms for t in traces if t.duration_ms is not None]
        costs = [t.total_cost for t in traces if t.total_cost is not None]
        total_tokens = sum(t.token_usage.get("total", 0) for t in traces)

        if not durations:
            logger.warning("No duration data for workflows")
            return None

        durations_sorted = sorted(durations)
        n = len(durations_sorted)

        stats = WorkflowStats(
            total_runs=len(traces),
            avg_duration_ms=statistics.mean(durations),
            p50_duration_ms=durations_sorted[int(n * 0.50)] if n > 0 else 0,
            p95_duration_ms=durations_sorted[int(n * 0.95)] if n > 1 else 0,
            p99_duration_ms=durations_sorted[int(n * 0.99)] if n > 1 else 0,
            success_rate=self._calculate_trace_success_rate(traces),
            total_cost=sum(costs) if costs else 0.0,
            avg_cost_per_run=statistics.mean(costs) if costs else 0.0,
            total_tokens=total_tokens,
            avg_tokens_per_run=total_tokens / len(traces) if traces else 0,
        )

        logger.info(f"Workflow summary: {stats.total_runs} runs, "
                   f"avg={stats.avg_duration_ms:.2f}ms, cost=${stats.total_cost:.4f}")
        return stats

    def _calculate_success_rate(self, spans: List[SpanInfo]) -> float:
        """Calculate success rate from spans."""
        if not spans:
            return 0.0

        successes = sum(1 for s in spans if s.level != "ERROR" and "error" not in s.metadata)
        return (successes / len(spans)) * 100

    def _calculate_trace_success_rate(self, traces: List[TraceInfo]) -> float:
        """Calculate success rate from traces."""
        if not traces:
            return 0.0

        successes = sum(1 for t in traces if not t.metadata.get("error"))
        return (successes / len(traces)) * 100


class AgentTrajectoryAnalyzer:
    """
    Analyze agent execution trajectories and workflow paths.

    Usage:
        analyzer = AgentTrajectoryAnalyzer()
        trajectories = analyzer.get_trajectories(session_id="session-123")
        path_analysis = analyzer.analyze_execution_paths(days=7)
    """

    def __init__(self, trace_reader: Optional[TraceReader] = None):
        """
        Initialize trajectory analyzer.

        Args:
            trace_reader: Optional TraceReader instance
        """
        self.trace_reader = trace_reader or TraceReader()
        logger.info("AgentTrajectoryAnalyzer initialized")

    def get_trajectories(
        self,
        session_id: Optional[str] = None,
        days: int = 7,
        limit: int = 50,
    ) -> List[AgentTrajectory]:
        """
        Get agent execution trajectories for workflows.

        Args:
            session_id: Optional session ID filter
            days: Number of days to analyze
            limit: Maximum number of workflows

        Returns:
            List of AgentTrajectory objects
        """
        from_date = datetime.now() - timedelta(days=days)

        traces = self.trace_reader.get_traces(
            limit=limit,
            session_id=session_id,
            from_timestamp=from_date,
        )

        trajectories = []

        for trace in traces:
            trajectory = self._build_trajectory(trace)
            trajectories.append(trajectory)

        logger.info(f"Retrieved {len(trajectories)} agent trajectories")
        return trajectories

    def analyze_execution_paths(
        self,
        days: int = 7,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """
        Analyze common execution paths and patterns.

        Args:
            days: Number of days to analyze
            limit: Maximum number of workflows

        Returns:
            Dictionary with path analysis
        """
        trajectories = self.get_trajectories(days=days, limit=limit)

        if not trajectories:
            logger.warning("No trajectories found for path analysis")
            return {}

        # Analyze paths
        path_counts = defaultdict(int)
        for trajectory in trajectories:
            path = " → ".join(trajectory.agent_sequence)
            path_counts[path] += 1

        # Sort by frequency
        sorted_paths = sorted(path_counts.items(), key=lambda x: x[1], reverse=True)

        analysis = {
            "total_workflows": len(trajectories),
            "unique_paths": len(path_counts),
            "most_common_path": sorted_paths[0] if sorted_paths else None,
            "path_distribution": dict(sorted_paths[:10]),  # Top 10 paths
            "avg_agents_per_workflow": statistics.mean([len(t.agent_sequence) for t in trajectories]),
        }

        logger.info(f"Path analysis: {analysis['unique_paths']} unique paths from {analysis['total_workflows']} workflows")
        return analysis

    def compare_trajectories(
        self,
        trace_id_1: str,
        trace_id_2: str,
    ) -> Dict[str, Any]:
        """
        Compare two workflow trajectories.

        Args:
            trace_id_1: First trace ID
            trace_id_2: Second trace ID

        Returns:
            Comparison dictionary
        """
        trace1 = self.trace_reader.get_trace_by_id(trace_id_1)
        trace2 = self.trace_reader.get_trace_by_id(trace_id_2)

        if not trace1 or not trace2:
            logger.error("One or both traces not found")
            return {}

        traj1 = self._build_trajectory(trace1)
        traj2 = self._build_trajectory(trace2)

        comparison = {
            "trace_1": {
                "id": trace_id_1,
                "duration_ms": traj1.total_duration_ms,
                "agents": traj1.agent_sequence,
                "success": traj1.success,
            },
            "trace_2": {
                "id": trace_id_2,
                "duration_ms": traj2.total_duration_ms,
                "agents": traj2.agent_sequence,
                "success": traj2.success,
            },
            "duration_diff_ms": traj2.total_duration_ms - traj1.total_duration_ms,
            "duration_diff_percent": ((traj2.total_duration_ms - traj1.total_duration_ms) / traj1.total_duration_ms) * 100 if traj1.total_duration_ms > 0 else 0,
            "same_path": traj1.agent_sequence == traj2.agent_sequence,
        }

        logger.info(f"Compared trajectories: {trace_id_1} vs {trace_id_2}")
        return comparison

    def _build_trajectory(self, trace: TraceInfo) -> AgentTrajectory:
        """Build agent trajectory from trace."""
        # Get all spans for this trace (representing agent executions)
        # For now, construct from available trace data
        trajectory = AgentTrajectory(
            trace_id=trace.id,
            session_id=trace.session_id,
            start_time=trace.timestamp,
            total_duration_ms=trace.duration_ms or 0.0,
            agent_sequence=[],
            agent_timings={},
            agent_costs={},
            errors=[],
            success=not trace.metadata.get("error"),
        )

        # In a real implementation, we would fetch all spans for this trace
        # and build the sequence. For now, use a simplified version.
        if trace.output:
            trajectory.success = True

        return trajectory
