# Observability Module

This module provides comprehensive observability for the multi-agent RAG system using LangFuse tracing and analytics.

## Features

- **Trace Reading API**: Query and filter LangFuse traces programmatically
- **Performance Analytics**: Agent-level metrics including latency, token usage, and costs
- **Trajectory Analysis**: Analyze agent execution paths and workflow patterns
- **Export Capabilities**: Export traces to JSON/CSV for external analysis

## Quick Start

### 1. Configure LangFuse

Add your LangFuse credentials to `.env`:

```bash
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key-here
LANGFUSE_SECRET_KEY=sk-lf-your-secret-key-here
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 2. Run Your Workflow

The system automatically traces all agent executions, LLM calls, and RAG operations.

### 3. Query Traces

Use the Python API to read and analyze traces:

```python
from observability import TraceReader, AgentPerformanceAnalyzer

# Initialize trace reader
reader = TraceReader()

# Get recent traces
traces = reader.get_traces(limit=10)

# Get traces for a specific session
session_traces = reader.get_traces(session_id="session-abc123")

# Filter by agent -- pass trace_id when you have it to scope the query server-side;
# unscoped queries on high-volume agent names can be slow against real usage history
retriever_spans = reader.filter_by_agent("retriever_agent", trace_id="trace-xyz", limit=50)

# Get specific trace
trace = reader.get_trace_by_id("trace-xyz")
```

## Trace Reader API

### TraceReader

Query and retrieve traces from LangFuse.

```python
from observability import TraceReader
from datetime import datetime, timedelta

reader = TraceReader()

# Get traces with filters
traces = reader.get_traces(
    limit=50,
    user_id="user-123",
    session_id="session-abc",
    from_timestamp=datetime.now() - timedelta(days=7),
    to_timestamp=datetime.now()
)

# Filter by date range
recent_traces = reader.filter_by_date_range(
    from_date=datetime.now() - timedelta(days=1),
    to_date=datetime.now(),
    limit=100
)

# Get LLM generations
generations = reader.get_generations(trace_id="trace-xyz")

# Export to files
reader.export_traces_to_json(traces, "traces.json")
reader.export_traces_to_csv(traces, "traces.csv")
```

## Performance Analytics API

### AgentPerformanceAnalyzer

Analyze agent performance metrics.

```python
from observability import AgentPerformanceAnalyzer

analyzer = AgentPerformanceAnalyzer()

# Get latency statistics for an agent
stats = analyzer.agent_latency_stats("retriever_agent", days=7)
print(f"Average latency: {stats.avg_latency_ms:.2f}ms")
print(f"P95 latency: {stats.p95_latency_ms:.2f}ms")
print(f"Success rate: {stats.success_rate:.1f}%")

# Get token usage breakdown
token_usage = analyzer.token_usage_breakdown(days=7)
for agent, usage in token_usage.items():
    print(f"{agent}: {usage['total']:,} tokens")

# Get cost breakdown per agent
costs = analyzer.cost_per_agent(session_id="session-abc")
for agent, cost in costs.items():
    print(f"{agent}: ${cost:.4f}")

# Get error rates
error_stats = analyzer.error_rates(days=30)
for agent, stats in error_stats.items():
    print(f"{agent}: {stats['error_rate_percent']:.2f}% errors")

# Get workflow performance summary
workflow_stats = analyzer.workflow_performance_summary(days=7)
print(f"Total runs: {workflow_stats.total_runs}")
print(f"Average duration: {workflow_stats.avg_duration_ms:.2f}ms")
print(f"Total cost: ${workflow_stats.total_cost:.4f}")
```

## Trajectory Analysis API

### AgentTrajectoryAnalyzer

Analyze agent execution paths and workflow patterns.

```python
from observability import AgentTrajectoryAnalyzer

analyzer = AgentTrajectoryAnalyzer()

# Get agent trajectories
trajectories = analyzer.get_trajectories(session_id="session-abc", days=7)

for traj in trajectories:
    print(f"Trace: {traj.trace_id}")
    print(f"Duration: {traj.total_duration_ms:.2f}ms")
    print(f"Path: {' → '.join(traj.agent_sequence)}")
    print(f"Success: {traj.success}")

# Analyze execution paths
path_analysis = analyzer.analyze_execution_paths(days=7)
print(f"Total workflows: {path_analysis['total_workflows']}")
print(f"Unique paths: {path_analysis['unique_paths']}")
print(f"Most common path: {path_analysis['most_common_path']}")

# Compare two workflow executions
comparison = analyzer.compare_trajectories("trace-1", "trace-2")
print(f"Duration difference: {comparison['duration_diff_ms']:.2f}ms")
print(f"Same path: {comparison['same_path']}")
```

## Data Models

### TraceInfo

```python
class TraceInfo(BaseModel):
    id: str
    name: str
    user_id: Optional[str]
    session_id: Optional[str]
    timestamp: datetime
    metadata: Dict[str, Any]
    duration_ms: Optional[float]
    total_cost: Optional[float]
    token_usage: Dict[str, int]
```

### AgentStats

```python
class AgentStats(BaseModel):
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
```

### WorkflowStats

```python
class WorkflowStats(BaseModel):
    total_runs: int
    avg_duration_ms: float
    p50_duration_ms: float
    p95_duration_ms: float
    p99_duration_ms: float
    success_rate: float
    total_cost: float
    avg_cost_per_run: float
    total_tokens: int
```

### AgentTrajectory

```python
class AgentTrajectory(BaseModel):
    trace_id: str
    session_id: Optional[str]
    start_time: datetime
    total_duration_ms: float
    agent_sequence: List[str]
    agent_timings: Dict[str, float]
    agent_costs: Dict[str, float]
    errors: List[str]
    success: bool
```

## Example: Performance Dashboard Script

```python
#!/usr/bin/env python3
"""Generate performance dashboard from traces."""

from datetime import datetime, timedelta
from observability import AgentPerformanceAnalyzer, AgentTrajectoryAnalyzer

def main():
    perf = AgentPerformanceAnalyzer()
    traj = AgentTrajectoryAnalyzer()

    print("=" * 60)
    print("AGENT PERFORMANCE DASHBOARD - Last 7 Days")
    print("=" * 60)

    # Workflow summary
    workflow_stats = perf.workflow_performance_summary(days=7)
    if workflow_stats:
        print(f"\nWorkflow Summary:")
        print(f"  Total Runs: {workflow_stats.total_runs}")
        print(f"  Avg Duration: {workflow_stats.avg_duration_ms/1000:.2f}s")
        print(f"  P95 Duration: {workflow_stats.p95_duration_ms/1000:.2f}s")
        print(f"  Success Rate: {workflow_stats.success_rate:.1f}%")
        print(f"  Total Cost: ${workflow_stats.total_cost:.4f}")
        print(f"  Avg Cost/Run: ${workflow_stats.avg_cost_per_run:.4f}")

    # Agent latency stats
    print(f"\nAgent Latency Statistics:")
    for agent_name in ["retriever_agent", "analyzer_agent", "synthesis_agent"]:
        stats = perf.agent_latency_stats(agent_name, days=7)
        if stats:
            print(f"\n  {agent_name}:")
            print(f"    Executions: {stats.execution_count}")
            print(f"    Avg Latency: {stats.avg_latency_ms/1000:.2f}s")
            print(f"    P95 Latency: {stats.p95_latency_ms/1000:.2f}s")
            print(f"    Success Rate: {stats.success_rate:.1f}%")

    # Cost breakdown
    print(f"\nCost Breakdown:")
    costs = perf.cost_per_agent(days=7)
    for agent, cost in sorted(costs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {agent}: ${cost:.4f}")

    # Path analysis
    print(f"\nExecution Path Analysis:")
    path_analysis = traj.analyze_execution_paths(days=7)
    if path_analysis:
        print(f"  Total Workflows: {path_analysis['total_workflows']}")
        print(f"  Unique Paths: {path_analysis['unique_paths']}")
        if path_analysis['most_common_path']:
            path, count = path_analysis['most_common_path']
            print(f"  Most Common: {path} ({count} times)")

if __name__ == "__main__":
    main()
```

Save as `scripts/performance_dashboard.py` and run:

```bash
python scripts/performance_dashboard.py
```

## Advanced Usage

### Custom Metrics

```python
from observability import TraceReader

reader = TraceReader()

# Calculate custom metric: papers processed per second
traces = reader.get_traces(limit=100)
total_papers = 0
total_time_ms = 0

for trace in traces:
    if trace.metadata.get("num_papers"):
        total_papers += trace.metadata["num_papers"]
        total_time_ms += trace.duration_ms or 0

if total_time_ms > 0:
    papers_per_second = (total_papers / total_time_ms) * 1000
    print(f"Papers/second: {papers_per_second:.2f}")
```

### Monitoring Alerts

```python
from observability import AgentPerformanceAnalyzer

analyzer = AgentPerformanceAnalyzer()

# Check if error rate exceeds threshold
error_stats = analyzer.error_rates(days=1)
for agent, stats in error_stats.items():
    if stats['error_rate_percent'] > 10:
        print(f"⚠️  ALERT: {agent} error rate is {stats['error_rate_percent']:.1f}%")

# Check if P95 latency is too high
stats = analyzer.agent_latency_stats("analyzer_agent", days=1)
if stats and stats.p95_latency_ms > 30000:  # 30 seconds
    print(f"⚠️  ALERT: Analyzer P95 latency is {stats.p95_latency_ms/1000:.1f}s")
```

## Troubleshooting

### No Traces Found

1. Check that LangFuse is enabled: `LANGFUSE_ENABLED=true`
2. Verify API keys are correct in `.env`
3. Ensure network connectivity to LangFuse Cloud
4. Check that at least one workflow has been executed

### Missing Token/Cost Data

- Token usage requires `langfuse-openai` instrumentation
- Ensure `instrument_openai()` is called before creating Azure OpenAI clients
- Cost data depends on LangFuse pricing configuration

### Slow Query Performance

- Reduce `limit` parameter for large trace datasets
- Use date range filters to narrow results
- Consider exporting traces to CSV for offline analysis

## See Also

- [LangFuse Documentation](https://langfuse.com/docs)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- Main README: `../README.md`
- Architecture: `../CLAUDE.md`
