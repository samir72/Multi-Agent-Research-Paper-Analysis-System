# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core Architecture

This is a **multi-agent RAG system** for analyzing academic papers from arXiv. The system uses **LangGraph** for workflow orchestration and **LangFuse** for comprehensive observability.

### Agent Pipeline Flow

```
User Query → Retriever → Analyzer → Filter → Synthesis → Citation → Output
                ↓          ↓         ↓         ↓           ↓
            [LangFuse Tracing for All Nodes]
```

**Orchestration**: The workflow is managed by LangGraph (`orchestration/workflow_graph.py`):
- Conditional routing (early termination if no papers found or all analyses fail)
- Automatic checkpointing with `MemorySaver`
- State management with type-safe `AgentState` TypedDict
- Node wrappers in `orchestration/nodes.py` with automatic tracing

**State Dictionary** (`utils/langgraph_state.py`): All agents operate on a shared state dictionary that flows through the pipeline:
- `query`: User's research question
- `category`: Optional arXiv category filter
- `num_papers`: Number of papers to analyze
- `papers`: List of Paper objects (populated by Retriever)
- `chunks`: List of PaperChunk objects (populated by Retriever)
- `analyses`: List of Analysis objects (populated by Analyzer)
- `synthesis`: SynthesisResult object (populated by Synthesis)
- `validated_output`: ValidatedOutput object (populated by Citation)
- `errors`: List of error messages accumulated across agents
- `token_usage`: Dict tracking input/output/embedding tokens
- `trace_id`: LangFuse trace identifier (for observability). Set on `initial_state` *before* `app.invoke()` runs (`orchestration/workflow_graph.py::run_workflow()`), so it is present in `state` for every node from `retriever` through `finalize` — not just stamped on afterward. Pre-generated deterministically via `Langfuse.create_trace_id(seed=session_id)` and forced onto the root span via `trace_context` (see `utils/langfuse_client.py::workflow_trace()`).
- `session_id`: User session tracking (Gradio per-query id, generated in `app.py`)
- `user_id`: Optional user identifier — populated from the Gradio request's `session_hash` (`request.session_hash` in `app.py`'s handler), giving an anonymous-but-stable per-browser-tab id since the app has no auth

**IMPORTANT**: Only msgpack-serializable data should be stored in the state. Do NOT add complex objects like Gradio Progress, file handles, or callbacks to the state dictionary (see BUGFIX_MSGPACK_SERIALIZATION.md).

### Agent Responsibilities

1. **RetrieverAgent** (`agents/retriever.py`):
   - Decorated with `@observe` for LangFuse tracing
   - Searches arXiv API using `ArxivClient`, `MCPArxivClient`, or `FastMCPArxivClient` (configurable via env)
   - Downloads PDFs to `data/papers/` (direct API) or MCP server storage (MCP mode)
   - **Intelligent Fallback**: Automatically falls back to direct API if primary MCP client fails
   - Processes PDFs with `PDFProcessor` (500-token chunks, 50-token overlap)
   - Generates embeddings via `EmbeddingGenerator` (Azure OpenAI text-embedding-3-small, traced)
   - Stores chunks in ChromaDB via `VectorStore`
   - **FastMCP Support**: Auto-start FastMCP server for standardized arXiv access

2. **AnalyzerAgent** (`agents/analyzer.py`):
   - Decorated with `@observe(as_type="generation")` for LLM call tracing
   - Analyzes each paper individually using RAG
   - Uses 4 broad queries per paper: methodology, results, conclusions, limitations
   - Deduplicates chunks by chunk_id
   - Calls Azure OpenAI with **temperature=0** and JSON mode
   - RAG retrieval automatically traced via `@observe` on `RAGRetriever.retrieve()`
   - Returns structured `Analysis` objects with confidence scores

3. **SynthesisAgent** (`agents/synthesis.py`):
   - Decorated with `@observe(as_type="generation")` for LLM call tracing
   - Compares findings across all papers
   - Identifies consensus points, contradictions, research gaps
   - Creates executive summary addressing user's query
   - Uses **temperature=0** for deterministic outputs
   - Returns `SynthesisResult` with confidence scores

4. **CitationAgent** (`agents/citation.py`):
   - Decorated with `@observe(as_type="span")` for data processing tracing
   - Generates APA-formatted citations for all papers
   - Validates synthesis claims against source papers
   - Calculates cost estimates (GPT-4o-mini pricing)
   - Creates final `ValidatedOutput` with all metadata

### Critical Architecture Patterns

**RAG Context Formatting**: `RAGRetriever.format_context()` creates structured context with:
```
[Chunk N] Paper: {title}
Authors: {authors}
Section: {section}
Page: {page_number}
Source: {arxiv_url}
--------------------------------------------------------------------------------
{content}
```

**Chunking Strategy**: PDFProcessor uses tiktoken encoding (cl100k_base) for precise token counting:
- Chunk size: 500 tokens
- Overlap: 50 tokens
- Page markers preserved: `[Page N]` tags in text
- Section detection via keyword matching (abstract, introduction, results, etc.)

**Vector Store Filtering**: ChromaDB searches support paper_id filtering:
- Single paper: `{"paper_id": "2401.00001"}`
- Multiple papers: `{"paper_id": {"$in": ["2401.00001", "2401.00002"]}}`

**Semantic Caching**: Cache hits when cosine similarity ≥ 0.95 between query embeddings. Cache key includes both query and category.

**Error Handling Philosophy**: Agents catch exceptions, log errors, append to `state["errors"]`, and return partial results rather than failing completely. For example, Analyzer returns confidence_score=0.0 on failure.

### LangGraph Orchestration (`orchestration/`)

**Workflow Graph** (`orchestration/workflow_graph.py`):
- `create_workflow_graph()`: Creates StateGraph with all nodes and conditional edges
- `run_workflow()`: Sync wrapper for Gradio compatibility (uses `nest-asyncio`)
- `run_workflow_async()`: Async streaming execution
- `get_workflow_state()`: Retrieve current state by thread ID

**Node Wrappers** (`orchestration/nodes.py`):
- `retriever_node()`: Executes RetrieverAgent with LangFuse tracing
- `analyzer_node()`: Executes AnalyzerAgent with LangFuse tracing
- `filter_node()`: Filters out low-confidence analyses (confidence_score < 0.7)
- `synthesis_node()`: Executes SynthesisAgent with LangFuse tracing
- `citation_node()`: Executes CitationAgent with LangFuse tracing

**Conditional Routing**:
- `should_continue_after_retriever()`: Returns "END" if no papers found, else "analyzer"
- `should_continue_after_filter()`: Returns "END" if all analyses filtered out, else "synthesis"

**Workflow Execution Flow**:
```python
# In app.py
workflow_app = create_workflow_graph(
    retriever_agent=self.retriever_agent,
    analyzer_agent=self.analyzer_agent,
    synthesis_agent=self.synthesis_agent,
    citation_agent=self.citation_agent
)

# Run workflow with checkpointing
config = {"configurable": {"thread_id": session_id}}
final_state = run_workflow(workflow_app, initial_state, config, progress)
```

**State Serialization**:
- LangGraph uses msgpack for state checkpointing
- **CRITICAL**: Only msgpack-serializable types allowed in state
- ✅ Primitives: str, int, float, bool, None
- ✅ Collections: list, dict
- ✅ Pydantic models (via `.model_dump()`)
- ❌ Complex objects: Gradio Progress, file handles, callbacks
- See BUGFIX_MSGPACK_SERIALIZATION.md for detailed fix documentation

## Development Commands

### Running the Application
```bash
# Start Gradio interface (http://localhost:7860)
python app.py
```

### Testing
```bash
# Run all tests with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_analyzer.py -v

# Run single test
pytest tests/test_analyzer.py::TestAnalyzerAgent::test_analyze_paper_success -v

# Run with coverage
pytest tests/ --cov=agents --cov=rag --cov=utils -v

# Run tests matching pattern
pytest tests/ -k "analyzer" -v
```

### Environment Setup
```bash
# Copy environment template
cp .env.example .env

# Required variables in .env:
# AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
# AZURE_OPENAI_API_KEY=your-key
# AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini
# AZURE_OPENAI_API_VERSION=2024-02-01  # optional

# Optional MCP (Model Context Protocol) variables:
# USE_MCP_ARXIV=false              # Set to 'true' to use MCP (FastMCP by default)
# USE_LEGACY_MCP=false              # Set to 'true' to use legacy MCP instead of FastMCP
# MCP_ARXIV_STORAGE_PATH=./data/mcp_papers/  # MCP server storage path
# FASTMCP_SERVER_PORT=5555          # Port for FastMCP server (auto-started)

# Optional LangFuse observability variables:
# LANGFUSE_ENABLED=true            # Enable LangFuse tracing
# LANGFUSE_PUBLIC_KEY=pk-lf-...    # LangFuse public key
# LANGFUSE_SECRET_KEY=sk-lf-...    # LangFuse secret key
# LANGFUSE_BASE_URL=https://us.cloud.langfuse.com  # LangFuse host (cloud region or self-hosted; renamed from LANGFUSE_HOST in v2.10)
# LANGFUSE_TRACE_ALL_LLM=true      # Auto-trace all Azure OpenAI calls
# LANGFUSE_TRACE_RAG=true          # Trace RAG operations
# LANGFUSE_FLUSH_AT=15             # Batch size for flushing traces
# LANGFUSE_FLUSH_INTERVAL=10       # Flush interval in seconds
```

### Data Management
```bash
# Clear vector store (useful for testing)
rm -rf data/chroma_db/

# Clear cached papers
rm -rf data/papers/

# Clear semantic cache
rm -rf data/cache/
```

## Key Implementation Details

### Azure OpenAI Integration

All agents use **temperature=0** and **response_format={"type": "json_object"}** for deterministic, structured outputs. Initialize clients like:

```python
from openai import AzureOpenAI
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)
```

### Pydantic Schemas (`utils/schemas.py` and `utils/langgraph_state.py`)

All data structures use Pydantic for validation:
- `Paper`: arXiv paper metadata
- `PaperChunk`: Text chunk with metadata
- `Analysis`: Individual paper analysis results
- `SynthesisResult`: Cross-paper synthesis with ConsensusPoint and Contradiction
- `ValidatedOutput`: Final output with citations and cost tracking
- `AgentState`: TypedDict for LangGraph state management (used in workflow orchestration)

**Observability Models** (`observability/trace_reader.py`):
- `TraceInfo`: Trace metadata and performance metrics
- `SpanInfo`: Agent execution data with timings
- `GenerationInfo`: LLM call details (prompt, completion, tokens, cost)

**Analytics Models** (`observability/analytics.py`):
- `AgentStats`: Per-agent performance statistics (latency, tokens, cost, errors)
- `WorkflowStats`: Workflow-level aggregated metrics
- `AgentTrajectory`: Complete execution path with timings

### Retry Logic

ArxivClient uses tenacity for resilient API calls:
- 3 retry attempts
- Exponential backoff (4s min, 10s max)
- Applied to search_papers() and download_paper()

**Important**: Use `arxiv.Client().results(search)`, never the deprecated `search.results()` shorthand. The `arxiv` package has removed `Search.results()` in some releases while keeping it (with a `DeprecationWarning`) in others — an unpinned `arxiv` version can silently break paper retrieval in one environment while working in another (this happened in production: HF Spaces resolved a version without `Search.results()`, while local dev had an older version that still had it). `ArxivClient` and `ArxivFastMCPServer` each hold a single shared `arxiv.Client()` instance (`self.client` / `self.arxiv_client`) — reuse it rather than constructing a new one per call.

### MCP (Model Context Protocol) Integration

The system supports **optional** integration with arXiv MCP servers as an alternative to direct arXiv API access. **FastMCP is now the default MCP implementation** when `USE_MCP_ARXIV=true`.

**Architecture Overview**:
- Three client options: Direct ArxivClient, Legacy MCPArxivClient, FastMCPArxivClient
- All clients implement the same interface for drop-in compatibility
- RetrieverAgent includes intelligent fallback from MCP to direct API
- App selects client based on environment variables with cascading fallback

**Client Selection Logic** (`app.py` lines 75-135):
1. `USE_MCP_ARXIV=false` → Direct ArxivClient (default)
2. `USE_MCP_ARXIV=true` + `USE_LEGACY_MCP=true` → Legacy MCPArxivClient
3. `USE_MCP_ARXIV=true` (default) → FastMCPArxivClient with auto-start server
4. Fallback cascade: FastMCP → Legacy MCP → Direct API

**FastMCP Implementation** (Recommended):

**Server** (`utils/fastmcp_arxiv_server.py`):
- Auto-start FastMCP server in background thread
- Implements tools: `search_papers`, `download_paper`, `list_papers`
- Uses standard `arxiv` library for arXiv API access
- Configurable port (default: 5555) via `FASTMCP_SERVER_PORT`
- Singleton pattern for application-wide server instance
- Graceful shutdown on app exit
- Compatible with local and HuggingFace Spaces deployment

**Client** (`utils/fastmcp_arxiv_client.py`):
- Async-first design with sync wrappers for Gradio compatibility
- Connects to FastMCP server via HTTP
- Lazy client initialization on first use
- Reuses legacy MCP's robust `_parse_mcp_paper()` logic
- **Built-in fallback**: Direct arXiv download if MCP fails
- Same retry logic (3 attempts, exponential backoff)
- Uses `nest-asyncio` for event loop compatibility

**Retriever Fallback Logic** (`agents/retriever.py` lines 68-156):
- Two-tier fallback: Primary client → Fallback client
- `_search_with_fallback()`: Try primary MCP, then fallback to direct API
- `_download_with_fallback()`: Try primary MCP, then fallback to direct API
- Ensures paper retrieval never fails due to MCP issues
- Detailed logging of fallback events

**Legacy MCP Client** (`utils/mcp_arxiv_client.py`):
- In-process handler calls (imports MCP server functions directly)
- Stdio protocol for external MCP servers
- Maintained for backward compatibility
- Enable via `USE_LEGACY_MCP=true` when `USE_MCP_ARXIV=true`
- All features from legacy implementation preserved

**Key Features Across All MCP Clients**:
- Async-first design with sync wrappers
- MCP tools: `search_papers`, `download_paper`, `list_papers`
- Transforms MCP responses to `Paper` Pydantic objects
- Same retry logic and caching behavior as ArxivClient
- Automatic direct download fallback if MCP storage inaccessible

**Zero Breaking Changes**:
- Downstream agents (Analyzer, Synthesis, Citation) unaffected
- Same state dictionary structure maintained
- PDF processing, chunking, and RAG unchanged
- Toggle via environment variables without code changes
- Legacy MCP remains available for compatibility

**Configuration** (`.env.example`):
```bash
# Enable MCP (FastMCP by default)
USE_MCP_ARXIV=true

# Force legacy MCP instead of FastMCP (optional)
USE_LEGACY_MCP=false

# Storage path for papers (used by all MCP clients)
MCP_ARXIV_STORAGE_PATH=./data/mcp_papers/

# FastMCP server port
FASTMCP_SERVER_PORT=5555
```

**Testing**:
- FastMCP: `pytest tests/test_fastmcp_arxiv.py -v` (38 tests)
- Legacy MCP: `pytest tests/test_mcp_arxiv_client.py -v` (21 tests)
- Both test suites cover: search, download, caching, error handling, fallback logic

### PDF Processing Edge Cases

- Some PDFs may be scanned images (extraction fails gracefully)
- Page markers `[Page N]` extracted during text extraction for chunk attribution
- Section detection is heuristic-based (checks first 5 lines of chunk)
- Empty pages or extraction failures logged as warnings, not errors

### Gradio UI Structure (`app.py`)

ResearchPaperAnalyzer class orchestrates the workflow:
1. Initialize LangFuse client and instrument Azure OpenAI (if enabled)
2. Create LangGraph workflow with all agents
3. Check semantic cache first
4. Initialize state dictionary with `create_initial_state()`
5. Generate unique `session_id` for trace tracking
6. Run LangGraph workflow via `run_workflow()` from orchestration module
7. Flush LangFuse traces to ensure upload
8. Cache results on success
9. Format output for 5 tabs: Papers, Analysis, Synthesis, Citations, Stats

**LangGraph Workflow Execution**:
- Nodes execute in order: retriever → analyzer → filter → synthesis → citation
- Conditional edges for early termination (no papers found, all analyses failed)
- Checkpointing enabled via `MemorySaver` for workflow state persistence
- Progress updates still work via local variable (NOT in state to avoid msgpack serialization issues)

## Testing Patterns

Tests use mocks to avoid external dependencies:

```python
# Mock RAG retriever
mock_retriever = Mock(spec=RAGRetriever)
mock_retriever.retrieve.return_value = {"chunks": [...], "chunk_ids": [...]}

# Mock Azure OpenAI
with patch('agents.analyzer.AzureOpenAI', return_value=mock_client):
    agent = AnalyzerAgent(rag_retriever=mock_retriever)
```

Current test coverage:
- **AnalyzerAgent** (18 tests): Core analysis workflow and error handling
- **MCPArxivClient** (21 tests): Legacy MCP tool integration, async/sync wrappers, response parsing
- **FastMCPArxiv** (38 tests): FastMCP server, client, integration, error handling, fallback logic

When adding tests for other agents, follow the same pattern:
- Fixtures for mock dependencies
- Test both success and error paths
- Verify state transformations
- Test edge cases (empty inputs, API failures)
- For async code, use `pytest-asyncio` with `@pytest.mark.asyncio`

## Observability and Analytics

### LangFuse Integration

The system automatically traces all agent executions and LLM calls when LangFuse is enabled, built on the **LangFuse v3 SDK** (OpenTelemetry-based; pinned `langfuse>=3.10.0,<4.0.0` in `requirements.txt` — v3 removed the v2 API this code originally targeted, e.g. `langfuse.decorators`, `client.trace()`, so an unbounded version constraint can silently break tracing in a fresh environment while an already-installed older SDK keeps working locally).

**Configuration** (`utils/langfuse_client.py`):
- `initialize_langfuse()`: Initialize global LangFuse client at startup
- `instrument_openai()`: Auto-trace all Azure OpenAI API calls
- `@observe` decorator: Trace custom functions/spans, wrapping the real v3 `langfuse.observe`. The enabled/disabled check happens at **call time**, not decoration time — `@observe` is applied to agent methods at module import time, before `initialize_langfuse()` runs at app startup, so gating at decoration time would permanently disable tracing regardless of configuration.
- `workflow_trace(name, session_id=None, user_id=None, metadata=None)`: Context manager that opens **one root span per workflow run** via `start_as_current_span()` and tags it with session/user via `update_current_trace()`. Wrapped around `app.invoke(...)` in `orchestration/workflow_graph.py::run_workflow()` so every node/agent span and LLM generation nests under a single trace instead of appearing as disconnected top-level traces. Safely no-ops (never raises) if the installed SDK lacks this method or tracing is disabled — a tracing failure must never crash the actual workflow. **Pre-generates the trace ID** via `Langfuse.create_trace_id(seed=session_id)` and forces the span onto it via `start_as_current_span(trace_context={"trace_id": ...})`, rather than reading the ID back after the span opens — this makes the real trace ID available to the caller *before* the wrapped workflow body runs, so `run_workflow()` can stamp it onto `initial_state["trace_id"]` prior to `app.invoke()` and have it flow through every node.
- `check_langfuse_auth()`: Live credential check against the LangFuse API (`client.auth_check()`), independent of the module-level singleton. Used by `scripts/validate_langfuse_keys.py` — run this before deploying to confirm keys actually authenticate, since presence of keys alone doesn't guarantee they're valid.
- `flush_langfuse()`: Ensure all traces uploaded before shutdown

**Correlating Python logs with LangFuse traces** (`utils/trace_context.py`, added v2.10):
- A `contextvars.ContextVar` holds the active `trace_id`; `set_current_trace_id()`/`reset_current_trace_id()` bind and unbind it around `app.invoke()` in `run_workflow()`.
- `TraceIdLogFilter` (a `logging.Filter`) reads that contextvar and injects `record.trace_id` into every `LogRecord`. It's attached to the root logger's **handler** (`logging.getLogger().handlers[0].addFilter(...)`, done once in `app.py`) rather than to a `Logger` object — filters on a `Logger` only apply to records logged directly through that logger instance, not to records from child loggers (e.g. `agents.analyzer`, `orchestration.nodes`) that just propagate up to the root handler.
- `app.py`'s `logging.basicConfig(...)` format string includes `[trace_id=%(trace_id)s]`. Because `logging.basicConfig()` is a no-op after the first call in a process, and `app.py` imports/configures logging before any `agents`/`rag`/`orchestration` module runs its own (identical, now-redundant) `basicConfig()` call, this one change makes **every existing `logger.info/warning/error(...)` call in every node and agent** include the trace ID — no per-file changes needed.
- Because `contextvars.copy_context()` snapshots *all* active contextvars (not just LangFuse's own OTEL ones), this trace ID contextvar is automatically inherited by `AnalyzerAgent`'s `ThreadPoolExecutor` workers too, via the same `copy_context().run(...)` submission pattern described below — so per-paper analyzer log lines are also correctly tagged.
- End result: application logs can be filtered/grepped by trace ID independent of the LangFuse UI, and the same ID correlates a log line to its exact LangFuse trace.

**Automatic Tracing**:
- All agent `run()` methods decorated with `@observe`
- LLM calls automatically captured (prompt, completion, tokens, cost)
- RAG operations traced (embeddings, vector search)
- Workflow state transitions logged
- **Thread-pool gotcha**: `contextvars` (which OTEL/LangFuse tracing relies on) are not propagated into new threads automatically. `AnalyzerAgent`'s `ThreadPoolExecutor` explicitly runs each submitted task inside `contextvars.copy_context()` for this reason — any new parallelized code path must do the same or its spans will be orphaned (see "Adding a new agent" below).

### Trace Querying (`observability/trace_reader.py`)

```python
from observability import TraceReader

reader = TraceReader()

# Get recent traces
traces = reader.get_traces(limit=10)

# Filter by user/session
traces = reader.get_traces(user_id="user-123", session_id="session-abc")

# Filter by date range
from datetime import datetime, timedelta
start = datetime.now() - timedelta(days=7)
traces = reader.filter_by_date_range(traces, start_date=start)

# Get specific agent executions
analyzer_spans = reader.filter_by_agent(traces, agent_name="analyzer_agent")

# Export traces
reader.export_traces_to_json(traces, "traces.json")
reader.export_traces_to_csv(traces, "traces.csv")
```

### Performance Analytics (`observability/analytics.py`)

```python
from observability import AgentPerformanceAnalyzer, AgentTrajectoryAnalyzer

# Performance metrics
perf_analyzer = AgentPerformanceAnalyzer()

# Get agent latency statistics
stats = perf_analyzer.agent_latency_stats("analyzer_agent", days=7)
print(f"P95 latency: {stats.p95_latency_ms:.2f}ms")

# Token usage breakdown
token_usage = perf_analyzer.token_usage_breakdown(days=7)
print(f"Total tokens: {sum(token_usage.values())}")

# Cost per agent
costs = perf_analyzer.cost_per_agent(days=7)
print(f"Total cost: ${sum(costs.values()):.4f}")

# Error rates
error_rates = perf_analyzer.error_rates(days=7)

# Workflow summary
summary = perf_analyzer.workflow_performance_summary(days=7)
print(f"Success rate: {summary.success_rate:.1f}%")
print(f"Avg duration: {summary.avg_duration_ms/1000:.2f}s")

# Trajectory analysis
traj_analyzer = AgentTrajectoryAnalyzer()
analysis = traj_analyzer.analyze_execution_paths(days=7)
print(f"Most common path: {analysis['most_common_path']}")
```

See `observability/README.md` for comprehensive documentation.

## Common Modification Points

**Adding a new agent**:
1. Create agent class with `run(state) -> state` method
2. Decorate `run()` with `@observe` for tracing
3. Add node wrapper in `orchestration/nodes.py`
4. Add node to workflow graph in `orchestration/workflow_graph.py`
5. Update conditional routing if needed
6. **If the agent parallelizes work with `ThreadPoolExecutor`** (see `AnalyzerAgent`), propagate tracing context into worker threads explicitly — `contextvars` (which LangFuse's OTEL-based tracing relies on) are not inherited by new threads automatically. Wrap submitted callables: `executor.submit(contextvars.copy_context().run, fn, *args)`. Without this, spans created inside worker threads become disconnected top-level traces instead of nesting under the current run.

**Modifying chunking**:
- Adjust `chunk_size` and `chunk_overlap` in PDFProcessor initialization
- Affects retrieval quality vs. context size tradeoff
- Default 500/50 balances precision and coverage

**Changing LLM model**:
- Update `AZURE_OPENAI_DEPLOYMENT_NAME` in .env
- Cost estimates in CitationAgent may need adjustment
- Temperature must stay 0 for deterministic outputs

**Adding arXiv categories**:
- Extend `ARXIV_CATEGORIES` list in `app.py`
- Format: `"code - Description"` (e.g., `"cs.AI - Artificial Intelligence"`)

**Switching between arXiv clients**:
- Set `USE_MCP_ARXIV=false` (default) → Direct ArxivClient
- Set `USE_MCP_ARXIV=true` → FastMCPArxivClient (default MCP)
- Set `USE_MCP_ARXIV=true` + `USE_LEGACY_MCP=true` → Legacy MCPArxivClient
- Configure `MCP_ARXIV_STORAGE_PATH` for MCP server's storage location
- Configure `FASTMCP_SERVER_PORT` for FastMCP server port (default: 5555)
- No code changes required - client selected automatically in `app.py`
- All clients implement identical interface for seamless switching
- FastMCP server auto-starts when FastMCP client is selected

## Cost and Performance Considerations

- Target: <$0.50 per 5-paper analysis
- Semantic cache reduces repeated query costs
- ChromaDB persistence prevents re-embedding same papers
- Batch embedding generation in PDFProcessor for efficiency
- Token usage tracked per request for monitoring
- LangFuse observability enables cost optimization insights
- LangGraph overhead: <1% for state management
- Trace upload overhead: ~5-10ms per trace (async, negligible impact)

## Key Files and Modules

### Core Application
- `app.py`: Gradio UI and workflow orchestration entry point
- `utils/config.py`: Configuration management (Azure OpenAI, LangFuse, MCP)
- `utils/schemas.py`: Pydantic data models for validation
- `utils/langgraph_state.py`: LangGraph state TypedDict and helpers

### Agents
- `agents/retriever.py`: Paper retrieval, PDF processing, embeddings
- `agents/analyzer.py`: Individual paper analysis with RAG
- `agents/synthesis.py`: Cross-paper synthesis and insights
- `agents/citation.py`: Citation generation and validation

### RAG Components
- `rag/pdf_processor.py`: PDF text extraction and chunking
- `rag/embeddings.py`: Batch embedding generation (Azure OpenAI)
- `rag/vector_store.py`: ChromaDB vector store management
- `rag/retrieval.py`: RAG retrieval with formatted context

### Orchestration (LangGraph)
- `orchestration/__init__.py`: Module exports
- `orchestration/nodes.py`: Node wrappers with tracing
- `orchestration/workflow_graph.py`: LangGraph workflow builder

### Observability (LangFuse)
- `observability/__init__.py`: Module exports
- `observability/trace_reader.py`: Trace querying and export API
- `observability/analytics.py`: Performance analytics and trajectory analysis
- `observability/README.md`: Comprehensive observability documentation
- `utils/langfuse_client.py`: LangFuse client initialization and helpers
- `utils/trace_context.py`: Contextvar + `logging.Filter` correlating Python logs with the active LangFuse trace ID (NEW v2.10)

### Utilities
- `utils/arxiv_client.py`: Direct arXiv API client with retry logic
- `utils/mcp_arxiv_client.py`: Legacy MCP client implementation
- `utils/fastmcp_arxiv_client.py`: FastMCP client (recommended)
- `utils/fastmcp_arxiv_server.py`: FastMCP server with auto-start
- `utils/semantic_cache.py`: Query caching with embeddings

### Scripts
- `scripts/validate_langfuse_keys.py`: Standalone CLI — live LangFuse API key validation (`python scripts/validate_langfuse_keys.py`). Run before deploying to catch invalid/missing keys and SDK/host issues before they surface as silent tracing failures.

### Documentation
- `CLAUDE.md`: This file - comprehensive developer guide
- `README.md`: User-facing project documentation
- `REFACTORING_SUMMARY.md`: LangGraph + LangFuse refactoring details
- `BUGFIX_MSGPACK_SERIALIZATION.md`: msgpack serialization fix documentation
- `.env.example`: Environment variable template with all options

## Version History and Recent Changes

### Version 2.10: trace_id/user_id Propagation Through State and Logs

**Fixed:**
- **`user_id` always null in LangFuse** — `app.py`'s Gradio handler never passed a `user_id` into `create_initial_state()`, so it silently defaulted to `None`. There's no auth in this app, so `run_workflow()` (both the module-level `analyze_research()` function and `ResearchPaperAnalyzer.run_workflow()`) now accept a `request: gr.Request` parameter (Gradio auto-injects it) and derive `user_id = request.session_hash` — an anonymous but stable per-browser-tab identifier.
- **`trace_id` always null in `state`, including in the `finalize_node` span's own captured input/output** — `create_initial_state()` hardcoded `trace_id: None` and nothing ever wrote to it during the run; it was previously only stamped onto `final_state` *after* `app.invoke()` returned, so no node ever actually saw it. Fixed by having `workflow_trace()` (`utils/langfuse_client.py`) **pre-generate** the trace ID via `Langfuse.create_trace_id(seed=session_id)` and force the root span onto it via `start_as_current_span(trace_context={"trace_id": ...})`, instead of reading the ID back after the span opens. This makes the real trace ID available to `run_workflow()` (`orchestration/workflow_graph.py`) *before* `app.invoke()` is called, so it can stamp `initial_state["trace_id"]` up front — since no node in `orchestration/nodes.py` ever overwrites that key, it now flows unchanged through every node (retriever → analyzer → filter → synthesis → citation → finalize).

**Added:**
- `utils/trace_context.py` — a `contextvars.ContextVar` holding the active trace ID, with `set_current_trace_id()`/`reset_current_trace_id()` bound around `app.invoke()`, plus `TraceIdLogFilter` (a `logging.Filter`) that injects `record.trace_id` into every `LogRecord`. Attached once to the root logger's handler in `app.py` (with `[trace_id=%(trace_id)s]` added to the log format string) — since `logging.basicConfig()` only takes effect on its first call per process and `app.py` configures logging before any `agents`/`rag`/`orchestration` module does, this single change makes **every existing log line in every node and agent** filterable by trace ID, with zero changes to `nodes.py` or any `agents/*.py` file. The contextvar is also automatically inherited by `AnalyzerAgent`'s `ThreadPoolExecutor` workers via the `contextvars.copy_context()` propagation pattern already in place there (see [Adding a new agent](#common-modification-points)), so per-paper analyzer logs are tagged too.

**Changed:**
- `utils/config.py`: `LangFuseConfig.host` now reads `LANGFUSE_BASE_URL` (default `https://us.cloud.langfuse.com`) instead of `LANGFUSE_HOST` (default `https://cloud.langfuse.com`). `.env.example` and `scripts/validate_langfuse_keys.py` (which read the host var independently of `utils/config.py`) updated to match, so the validator's diagnostic output stays accurate.

**Lesson reinforced**: when a value needs to be visible *during* a run (not just attached to the result afterward), it has to be established before the run starts — `workflow_trace()` had to switch from *reading back* an auto-generated trace ID to *pre-generating* one so it could be seeded into state ahead of `app.invoke()`, the same "compute it early enough for downstream consumers" pattern behind `session_id` and now `trace_id`.

### Version 2.9: LangFuse v3 SDK Migration + Production Dependency Fixes

**Fixed:**
- **LangFuse traces not appearing in dashboard** — `requirements.txt`'s unbounded `langfuse>=2.0.0` let pip resolve LangFuse v3, an OTEL-based rewrite that removed the v2 API (`langfuse.decorators`, `client.trace()`) `utils/langfuse_client.py` was written against, with every failure silently swallowed by broad `except` blocks. Migrated `observe()` and replaced `start_trace()`/`trace_context` (both dead code) with `workflow_trace()`, a context manager opening one root span per workflow run.
- **`@observe` decoration-time gating bug** — the enabled/disabled check was evaluated when `@observe` was applied (at module import time, before `initialize_langfuse()` ever runs at startup), permanently disabling tracing regardless of config. Moved the check to call time.
- **`ThreadPoolExecutor` orphaning spans** — `agents/analyzer.py`'s parallel paper analysis broke span nesting because `contextvars` aren't propagated into new threads by default. Fixed by submitting tasks via `contextvars.copy_context().run(...)`.
- **HF Spaces crash: `'Langfuse' object has no attribute 'start_as_current_span'`** — pinned `langfuse>=3.10.0,<4.0.0` (preventing a repeat of the unbounded-version issue) and hardened `workflow_trace()` so a tracing-layer failure degrades to running the workflow untraced instead of crashing the entire request — the same lesson as the RAG/agent error-handling philosophy, now applied to the tracing layer itself.
- **HF Spaces returning zero papers: `'Search' object has no attribute 'results'`** — unrelated to LangFuse, but the same root-cause pattern: unbounded `arxiv>=2.0.0` let a fresh install resolve a version that removed the deprecated `Search.results()` shorthand used in `utils/arxiv_client.py` and `utils/fastmcp_arxiv_server.py`. Migrated both to the non-deprecated `arxiv.Client().results(search)` API (each class now holds one shared `arxiv.Client()` instance).

**Added:**
- `scripts/validate_langfuse_keys.py` + `check_langfuse_auth()` — live API key validation (not just a presence check), run before deploying
- Single-trace-per-query tracing: `research_workflow_run` root trace containing all node/agent spans and LLM generations properly nested, verified directly via the LangFuse API's trace observation tree (not just visual inspection)

**Lesson reinforced**: unbounded `>=` version constraints on fast-moving dependencies (LangFuse, arxiv) let different environments (local dev vs. fresh HF Spaces build) silently resolve different major versions with breaking API changes. Prefer bounded ranges once a specific API surface is depended on, and — just as importantly — make failures in secondary systems (observability, tracing) degrade gracefully rather than take down the primary workflow.

### Version 2.6: LangGraph Orchestration + LangFuse Observability
**Added:**
- LangGraph workflow orchestration with conditional routing
- LangFuse automatic tracing for all agents and LLM calls
- Observability Python API for trace querying and analytics
- Performance analytics (latency, tokens, cost, error rates)
- Agent trajectory analysis
- Checkpointing with `MemorySaver`

**Fixed:**
- msgpack serialization error (removed Gradio Progress from state)

**Dependencies Added:**
- `langgraph>=0.2.0`
- `langfuse>=2.0.0`
- `langfuse-openai>=1.0.0`

**Breaking Changes:**
- None! Fully backward compatible

**Documentation:**
- Created `observability/README.md`
- Created `REFACTORING_SUMMARY.md`
- Created `BUGFIX_MSGPACK_SERIALIZATION.md`
- Updated `CLAUDE.md` (this file)
- Updated `.env.example`

See `REFACTORING_SUMMARY.md` for detailed migration guide and architecture changes.
