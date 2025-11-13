# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core Architecture

This is a **multi-agent RAG system** for analyzing academic papers from arXiv. The system uses a sequential agent workflow where state flows through 4 specialized agents:

### Agent Pipeline Flow

```
User Query → Retriever → Analyzer → Synthesis → Citation → Output
```

**State Dictionary**: All agents operate on a shared state dictionary that flows through the pipeline:
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

### Agent Responsibilities

1. **RetrieverAgent** (`agents/retriever.py`):
   - Searches arXiv API using `ArxivClient`, `MCPArxivClient`, or `FastMCPArxivClient` (configurable via env)
   - Downloads PDFs to `data/papers/` (direct API) or MCP server storage (MCP mode)
   - **Intelligent Fallback**: Automatically falls back to direct API if primary MCP client fails
   - Processes PDFs with `PDFProcessor` (500-token chunks, 50-token overlap)
   - Generates embeddings via `EmbeddingGenerator` (Azure OpenAI text-embedding-3-small)
   - Stores chunks in ChromaDB via `VectorStore`
   - **FastMCP Support**: Auto-start FastMCP server for standardized arXiv access

2. **AnalyzerAgent** (`agents/analyzer.py`):
   - Analyzes each paper individually using RAG
   - Uses 4 broad queries per paper: methodology, results, conclusions, limitations
   - Deduplicates chunks by chunk_id
   - Calls Azure OpenAI with **temperature=0** and JSON mode
   - Returns structured `Analysis` objects with confidence scores

3. **SynthesisAgent** (`agents/synthesis.py`):
   - Compares findings across all papers
   - Identifies consensus points, contradictions, research gaps
   - Creates executive summary addressing user's query
   - Uses **temperature=0** for deterministic outputs
   - Returns `SynthesisResult` with confidence scores

4. **CitationAgent** (`agents/citation.py`):
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

### Pydantic Schemas (`utils/schemas.py`)

All data structures use Pydantic for validation:
- `Paper`: arXiv paper metadata
- `PaperChunk`: Text chunk with metadata
- `Analysis`: Individual paper analysis results
- `SynthesisResult`: Cross-paper synthesis with ConsensusPoint and Contradiction
- `ValidatedOutput`: Final output with citations and cost tracking
- `AgentState`: Complete state dictionary (not actively used but defined)

### Retry Logic

ArxivClient uses tenacity for resilient API calls:
- 3 retry attempts
- Exponential backoff (4s min, 10s max)
- Applied to search_papers() and download_paper()

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
1. Check semantic cache first
2. Initialize state dictionary
3. Run agents sequentially with progress updates
4. Cache results on success
5. Format output for 5 tabs: Papers, Analysis, Synthesis, Citations, Stats

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

## Common Modification Points

**Adding a new agent**:
1. Create agent class with `run(state) -> state` method
2. Add to `ResearchPaperAnalyzer.__init__()` in `app.py`
3. Insert into workflow in `ResearchPaperAnalyzer.run_workflow()`
4. Update progress tracking

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
