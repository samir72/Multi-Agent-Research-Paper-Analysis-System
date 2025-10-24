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
   - Searches arXiv API using `ArxivClient`
   - Downloads PDFs to `data/papers/` (cached)
   - Processes PDFs with `PDFProcessor` (500-token chunks, 50-token overlap)
   - Generates embeddings via `EmbeddingGenerator` (Azure OpenAI text-embedding-3-small)
   - Stores chunks in ChromaDB via `VectorStore`

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

Current test coverage focuses on AnalyzerAgent (18 tests). When adding tests for other agents, follow the same pattern:
- Fixtures for mock dependencies
- Test both success and error paths
- Verify state transformations
- Test edge cases (empty inputs, API failures)

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

## Cost and Performance Considerations

- Target: <$0.50 per 5-paper analysis
- Semantic cache reduces repeated query costs
- ChromaDB persistence prevents re-embedding same papers
- Batch embedding generation in PDFProcessor for efficiency
- Token usage tracked per request for monitoring
