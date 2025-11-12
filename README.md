---
title: Research Paper Analyzer
emoji: 📚
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
---

# Multi-Agent Research Paper Analysis System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gradio](https://img.shields.io/badge/Gradio-5.49.1-orange)](https://gradio.app/)
[![Azure OpenAI](https://img.shields.io/badge/Azure-OpenAI-0078D4)](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
[![Sync to HF Space](https://github.com/samir72/Multi-Agent-Research-Paper-Analysis-System/actions/workflows/sync-to-hf-space.yml/badge.svg)](https://github.com/samir72/Multi-Agent-Research-Paper-Analysis-System/actions/workflows/sync-to-hf-space.yml)

A production-ready multi-agent system that analyzes academic papers from arXiv, extracts insights, synthesizes findings across papers, and provides deterministic, citation-backed responses to research questions.

**🚀 Quick Start**: See [QUICKSTART.md](QUICKSTART.md) for a 5-minute setup guide.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Technical Stack](#technical-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [Testing](#testing)
- [Performance](#performance)
- [Deployment](#deployment)
  - [GitHub Actions - Automated Deployment](#github-actions---automated-deployment)
  - [Hugging Face Spaces](#hugging-face-spaces-manual-deployment)
  - [Local Docker](#local-docker)
- [Programmatic Usage](#programmatic-usage)
- [Contributing](#contributing)
- [Support](#support)
- [Changelog](#changelog)

## Features

- **Automated Paper Retrieval**: Search and download papers from arXiv (direct API or MCP server)
- **RAG-Based Analysis**: Extract methodology, findings, conclusions, and limitations using retrieval-augmented generation
- **Cross-Paper Synthesis**: Identify consensus points, contradictions, and research gaps
- **Citation Management**: Generate proper APA-style citations with source validation
- **Semantic Caching**: Optimize costs by caching similar queries
- **Deterministic Outputs**: Temperature=0 and structured outputs for reproducibility
- **MCP Integration**: Optional Model Context Protocol support with automatic fallback
- **High Performance**: 4x faster with parallel processing (2-3 min for 5 papers)
- **Smart Error Handling**: Circuit breaker, graceful degradation, friendly error messages
- **Progressive UI**: Real-time updates as papers are analyzed with streaming results
- **Smart Quality Filtering**: Automatically excludes failed analyses (0% confidence) from synthesis
- **Enhanced UX**: Clickable PDF links, paper titles + confidence scores, status indicators
- **Comprehensive Testing**: 39 total tests (18 analyzer + 21 MCP) with diagnostic tools

## Architecture

### Agent Workflow

```
User Query → Retriever Agent → Analyzer Agent → Synthesis Agent → Citation Agent → User
```

**Streaming Workflow (v2.1):**
```
User Query → Retriever → [Has papers?]
              ├─ Yes → Analyzer (parallel 4x, streaming) → Filter (0% confidence) → Synthesis → Citation → User
              └─ No → END (graceful error)
```

**Key Features:**
- **Progressive Streaming**: Real-time UI updates using Python generators
- **Parallel Execution**: 4 papers analyzed concurrently with live status
- **Smart Filtering**: Removes failed analyses (0% confidence) before synthesis
- **Circuit Breaker**: Auto-stops after 2 consecutive failures
- **Status Tracking**: ⏸️ Pending → ⏳ Analyzing → ✅ Complete / ⚠️ Failed

### 4 Specialized Agents

1. **Retriever Agent**
   - Queries arXiv API based on user input
   - Downloads and parses PDF papers
   - Extracts metadata (title, authors, abstract, publication date)
   - Chunks papers into 500-token segments with 50-token overlap

2. **Analyzer Agent** (Performance Optimized v2.0)
   - **Parallel processing**: Analyzes up to 4 papers simultaneously
   - **Circuit breaker**: Stops after 2 consecutive failures
   - **Timeout**: 60s with max_tokens=1500 for fast responses
   - Extracts methodology, findings, conclusions, limitations, contributions
   - Returns structured JSON with confidence scores

3. **Synthesis Agent**
   - Compares findings across multiple papers
   - Identifies consensus points and contradictions
   - Generates deterministic summary grounded in retrieved content
   - Highlights research gaps

4. **Citation Agent**
   - Validates all claims against source papers
   - Provides exact section references with page numbers
   - Generates properly formatted citations (APA style)
   - Ensures every statement is traceable to source

## Technical Stack

- **LLM**: Azure OpenAI (gpt-4o-mini or Phi-4-multimodal-instruct) with temperature=0
- **Embeddings**: Azure OpenAI text-embedding-3-small
- **Vector Store**: ChromaDB with persistent storage
- **Agent Framework**: Generator-based streaming workflow with progressive UI updates
- **Parallel Processing**: ThreadPoolExecutor (4 concurrent workers) with as_completed for streaming
- **UI**: Gradio 5.49.1 with tabbed interface and real-time updates
- **Data Source**: arXiv API (direct) or arXiv MCP server (optional)
- **MCP Integration**: Model Context Protocol client with automatic fallback
- **Testing**: pytest with comprehensive test suite (pytest-asyncio for async tests)
- **Type Safety**: Pydantic V2 schemas for validation
- **Pricing**: Configurable pricing system (JSON + environment overrides)

## Installation

### Prerequisites

- Python 3.10+
- Azure OpenAI account with API access

### Setup

1. Clone the repository:
```bash
git clone https://github.com/samir72/Multi-Agent-Research-Paper-Analysis-System.git
cd Multi-Agent-Research-Paper-Analysis-System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your Azure OpenAI credentials
```

Required environment variables:
- `AZURE_OPENAI_ENDPOINT`: Your Azure OpenAI endpoint (e.g., https://your-resource.openai.azure.com/)
- `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key
- `AZURE_OPENAI_DEPLOYMENT_NAME`: Your deployment name (e.g., gpt-4o-mini or phi-4-multimodal-instruct)
- `AZURE_OPENAI_API_VERSION`: API version (optional, defaults in code)

Optional:
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`: Custom embedding model deployment name
- `PRICING_INPUT_PER_1M`: Override input token pricing for all models (per 1M tokens)
- `PRICING_OUTPUT_PER_1M`: Override output token pricing for all models (per 1M tokens)
- `PRICING_EMBEDDING_PER_1M`: Override embedding token pricing (per 1M tokens)

**MCP (Model Context Protocol) Support** (Optional):
- `USE_MCP_ARXIV`: Set to `true` to use arXiv MCP server instead of direct arXiv API (default: `false`)
- `MCP_ARXIV_STORAGE_PATH`: Path where MCP server stores papers (default: `./data/mcp_papers/`)

**Note**: Pricing is configured in `config/pricing.json` with support for phi-4-multimodal-instruct, gpt-4o-mini, and gpt-4o. Environment variables override JSON settings.

### MCP (Model Context Protocol) Integration

The system supports using an external arXiv MCP server as an alternative to direct arXiv API access. This provides a standardized protocol for accessing arXiv papers through the Model Context Protocol.

**Setup Instructions:**

1. Install the arXiv MCP server:
```bash
pip install arxiv-mcp-server
```

2. Configure the MCP server (e.g., in Claude Desktop's config):
```json
{
  "mcpServers": {
    "arxiv": {
      "command": "arxiv-mcp-server",
      "env": {
        "ARXIV_STORAGE_PATH": "/path/to/papers"
      }
    }
  }
}
```

3. Enable MCP in your `.env`:
```bash
USE_MCP_ARXIV=true
MCP_ARXIV_STORAGE_PATH=/path/to/papers
```

**Features:**
- Standardized access to arXiv papers via MCP tools
- Same functionality as direct API (search, download, metadata extraction)
- Compatible with existing workflow (zero breaking changes)
- **Automatic fallback**: Downloads directly from arXiv if MCP server storage is inaccessible
- Comprehensive tool discovery and diagnostic logging

**Note:** The system defaults to direct arXiv API access. MCP integration is optional and can be toggled via environment variable.

**Troubleshooting MCP Downloads:**
If PDFs aren't appearing in your storage directory, see [MCP_FIX_DOCUMENTATION.md](MCP_FIX_DOCUMENTATION.md) for:
- Root cause explanation (client-server storage mismatch)
- How the automatic fallback works
- Diagnostic script: `python test_mcp_diagnostic.py`
- Detailed troubleshooting guide

**Data Management:**

```bash
# Clear MCP cached papers
rm -rf data/mcp_papers/

# Clear direct API cached papers
rm -rf data/papers/

# Clear vector store (useful for testing)
rm -rf data/chroma_db/

# Clear semantic cache
rm -rf data/cache/
```

4. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:7860`

## Usage

1. **Enter Research Question**: Type your research question in the text box
2. **Select Category**: Choose an arXiv category or leave as "All"
3. **Set Number of Papers**: Use the slider to select 1-20 papers
4. **Click Analyze**: The system will process your request with real-time updates
5. **View Results**: Explore the five output tabs with progressive updates:
   - **Papers**: Table of retrieved papers with clickable PDF links and live status (⏸️ Pending → ⏳ Analyzing → ✅ Complete / ⚠️ Failed)
   - **Analysis**: Detailed analysis of each paper (updates as each completes)
   - **Synthesis**: Executive summary with consensus and contradictions (populated after all analyses)
   - **Citations**: APA-formatted references with validation
   - **Stats**: Processing statistics, token usage, and cost estimates

## Project Structure

```
Multi-Agent-Research-Paper-Analysis-System/
├── app.py                          # Main Gradio application with streaming workflow
├── requirements.txt                # Python dependencies
├── README.md                       # This file - full documentation
├── QUICKSTART.md                   # Quick setup guide (5 minutes)
├── .env.example                    # Environment variable template
├── .gitignore                      # Git ignore rules
├── agents/
│   ├── __init__.py
│   ├── retriever.py               # Paper retrieval & chunking
│   ├── analyzer.py                # Individual paper analysis (parallel + streaming)
│   ├── synthesis.py               # Cross-paper synthesis
│   └── citation.py                # Citation validation & formatting
├── rag/
│   ├── __init__.py
│   ├── vector_store.py            # ChromaDB vector storage
│   ├── embeddings.py              # Azure OpenAI text embeddings
│   └── retrieval.py               # RAG retrieval & context formatting
├── utils/
│   ├── __init__.py
│   ├── arxiv_client.py            # arXiv API wrapper (direct API)
│   ├── mcp_arxiv_client.py        # arXiv MCP client (optional)
│   ├── pdf_processor.py           # PDF parsing & chunking
│   ├── cache.py                   # Semantic caching layer
│   ├── config.py                  # Pricing configuration management
│   └── schemas.py                 # Pydantic data models
├── config/
│   └── pricing.json               # Model pricing configuration
├── tests/
│   ├── __init__.py
│   ├── test_analyzer.py           # Unit tests for analyzer agent
│   └── test_mcp_arxiv_client.py   # Unit tests for MCP client (21 tests)
├── test_mcp_diagnostic.py         # MCP setup diagnostic script (NEW)
├── MCP_FIX_DOCUMENTATION.md       # MCP troubleshooting guide (NEW)
├── MCP_FIX_SUMMARY.md             # MCP fix quick reference (NEW)
└── data/                           # Created at runtime
    ├── papers/                     # Downloaded PDFs (direct API, cached)
    ├── mcp_papers/                 # Downloaded PDFs (MCP mode, cached)
    └── chroma_db/                  # Vector store persistence
```

## Key Features

### Progressive Streaming UI

The system provides real-time feedback during analysis with a generator-based streaming workflow:

1. **Papers Tab Updates**: Status changes live as papers are processed
   - ⏸️ **Pending**: Paper queued for analysis
   - ⏳ **Analyzing**: Analysis in progress
   - ✅ **Complete**: Analysis successful with confidence score
   - ⚠️ **Failed**: Analysis failed (0% confidence, excluded from synthesis)
2. **Incremental Results**: Analysis tab populates as each paper completes
3. **ThreadPoolExecutor**: Up to 4 papers analyzed concurrently with `as_completed()` for streaming
4. **Python Generators**: Uses `yield` to stream results without blocking

### Deterministic Output Strategy

The system implements multiple techniques to minimize hallucinations:

1. **Temperature=0**: All Azure OpenAI calls use temperature=0
2. **Structured Outputs**: JSON mode for agent responses with strict schemas
3. **RAG Grounding**: Every response includes retrieved chunk IDs
4. **Source Validation**: Cross-reference all claims with original text
5. **Semantic Caching**: Hash query embeddings, return cached results for cosine similarity >0.95
6. **Confidence Scores**: Return uncertainty metrics with each response
7. **Smart Filtering**: Papers with 0% confidence automatically excluded from synthesis

### Cost Optimization

- **Configurable Pricing System**: `config/pricing.json` for easy model switching
  - Supports phi-4-multimodal-instruct ($0.08/$0.32 per 1M tokens)
  - Supports gpt-4o-mini ($0.15/$0.60 per 1M tokens)
  - Environment variable overrides for testing and custom pricing
- **Thread-safe Token Tracking**: Accurate counts across parallel processing
- **Request Batching**: Batch embeddings for efficiency
- **Cached Embeddings**: ChromaDB stores embeddings (don't re-embed same papers)
- **Semantic Caching**: Return cached results for similar queries (cosine similarity >0.95)
- **Token Usage Logging**: Track input/output/embedding tokens per request
- **Target**: <$0.50 per analysis session (5 papers with phi-4)

### Error Handling

- **Smart Quality Control**: Automatically filters out 0% confidence analyses from synthesis
- **Visual Status Indicators**: Papers tab shows ⚠️ Failed for problematic papers
- **Graceful Degradation**: Failed papers don't block overall workflow
- **Circuit Breaker**: Stops after 2 consecutive failures in parallel processing
- **Timeout Protection**: 60s analyzer, 90s synthesis timeouts
- **Graceful Fallbacks**: Handle arXiv API downtime and PDF parsing failures
- **User-friendly Messages**: Clear error descriptions in Gradio UI
- **Comprehensive Logging**: Detailed error tracking for debugging

## Testing

The project includes a comprehensive test suite to ensure reliability and correctness.

### Running Tests

```bash
# Install testing dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_analyzer.py -v

# Run with coverage report
pytest tests/ --cov=agents --cov=rag --cov=utils -v

# Run specific test
pytest tests/test_analyzer.py::TestAnalyzerAgent::test_analyze_paper_success -v
```

### Test Coverage

**Current Test Suite:**
- **Analyzer Agent** (`tests/test_analyzer.py`): 18 comprehensive tests
  - Unit tests for initialization, prompt creation, and analysis
  - Error handling and edge cases
  - State management and workflow tests
  - Integration tests with mocked dependencies
  - Azure OpenAI client initialization tests
- **MCP arXiv Client** (`tests/test_mcp_arxiv_client.py`): 21 comprehensive tests
  - Async/sync wrapper tests for all client methods
  - MCP tool call mocking and response parsing
  - Error handling and fallback mechanisms
  - PDF caching and storage path management
  - Integration with Paper schema validation
  - Tool discovery and diagnostics
  - Direct download fallback scenarios

**What's Tested:**
- ✅ Agent initialization and configuration
- ✅ Individual paper analysis workflow
- ✅ Multi-query retrieval and chunk deduplication
- ✅ Error handling and graceful failures
- ✅ State transformation through agent runs
- ✅ Confidence score calculation
- ✅ Integration with RAG retrieval system
- ✅ Mock Azure OpenAI API responses

**Coming Soon:**
- Tests for Retriever Agent (arXiv download, PDF processing)
- Tests for Synthesis Agent (cross-paper comparison)
- Tests for Citation Agent (APA formatting, validation)
- Integration tests for full workflow
- RAG component tests (vector store, embeddings, retrieval)

### Test Architecture

Tests use:
- **pytest**: Test framework with fixtures
- **pytest-asyncio**: Async test support for MCP client
- **pytest-cov**: Code coverage reporting
- **unittest.mock**: Mocking external dependencies (Azure OpenAI, RAG components, MCP tools)
- **Pydantic models**: Type-safe test data structures
- **Isolated testing**: No external API calls in unit tests

### MCP Diagnostic Testing

For MCP integration troubleshooting, run the diagnostic script:

```bash
# Test MCP setup and configuration
python test_mcp_diagnostic.py
```

This diagnostic tool:
- ✅ Validates environment configuration (`USE_MCP_ARXIV`, `MCP_ARXIV_STORAGE_PATH`)
- ✅ Verifies storage directory setup and permissions
- ✅ Lists available MCP tools via tool discovery
- ✅ Tests search functionality with real queries
- ✅ Tests download with file verification
- ✅ Shows file system state before/after operations
- ✅ Provides detailed logging for troubleshooting

See [MCP_FIX_DOCUMENTATION.md](MCP_FIX_DOCUMENTATION.md) for detailed troubleshooting guidance.

## Performance

**Version 2.0 Metrics (October 2025):**

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **5 papers total** | 5-10 min | 2-3 min | **60-70% faster** |
| **Per paper** | 60-120s | 30-40s | **50-70% faster** |
| **Throughput** | 1 paper/min | ~3 papers/min | **3x increase** |
| **Token usage** | ~5,500/paper | ~5,200/paper | **5-10% reduction** |

**Key Optimizations:**
- ⚡ Parallel processing with ThreadPoolExecutor (4 concurrent workers)
- ⏱️ Smart timeouts: 60s analyzer, 90s synthesis
- 🔢 Token limits: max_tokens 1500/2500
- 🔄 Circuit breaker: stops after 2 consecutive failures
- 📝 Optimized prompts: reduced metadata overhead
- 📊 Enhanced logging: timestamps across all modules

**Cost**: <$0.50 per analysis session
**Accuracy**: Deterministic outputs with confidence scores
**Scalability**: 1-20 papers with graceful error handling

## Deployment

### GitHub Actions - Automated Deployment

This repository includes a GitHub Actions workflow that automatically syncs to Hugging Face Spaces on every push to the `main` branch.

**Workflow File:** `.github/workflows/sync-to-hf-space.yml`

**Features:**
- ✅ Auto-deploys to Hugging Face Space on every push to main
- ✅ Manual trigger available via `workflow_dispatch`
- ✅ Includes Git LFS support for large files
- ✅ Force pushes to keep Space in sync with GitHub

**Setup Instructions:**

1. Create a Hugging Face Space at `https://huggingface.co/spaces/your-username/your-space-name`
2. Get your Hugging Face token from [Settings > Access Tokens](https://huggingface.co/settings/tokens)
3. Add the token as a GitHub secret:
   - Go to your GitHub repository → Settings → Secrets and variables → Actions
   - Add a new secret named `HF_TOKEN` with your Hugging Face token
4. Update the workflow file with your Hugging Face username and space name (line 31)
5. Push to main branch - the workflow will automatically deploy!

**Monitoring:**
- View workflow runs: [Actions tab](https://github.com/samir72/Multi-Agent-Research-Paper-Analysis-System/actions)
- Workflow status badge shows current deployment status

### Hugging Face Spaces (Manual Deployment)

1. Create a new Space on Hugging Face
2. Upload all files from this repository
3. Add the following secrets in Space settings:
   - `AZURE_OPENAI_ENDPOINT`
   - `AZURE_OPENAI_API_KEY`
   - `AZURE_OPENAI_DEPLOYMENT_NAME`
4. The app will automatically deploy

### Local Docker

```bash
docker build -t research-analyzer .
docker run -p 7860:7860 --env-file .env research-analyzer
```

## Programmatic Usage

The system can be used programmatically without the Gradio UI:

```python
from app import ResearchPaperAnalyzer

# Initialize the analyzer
analyzer = ResearchPaperAnalyzer()

# Run analysis workflow
papers_df, analysis_html, synthesis_html, citations_html, stats = analyzer.run_workflow(
    query="What are the latest advances in multi-agent reinforcement learning?",
    category="cs.AI",
    num_papers=5
)

# Access individual agents
from utils.schemas import Paper
from datetime import datetime

# Create a paper object
paper = Paper(
    arxiv_id="2401.00001",
    title="Sample Paper",
    authors=["Author A", "Author B"],
    abstract="Paper abstract...",
    pdf_url="https://arxiv.org/pdf/2401.00001.pdf",
    published=datetime.now(),
    categories=["cs.AI"]
)

# Use individual agents
analysis = analyzer.analyzer_agent.analyze_paper(paper)
print(f"Methodology: {analysis.methodology}")
print(f"Key Findings: {analysis.key_findings}")
print(f"Confidence: {analysis.confidence_score:.2%}")
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes with tests (see [Testing](#testing) section)
4. Commit your changes (`git commit -m 'Add some feature'`)
5. Push to the branch (`git push origin feature/your-feature`)
6. Submit a pull request

### Development Guidelines

- Write tests for new features (see `tests/test_analyzer.py` for examples)
- Follow existing code style and patterns
- Update documentation for new features
- Ensure all tests pass: `pytest tests/ -v`
- Add type hints using Pydantic schemas where applicable

## License

MIT License - see LICENSE file for details

## Citation

If you use this system in your research, please cite:

```bibtex
@software{research_paper_analyzer,
  title={Multi-Agent Research Paper Analysis System},
  author={Sayed A Rizvi},
  year={2025},
  url={https://github.com/samir72/Multi-Agent-Research-Paper-Analysis-System}
}
```

## Acknowledgments

- arXiv for providing open access to research papers
- Azure OpenAI for LLM and embedding models
- ChromaDB for vector storage
- Gradio for the UI framework

## Support

For issues, questions, or feature requests, please:
- Open an issue on [GitHub](https://github.com/samir72/Multi-Agent-Research-Paper-Analysis-System/issues)
- Check [QUICKSTART.md](QUICKSTART.md) for common troubleshooting tips
- Review the [Testing](#testing) section for running tests

## Changelog

### Version 2.2 - November 2025 (Latest)

**🔌 MCP (Model Context Protocol) Integration:**
- ✅ **Optional MCP Support** - Use arXiv MCP server as alternative to direct API
  - New `MCPArxivClient` with same interface as `ArxivClient` for seamless switching
  - Toggle via `USE_MCP_ARXIV` environment variable (default: `false`)
  - Configurable storage path via `MCP_ARXIV_STORAGE_PATH` environment variable
  - Async-first design with sync wrappers for compatibility
- ✅ **MCP Download Fallback** - Guaranteed PDF downloads regardless of MCP server configuration
  - Automatic fallback to direct arXiv download when MCP storage is inaccessible
  - Handles remote MCP servers that don't share filesystem with client
  - Comprehensive tool discovery logging for diagnostics
  - Run `python test_mcp_diagnostic.py` to test MCP setup
- ✅ **Zero Breaking Changes** - Complete backward compatibility
  - RetrieverAgent accepts both `ArxivClient` and `MCPArxivClient` via dependency injection
  - Same state dictionary structure maintained across all agents
  - PDF processing, chunking, and RAG workflow unchanged
  - Client selection automatic based on environment variables

**📦 Dependencies Updated:**
- ✅ **New MCP packages** - Added to `requirements.txt`
  - `mcp>=0.9.0` - Model Context Protocol client library
  - `arxiv-mcp-server>=0.1.0` - arXiv MCP server implementation
  - `nest-asyncio>=1.5.0` - Async/sync event loop compatibility
  - `pytest-asyncio>=0.21.0` - Async testing support
  - `pytest-cov>=4.0.0` - Test coverage reporting
- ✅ **Environment configuration** - Updated `.env.example`
  - `USE_MCP_ARXIV` - Toggle MCP vs direct API (default: `false`)
  - `MCP_ARXIV_STORAGE_PATH` - MCP server storage location (default: `./data/mcp_papers/`)

**🧪 Testing & Diagnostics:**
- ✅ **MCP Test Suite** - 21 comprehensive tests in `tests/test_mcp_arxiv_client.py`
  - Async/sync wrapper tests for all client methods
  - MCP tool call mocking and response parsing
  - Error handling and fallback mechanisms
  - PDF caching and storage path management
- ✅ **Diagnostic Script** - New `test_mcp_diagnostic.py` for troubleshooting
  - Environment configuration validation
  - Storage directory verification
  - MCP tool discovery and listing
  - Search and download functionality testing
  - File system state inspection

**📚 Documentation:**
- ✅ **MCP Integration Guide** - Comprehensive documentation added
  - `MCP_FIX_DOCUMENTATION.md` - Root cause analysis, architecture, troubleshooting
  - `MCP_FIX_SUMMARY.md` - Quick reference for the MCP download fix
  - Updated `CLAUDE.md` - Developer documentation with MCP integration details
  - Updated README - MCP setup instructions and configuration guide

### Version 2.1 - November 2025

**🎨 Enhanced User Experience:**
- ✅ **Progressive Papers Tab** - Real-time updates as papers are analyzed
  - Papers table "paints" progressively showing status: ⏸️ Pending → ⏳ Analyzing → ✅ Complete / ⚠️ Failed
  - Analysis HTML updates incrementally as each paper completes
  - Synthesis and Citations populate after all analyses finish
  - Smooth streaming experience using Python generators (`yield`)
- ✅ **Clickable PDF Links** - Papers tab links now HTML-enabled
  - Link column renders as markdown for clickable "View PDF" links
  - Direct access to arXiv PDFs from results table
- ✅ **Smart Confidence Filtering** - Improved result quality
  - Papers with 0% confidence (failed analyses) excluded from synthesis and citations
  - Failed papers remain visible in Papers tab with ⚠️ Failed status
  - Prevents low-quality analyses from contaminating final output
  - Graceful handling when all analyses fail

**💰 Configurable Pricing System (November 5, 2025):**
- ✅ **Dynamic pricing configuration** - No code changes needed when switching models
  - New `config/pricing.json` with pricing for phi-4-multimodal-instruct, gpt-4o-mini, gpt-4o
  - New `utils/config.py` with PricingConfig class
  - Support for multiple embedding models (text-embedding-3-small, text-embedding-3-large)
- ✅ **Environment variable overrides** - Easy testing and custom pricing
  - `PRICING_INPUT_PER_1M` - Override input token pricing for all models
  - `PRICING_OUTPUT_PER_1M` - Override output token pricing for all models
  - `PRICING_EMBEDDING_PER_1M` - Override embedding token pricing
- ✅ **Thread-safe token tracking** - Accurate counts in parallel processing
  - threading.Lock in AnalyzerAgent for concurrent token accumulation
  - Model names (llm_model, embedding_model) tracked in state
  - Embedding token estimation (~300 tokens per chunk average)

**🔧 Critical Bug Fixes:**
- ✅ **Stats tab fix (November 5, 2025)** - Fixed zeros displaying in Stats tab
  - Processing time now calculated from start_time (was showing 0.0s)
  - Token usage tracked across all agents (was showing zeros)
  - Cost estimates calculated with accurate token counts (was showing $0.00)
  - Thread-safe token accumulation in parallel processing
- ✅ **LLM Response Normalization** - Prevents Pydantic validation errors
  - Handles cases where LLM returns strings for array fields
  - Auto-converts "Not available" strings to proper list format
  - Robust handling of JSON type mismatches

**🏗️ Architecture Improvements:**
- ✅ **Streaming Workflow** - Replaced LangGraph with generator-based streaming
  - Better user feedback with progressive updates
  - More control over workflow execution
  - Improved error handling and recovery
- ✅ **State Management** - Enhanced data flow
  - `filtered_papers` and `filtered_analyses` for quality control
  - `model_desc` dictionary for model metadata
  - Cleaner separation of display vs. processing data

### Version 2.0 - October 2025

> **Note**: LangGraph was later replaced in v2.1 with a generator-based streaming workflow for better real-time user feedback and progressive UI updates.

**🏗️ Architecture Overhaul:**
- ✅ **LangGraph integration** - Professional workflow orchestration framework
- ✅ **Conditional routing** - Skips downstream agents when no papers found
- ✅ **Parallel processing** - Analyze 4 papers simultaneously (ThreadPoolExecutor)
- ✅ **Circuit breaker** - Stops after 2 consecutive failures

**⚡ Performance Improvements (3x Faster):**
- ✅ **Timeout management** - 60s analyzer, 90s synthesis
- ✅ **Token limits** - max_tokens 1500/2500 prevents slow responses
- ✅ **Optimized prompts** - Reduced metadata overhead (-10% tokens)
- ✅ **Result**: 2-3 min for 5 papers (was 5-10 min)

**🎨 UX Enhancements:**
- ✅ **Paper titles in Synthesis** - Shows "Title (arXiv ID)" instead of just IDs
- ✅ **Confidence for contradictions** - Displayed alongside consensus points
- ✅ **Graceful error messages** - Friendly DataFrame with actionable suggestions
- ✅ **Enhanced error UI** - Contextual icons and helpful tips

**🐛 Critical Bug Fixes:**
- ✅ **Cache mutation fix** - Deep copy prevents repeated query errors
- ✅ **No papers crash fix** - Graceful termination instead of NoneType error
- ✅ **Validation fix** - Removed processing_time from initial state

**📊 Observability:**
- ✅ **Timestamp logging** - Added to all 10 modules for better debugging

**🔧 Bug Fix (October 28, 2025):**
- ✅ **Circuit breaker fix** - Reset counter per batch to prevent cascade failures in parallel processing
  - Fixed issue where 2 failures in one batch caused all papers in next batch to skip
  - Each batch now gets fresh attempt regardless of previous batch failures
  - Maintains failure tracking within batch without cross-batch contamination

### Previous Updates (Early 2025)
- ✅ Fixed datetime JSON serialization error (added `mode='json'` to `model_dump()`)
- ✅ Fixed AttributeError when formatting cached results (separated cache data from output data)
- ✅ Fixed Pydantic V2 deprecation warning (replaced `.dict()` with `.model_dump()`)
- ✅ Added GitHub Actions workflow for automated deployment to Hugging Face Spaces
- ✅ Fixed JSON serialization error in semantic cache (Pydantic model conversion)
- ✅ Added comprehensive test suite for Analyzer Agent (18 tests)
- ✅ Added pytest and pytest-mock to dependencies
- ✅ Enhanced error handling and logging across agents
- ✅ Updated documentation with testing guidelines
- ✅ Improved type safety with Pydantic schemas
- ✅ Added QUICKSTART.md for quick setup

### Completed Features (Recent)
- [x] MCP (Model Context Protocol) integration with arXiv
- [x] MCP diagnostic tools and comprehensive testing
- [x] Automatic fallback mechanism for reliable downloads
- [x] Configurable pricing system
- [x] Progressive UI with streaming results
- [x] Smart quality filtering (0% confidence exclusion)

### Coming Soon
- [ ] Tests for Retriever, Synthesis, and Citation agents
- [ ] Integration tests for full workflow
- [ ] CI/CD pipeline with automated testing (GitHub Actions already set up for deployment)
- [ ] Docker containerization improvements
- [ ] Performance benchmarking suite
- [ ] Pre-commit hooks for code quality
- [ ] Additional MCP server support (beyond arXiv)

---

**Built with ❤️ using Azure OpenAI, ChromaDB, LangChain, and Gradio**
