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

- **Automated Paper Retrieval**: Search and download papers from arXiv
- **RAG-Based Analysis**: Extract methodology, findings, conclusions, and limitations using retrieval-augmented generation
- **Cross-Paper Synthesis**: Identify consensus points, contradictions, and research gaps
- **Citation Management**: Generate proper APA-style citations with source validation
- **Semantic Caching**: Optimize costs by caching similar queries
- **Deterministic Outputs**: Temperature=0 and structured outputs for reproducibility

## Architecture

### Agent Workflow

```
User Query → Retriever Agent → Analyzer Agent → Synthesis Agent → Citation Agent → User
```

### 4 Specialized Agents

1. **Retriever Agent**
   - Queries arXiv API based on user input
   - Downloads and parses PDF papers
   - Extracts metadata (title, authors, abstract, publication date)
   - Chunks papers into 500-token segments with 50-token overlap

2. **Analyzer Agent**
   - Processes individual papers using RAG context
   - Extracts: methodology, key findings, conclusions, limitations
   - Identifies main contributions
   - Outputs structured JSON with citations

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
- **Agent Framework**: Custom multi-agent orchestration
- **UI**: Gradio 5.49.1 with tabbed interface
- **Data Source**: arXiv API
- **Testing**: pytest with comprehensive test suite
- **Type Safety**: Pydantic V2 schemas for validation

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
- `AZURE_OPENAI_DEPLOYMENT_NAME`: Your deployment name (e.g., gpt-4o-mini)
- `AZURE_OPENAI_API_VERSION`: API version (optional, defaults in code)

Optional:
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`: Custom embedding model deployment name

4. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:7860`

## Usage

1. **Enter Research Question**: Type your research question in the text box
2. **Select Category**: Choose an arXiv category or leave as "All"
3. **Set Number of Papers**: Use the slider to select 1-20 papers
4. **Click Analyze**: The system will process your request
5. **View Results**: Explore the four output tabs:
   - **Papers**: Table of retrieved papers with links
   - **Analysis**: Detailed analysis of each paper
   - **Synthesis**: Executive summary with consensus and contradictions
   - **Citations**: APA-formatted references
   - **Stats**: Processing statistics and cost estimates

## Project Structure

```
Multi-Agent-Research-Paper-Analysis-System/
├── app.py                          # Main Gradio application
├── requirements.txt                # Python dependencies
├── README.md                       # This file - full documentation
├── QUICKSTART.md                   # Quick setup guide (5 minutes)
├── .env.example                    # Environment variable template
├── .gitignore                      # Git ignore rules
├── agents/
│   ├── __init__.py
│   ├── retriever.py               # Paper retrieval & chunking
│   ├── analyzer.py                # Individual paper analysis
│   ├── synthesis.py               # Cross-paper synthesis
│   └── citation.py                # Citation validation & formatting
├── rag/
│   ├── __init__.py
│   ├── vector_store.py            # ChromaDB vector storage
│   ├── embeddings.py              # Azure OpenAI text embeddings
│   └── retrieval.py               # RAG retrieval & context formatting
├── utils/
│   ├── __init__.py
│   ├── arxiv_client.py            # arXiv API wrapper
│   ├── pdf_processor.py           # PDF parsing & chunking
│   ├── cache.py                   # Semantic caching layer
│   └── schemas.py                 # Pydantic data models
├── tests/
│   ├── __init__.py
│   └── test_analyzer.py           # Unit tests for analyzer agent
└── data/                           # Created at runtime
    ├── papers/                     # Downloaded PDFs (cached)
    └── chroma_db/                  # Vector store persistence
```

## Key Features

### Deterministic Output Strategy

The system implements multiple techniques to minimize hallucinations:

1. **Temperature=0**: All Azure OpenAI calls use temperature=0
2. **Structured Outputs**: JSON mode for agent responses with strict schemas
3. **RAG Grounding**: Every response includes retrieved chunk IDs
4. **Source Validation**: Cross-reference all claims with original text
5. **Semantic Caching**: Hash query embeddings, return cached results for cosine similarity >0.95
6. **Confidence Scores**: Return uncertainty metrics with each response

### Cost Optimization

- Request batching for embeddings
- Cached embeddings in ChromaDB (don't re-embed same papers)
- Token usage logging per request
- Semantic caching for repeated queries with Pydantic V2 model serialization (`mode='json'`)
- JSON-compatible cache storage with automatic datetime conversion
- Target: <$0.50 per analysis session

### Error Handling

- Graceful fallback if arXiv API is down
- Handle PDF parsing failures (some papers may be scanned images)
- Timeout protection for long-running analyses
- User-friendly error messages in Gradio UI
- Comprehensive error logging for debugging

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
- **unittest.mock**: Mocking external dependencies (Azure OpenAI, RAG components)
- **Pydantic models**: Type-safe test data structures
- **Isolated testing**: No external API calls in unit tests

## Performance

- **Speed**: Complete 5-paper analysis in <2 minutes
- **Cost**: <$0.50 per analysis session
- **Accuracy**: Deterministic outputs with confidence scores
- **Scalability**: Handles 1-20 papers per query

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

### Latest Updates (2025)
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

### Coming Soon
- [ ] Tests for Retriever, Synthesis, and Citation agents
- [ ] Integration tests for full workflow
- [ ] CI/CD pipeline with automated testing (GitHub Actions already set up for deployment)
- [ ] Docker containerization
- [ ] Performance benchmarking suite
- [ ] Pre-commit hooks for code quality

---

**Built with ❤️ using Azure OpenAI, ChromaDB, LangChain, and Gradio**
