# AGENTS.md

**A Technical Deep-Dive into Multi-Agent Architecture**

This document provides a comprehensive technical reference for understanding, building, and debugging agents in the Multi-Agent Research Paper Analysis System. It focuses on agent design patterns, state transformations, error handling, observability, and extensibility.

---

## Table of Contents

1. [Introduction](#1-introduction)
   - [The 4-Agent Sequential Pipeline](#the-4-agent-sequential-pipeline)
   - [Agent Design Philosophy](#agent-design-philosophy)
   - [How Agents Differ from Traditional Microservices](#how-agents-differ-from-traditional-microservices)

2. [Agent Architecture Fundamentals](#2-agent-architecture-fundamentals)
   - [The Common Agent Interface](#the-common-agent-interface)
   - [State Transformation Contract](#state-transformation-contract)
   - [Dependency Injection Pattern](#dependency-injection-pattern)
   - [LangGraph Integration Through Node Wrappers](#langgraph-integration-through-node-wrappers)

3. [Individual Agent Deep Dives](#3-individual-agent-deep-dives)
   - [RetrieverAgent](#retrieveragent)
   - [AnalyzerAgent](#analyzeragent)
   - [SynthesisAgent](#synthesisagent)
   - [CitationAgent](#citationagent)

4. [Cross-Cutting Patterns](#4-cross-cutting-patterns)
   - [State Management](#state-management)
   - [Error Handling Philosophy](#error-handling-philosophy)
   - [Observability Integration](#observability-integration)
   - [Performance Optimizations](#performance-optimizations)

5. [Workflow Orchestration](#5-workflow-orchestration)
   - [LangGraph Workflow Structure](#langgraph-workflow-structure)
   - [Node Wrapper Pattern](#node-wrapper-pattern)
   - [Conditional Routing](#conditional-routing)
   - [Checkpointing and State Persistence](#checkpointing-and-state-persistence)

6. [Building New Agents](#6-building-new-agents)
   - [Step-by-Step Development Guide](#step-by-step-development-guide)
   - [Minimal Agent Template](#minimal-agent-template)
   - [Testing Patterns](#testing-patterns)
   - [Best Practices Checklist](#best-practices-checklist)

7. [Agent Comparison Reference](#7-agent-comparison-reference)

8. [Troubleshooting and Debugging](#8-troubleshooting-and-debugging)
   - [Common Issues and Solutions](#common-issues-and-solutions)
   - [Reading LangFuse Traces](#reading-langfuse-traces)
   - [State Inspection Techniques](#state-inspection-techniques)
   - [Log Analysis Patterns](#log-analysis-patterns)

---

## 1. Introduction

### The 4-Agent Sequential Pipeline

The Multi-Agent Research Paper Analysis System implements a **sequential pipeline** of four specialized agents orchestrated by LangGraph:

```
User Query → Retriever → Analyzer → Filter → Synthesis → Citation → Output
                ↓          ↓         ↓         ↓           ↓
            [LangFuse Tracing for All Nodes]
```

Each agent:
- Operates on a **shared state dictionary** that flows through the pipeline
- Performs a **specialized task** (retrieval, analysis, synthesis, citation)
- Transforms the state by **reading inputs** and **writing outputs**
- **Never blocks the workflow** - returns partial results on failure
- Is **automatically traced** by LangFuse for observability

### Agent Design Philosophy

The architecture follows these core principles:

**1. Pure Functions, Not Stateful Services**
- Agents are pure functions: `run(state) -> state`
- No instance state between invocations
- Deterministic outputs for same inputs (temperature=0)

**2. Resilience Through Graceful Degradation**
- Never raise exceptions from `run()`
- Return partial results with degraded confidence scores
- Append errors to state for debugging
- Circuit breakers prevent cascading failures

**3. Observability by Design**
- All agents decorated with `@observe` for automatic tracing
- Three-tier tracing: node-level, agent-level, LLM-level
- Session IDs track multi-turn conversations
- Token usage accumulated for cost monitoring

**4. Separation of Concerns**
- Agent logic: Domain-specific transformations
- Node wrappers: Orchestration concerns (tracing, error handling, logging)
- Workflow graph: Routing and conditional execution

**5. Explicit Contracts**
- Pydantic schemas validate all data structures
- AgentState TypedDict defines state shape
- msgpack serialization constraints enforced

### How Agents Differ from Traditional Microservices

| Aspect | Traditional Microservices | Our Agents |
|--------|--------------------------|------------|
| **Communication** | HTTP/gRPC between services | Shared state dictionary |
| **State** | Each service has database | Stateless, state flows through pipeline |
| **Failure Handling** | Retry with exponential backoff | Graceful degradation with partial results |
| **Orchestration** | Service mesh, API gateway | LangGraph with conditional routing |
| **Observability** | Distributed tracing (Jaeger, Zipkin) | LangFuse with automatic instrumentation |
| **Deployment** | Independent containers | Single process, modular architecture |
| **Scaling** | Horizontal scaling | Parallel processing within agents (ThreadPoolExecutor) |

**Key Insight**: Agents are lightweight, composable functions orchestrated by a workflow graph, not heavyweight network services.

---

## 2. Agent Architecture Fundamentals

### The Common Agent Interface

All agents implement a consistent interface:

```python
from typing import Dict, Any

class BaseAgent:
    """Base interface for all agents in the system."""

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform the workflow state.

        Args:
            state: Current workflow state (AgentState TypedDict)

        Returns:
            Updated state with new keys/values added

        Raises:
            Never raises - catches all exceptions and appends to state["errors"]
        """
        raise NotImplementedError
```

**Critical Contract Rules:**

1. **Never Mutate State In-Place**: Always return a new/modified dictionary
2. **Never Raise Exceptions**: Catch all exceptions, append to `state["errors"]`
3. **Always Return State**: Even on failure, return state with partial results
4. **Use Pydantic Models**: Validate outputs before adding to state

### State Transformation Contract

Agents follow a clear input/output pattern:

```python
# Input: Read specific keys from state
query = state.get("query")
papers = state.get("papers", [])
category = state.get("category")

# Processing: Transform data using dependencies
results = self.process(query, papers)

# Output: Write new keys to state (never overwrite critical keys)
state["analyses"] = results
state["token_usage"]["input_tokens"] += prompt_tokens
state["errors"].append(error_message)  # Only on error

# Return: Modified state
return state
```

**State Flow Example:**

```python
# Initial state (from user input)
{
    "query": "What are recent advances in transformer architectures?",
    "category": "cs.AI",
    "num_papers": 5,
    "errors": [],
    "token_usage": {"input_tokens": 0, "output_tokens": 0, "embedding_tokens": 0}
}

# After RetrieverAgent
{
    # ... original keys ...
    "papers": [Paper(...), Paper(...), ...],  # NEW
    "chunks": [PaperChunk(...), ...],          # NEW
    "token_usage": {"embedding_tokens": 15000} # UPDATED
}

# After AnalyzerAgent
{
    # ... all previous keys ...
    "analyses": [Analysis(...), Analysis(...), ...],  # NEW
    "token_usage": {
        "input_tokens": 12000,   # UPDATED
        "output_tokens": 3000,   # UPDATED
        "embedding_tokens": 15000
    }
}

# And so on through the pipeline...
```

### Dependency Injection Pattern

Agents receive their dependencies via constructor injection:

```python
# agents/analyzer.py
class AnalyzerAgent:
    """Analyzes individual papers using RAG."""

    def __init__(
        self,
        rag_retriever,           # Injected dependency
        azure_openai_config: Dict[str, str],
        max_workers: int = 4,
        timeout: int = 60
    ):
        self.rag_retriever = rag_retriever
        self.client = self._initialize_client(azure_openai_config)
        self.max_workers = max_workers
        self.timeout = timeout
        self.consecutive_failures = 0
        self.max_consecutive_failures = 2
        self.token_lock = threading.Lock()
```

**Benefits:**

- **Testability**: Easy to mock dependencies in tests
- **Flexibility**: Different implementations can be injected (e.g., ArxivClient vs FastMCPArxivClient)
- **Clarity**: Dependencies are explicit in constructor signature

**Initialization in app.py:**

```python
# app.py:298-345
rag_retriever = RAGRetriever(vector_store=vector_store, embedding_generator=embedding_generator)
analyzer_agent = AnalyzerAgent(rag_retriever=rag_retriever, azure_openai_config=azure_config)
synthesis_agent = SynthesisAgent(rag_retriever=rag_retriever, azure_openai_config=azure_config)
citation_agent = CitationAgent(rag_retriever=rag_retriever)
```

### LangGraph Integration Through Node Wrappers

Agents integrate with LangGraph through a **node wrapper pattern**:

```python
# orchestration/nodes.py
from langfuse.decorators import observe

@observe(name="analyzer_agent", as_type="span")
def analyzer_node(state: AgentState, analyzer_agent) -> AgentState:
    """
    Node wrapper for AnalyzerAgent.

    Responsibilities:
    - LangFuse tracing (via @observe decorator)
    - Structured logging
    - Error handling (catch exceptions)
    - State transformation delegation
    """
    logger.info("Starting analyzer agent...")

    try:
        # Delegate to agent's run() method
        updated_state = analyzer_agent.run(state)

        logger.info(f"Analyzer completed. Analyses: {len(updated_state.get('analyses', []))}")
        return updated_state

    except Exception as e:
        logger.error(f"Analyzer node failed: {str(e)}", exc_info=True)
        state["errors"].append(f"Analyzer failed: {str(e)}")
        return state
```

**Workflow Graph Definition:**

```python
# orchestration/workflow_graph.py:75-88
from langgraph.graph import StateGraph, END

workflow = StateGraph(AgentState)

# Add nodes (lambda binds agent instance to node wrapper)
workflow.add_node("retriever", lambda state: retriever_node(state, retriever_agent))
workflow.add_node("analyzer", lambda state: analyzer_node(state, analyzer_agent))
workflow.add_node("filter", filter_node)
workflow.add_node("synthesis", lambda state: synthesis_node(state, synthesis_agent))
workflow.add_node("citation", lambda state: citation_node(state, citation_agent))

# Define edges (execution flow)
workflow.set_entry_point("retriever")
workflow.add_edge("analyzer", "filter")
workflow.add_edge("synthesis", "citation")
workflow.add_edge("citation", END)
```

**Why Node Wrappers?**

1. **Separation of Concerns**: Agent logic stays pure, orchestration concerns in wrapper
2. **Automatic Tracing**: `@observe` decorator applies to all agents uniformly
3. **Centralized Error Handling**: Catch-all exception handling prevents workflow crashes
4. **Consistent Logging**: Structured logs with same format across all agents

---

## 3. Individual Agent Deep Dives

### RetrieverAgent

**File**: `agents/retriever.py`

**Core Responsibilities:**
1. Search arXiv for papers matching user query and category
2. Download PDFs via configurable clients (Direct API, FastMCP)
3. Process PDFs into 500-token chunks with 50-token overlap
4. Generate embeddings using Azure OpenAI text-embedding-3-small
5. Store chunks in ChromaDB vector database

**State Transformations:**

```python
# Input Keys
query = state.get("query")           # str: "What are recent advances in transformers?"
category = state.get("category")     # Optional[str]: "cs.AI"
num_papers = state.get("num_papers", 5)  # int: 5

# Output Keys (added to state)
state["papers"] = [Paper(...), ...]        # List[Paper]: Paper metadata
state["chunks"] = [PaperChunk(...), ...]   # List[PaperChunk]: Text chunks
state["token_usage"]["embedding_tokens"] = 15000  # Estimated tokens
state["errors"].append("Failed to download paper X")  # On partial failure
```

**Dependencies:**

```python
def __init__(
    self,
    arxiv_client,        # ArxivClient | FastMCPArxivClient
    pdf_processor,       # PDFProcessor
    embedding_generator, # EmbeddingGenerator
    vector_store,        # VectorStore (ChromaDB)
    fallback_client=None # Optional fallback client
):
```

**Key Design Pattern: Two-Tier Fallback**

```python
# agents/retriever.py:69-97
def _search_with_fallback(
    self,
    query: str,
    max_results: int,
    category: Optional[str] = None
) -> List[Paper]:
    """Search with automatic fallback to secondary client."""

    # Try primary client (e.g., FastMCP)
    try:
        logger.info(f"Searching with primary client: {type(self.arxiv_client).__name__}")
        papers = self.arxiv_client.search_papers(
            query=query,
            max_results=max_results,
            category=category
        )
        if papers:
            return papers
        logger.warning("Primary client returned no results, trying fallback...")

    except Exception as e:
        logger.warning(f"Primary client failed: {str(e)}, trying fallback...")

    # Fallback to secondary client (e.g., Direct API)
    if self.fallback_client:
        try:
            logger.info(f"Searching with fallback client: {type(self.fallback_client).__name__}")
            return self.fallback_client.search_papers(
                query=query,
                max_results=max_results,
                category=category
            )
        except Exception as e:
            logger.error(f"Fallback client also failed: {str(e)}")
            return []

    return []
```

**Why This Pattern?**
- **Resilience**: MCP servers may be unavailable, fallback ensures retrieval succeeds
- **Transparency**: Logs show which client succeeded
- **Zero User Impact**: Fallback is automatic and invisible

**Key Design Pattern: Data Validation Filtering**

```python
# agents/retriever.py:198-242
def _validate_papers(self, papers: List[Paper]) -> List[Paper]:
    """Validate and filter papers to ensure Pydantic compliance."""

    valid_papers = []
    for paper in papers:
        try:
            # Ensure all list fields are actually lists
            if not isinstance(paper.authors, list):
                paper.authors = [paper.authors] if paper.authors else []
            if not isinstance(paper.categories, list):
                paper.categories = [paper.categories] if paper.categories else []

            # Re-validate with Pydantic
            validated_paper = Paper(**paper.model_dump())
            valid_papers.append(validated_paper)

        except Exception as e:
            logger.warning(f"Skipping invalid paper {paper.arxiv_id}: {str(e)}")
            continue

    logger.info(f"Validated {len(valid_papers)}/{len(papers)} papers")
    return valid_papers
```

**Why This Pattern?**
- **Defensive Programming**: MCP servers may return malformed data
- **Partial Success**: Continue with valid papers instead of failing completely
- **Type Safety**: Ensures downstream agents can rely on Pydantic schemas

**Error Handling Strategy:**

```python
# agents/retriever.py:249-302
@observe(name="retriever_agent_run", as_type="generation")
def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # Step 1: Search (with fallback)
        papers = self._search_with_fallback(query, max_results, category)
        if not papers:
            state["errors"].append("No papers found for query")
            return state  # Early return, no exception

        # Step 2: Download PDFs (continue on partial failures)
        for paper in papers:
            try:
                pdf_path = self._download_with_fallback(paper)
                # Process PDF...
            except Exception as e:
                logger.warning(f"Failed to process {paper.arxiv_id}: {str(e)}")
                continue  # Skip this paper, process others

        # Step 3: Generate embeddings (batch operation)
        try:
            embeddings = self.embedding_generator.generate_batch(chunks)
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            state["errors"].append("Embedding generation failed")
            return state  # Return papers/chunks without embeddings

        # Success: Return enriched state
        state["papers"] = papers
        state["chunks"] = chunks
        state["token_usage"]["embedding_tokens"] = len(chunks) * 300
        return state

    except Exception as e:
        logger.error(f"Retriever agent failed: {str(e)}", exc_info=True)
        state["errors"].append(f"Retriever failed: {str(e)}")
        return state  # Never raise
```

**Observability Integration:**

```python
@observe(name="retriever_agent_run", as_type="generation")
```

- **Type**: `"generation"` (includes embedding generation)
- **Trace Data**: Search query, paper count, chunk count, embedding tokens
- **LangFuse View**: Shows retrieval duration, embedding API calls

**Critical File Paths:**
- `agents/retriever.py:69-97` - Fallback search logic
- `agents/retriever.py:100-157` - Fallback download logic
- `agents/retriever.py:198-242` - Paper validation
- `agents/retriever.py:249-302` - Main `run()` method

---

### AnalyzerAgent

**File**: `agents/analyzer.py`

**Core Responsibilities:**
1. Analyze each paper individually using RAG context
2. Execute 4 broad queries per paper for comprehensive coverage
3. Call Azure OpenAI (GPT-4o-mini) with temperature=0 for deterministic JSON
4. Extract methodology, findings, conclusions, limitations, contributions
5. Calculate confidence scores based on context completeness

**State Transformations:**

```python
# Input Keys
papers = state.get("papers", [])  # List[Paper] from RetrieverAgent

# Output Keys (added to state)
state["analyses"] = [Analysis(...), ...]  # List[Analysis]: One per paper
state["token_usage"]["input_tokens"] += 12000   # Cumulative prompt tokens
state["token_usage"]["output_tokens"] += 3000   # Cumulative completion tokens
state["errors"].append("Failed to analyze paper X")  # On failure
```

**Dependencies:**

```python
def __init__(
    self,
    rag_retriever,       # RAGRetriever: Semantic search + context formatting
    azure_openai_config: Dict[str, str],
    max_workers: int = 4,   # Parallel analysis threads
    timeout: int = 60       # LLM call timeout
):
```

**Key Design Pattern: Parallel Processing with Circuit Breaker**

```python
# agents/analyzer.py:333-359
def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
    papers = state.get("papers", [])
    analyses = []

    # Reset circuit breaker
    self.consecutive_failures = 0

    # Parallel processing with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
        future_to_paper = {
            executor.submit(self.analyze_paper, paper): paper
            for paper in papers
        }

        for future in as_completed(future_to_paper):
            # Circuit breaker check
            if self.consecutive_failures >= self.max_consecutive_failures:
                logger.error(f"Circuit breaker triggered after {self.consecutive_failures} failures")
                break

            paper = future_to_paper[future]
            try:
                analysis = future.result()

                if analysis.confidence_score > 0:
                    analyses.append(analysis)
                    self.consecutive_failures = 0  # Reset on success
                else:
                    self.consecutive_failures += 1

            except Exception as e:
                logger.error(f"Analysis failed for {paper.arxiv_id}: {str(e)}")
                self.consecutive_failures += 1

    state["analyses"] = analyses
    return state
```

**Why This Pattern?**
- **Throughput**: Analyzes 4 papers concurrently (max_workers=4)
- **Circuit Breaker**: Stops after 2 consecutive failures (prevents wasted API calls)
- **Thread Safety**: `self.token_lock` protects shared token counter
- **Graceful Degradation**: Partial analyses returned even if some papers fail

**Key Design Pattern: Comprehensive RAG Queries**

```python
# agents/analyzer.py:208-252
def _retrieve_comprehensive_context(self, paper: Paper, top_k: int = 10) -> Tuple[str, List[str]]:
    """
    Retrieve chunks using multiple broad queries to ensure full coverage.
    """
    # 4 broad queries to cover different aspects
    queries = [
        "methodology approach methods experimental setup techniques",
        "results findings data experiments performance evaluation",
        "conclusions contributions implications significance impact",
        "limitations future work challenges open problems directions"
    ]

    all_chunks = []
    all_chunk_ids = []

    # Retrieve top_k/4 chunks per query (10 total chunks by default)
    chunks_per_query = max(1, top_k // len(queries))

    for query in queries:
        result = self.rag_retriever.retrieve(
            query=query,
            top_k=chunks_per_query,
            paper_ids=[paper.arxiv_id]  # Filter to this paper only
        )
        all_chunks.extend(result["chunks"])
        all_chunk_ids.extend(result["chunk_ids"])

    # Deduplicate by chunk_id
    seen = set()
    unique_chunks = []
    unique_ids = []

    for chunk, chunk_id in zip(all_chunks, all_chunk_ids):
        if chunk_id not in seen:
            seen.add(chunk_id)
            unique_chunks.append(chunk)
            unique_ids.append(chunk_id)

    # Format context with metadata
    context = self.rag_retriever.format_context(unique_chunks)
    return context, unique_ids
```

**Why This Pattern?**
- **Comprehensive Coverage**: Single query misses sections (e.g., "methods" misses conclusions)
- **Semantic Diversity**: Broad queries capture different aspects of the paper
- **Deduplication**: Prevents redundant chunks from multiple queries
- **Filtered Search**: `paper_ids` ensures we only retrieve from current paper

**Key Design Pattern: LLM Response Normalization**

```python
# agents/analyzer.py:107-178
def _normalize_analysis_response(self, data: dict) -> dict:
    """
    Normalize malformed LLM responses to match Pydantic schema.

    Common issues:
    - Nested lists: ["finding 1", ["finding 2", "finding 3"]]
    - None values in lists: [None, "valid finding"]
    - Mixed types: [123, "text", {"key": "value"}]
    """
    def flatten_and_clean(value):
        """Recursively flatten nested lists and convert to strings."""
        if value is None:
            return ""
        elif isinstance(value, list):
            flattened = []
            for item in value:
                cleaned = flatten_and_clean(item)
                if isinstance(cleaned, list):
                    flattened.extend(cleaned)
                elif cleaned:  # Skip empty strings
                    flattened.append(cleaned)
            return flattened
        elif isinstance(value, (dict, int, float, bool)):
            return str(value)
        else:
            return str(value)

    # Normalize all list fields
    normalized = {}
    list_fields = ["methodology", "key_findings", "conclusions", "limitations", "contributions"]

    for field in list_fields:
        if field in data:
            cleaned = flatten_and_clean(data[field])
            normalized[field] = cleaned if isinstance(cleaned, list) else [cleaned]
        else:
            normalized[field] = []

    # Preserve scalar fields
    normalized["confidence_score"] = float(data.get("confidence_score", 0.0))
    normalized["arxiv_id"] = data.get("arxiv_id", "")
    normalized["title"] = data.get("title", "")

    return normalized
```

**Why This Pattern?**
- **LLM Hallucinations**: GPT-4o-mini occasionally returns malformed JSON
- **Defensive Parsing**: Prevents Pydantic validation errors
- **Data Salvage**: Extracts valid data even from malformed responses

**Error Handling Strategy:**

```python
# agents/analyzer.py:260-325
def analyze_paper(self, paper: Paper) -> Analysis:
    """Analyze a single paper (called by ThreadPoolExecutor)."""
    try:
        # Step 1: Retrieve context via RAG
        context, chunk_ids = self._retrieve_comprehensive_context(paper)

        # Step 2: Call LLM with structured prompt
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": "You are a research paper analyzer..."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,  # Deterministic
            response_format={"type": "json_object"},  # Force JSON
            max_tokens=2000,
            timeout=self.timeout
        )

        # Step 3: Parse and normalize response
        data = json.loads(response.choices[0].message.content)
        normalized = self._normalize_analysis_response(data)

        # Step 4: Create Pydantic model
        analysis = Analysis(**normalized)

        # Step 5: Track tokens (thread-safe)
        with self.token_lock:
            self.total_input_tokens += response.usage.prompt_tokens
            self.total_output_tokens += response.usage.completion_tokens

        return analysis

    except Exception as e:
        logger.error(f"Failed to analyze {paper.arxiv_id}: {str(e)}", exc_info=True)
        # Return minimal analysis with confidence=0.0
        return Analysis(
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            methodology=[],
            key_findings=[],
            conclusions=[],
            limitations=[],
            contributions=[],
            confidence_score=0.0
        )
```

**Observability Integration:**

```python
@observe(name="analyzer_agent_run", as_type="generation")
```

- **Type**: `"generation"` (LLM-heavy workload)
- **Trace Data**: Paper count, analysis count, token usage, parallel execution
- **LangFuse View**: Shows individual LLM calls via `langfuse-openai` instrumentation

**Critical File Paths:**
- `agents/analyzer.py:107-178` - Response normalization
- `agents/analyzer.py:208-252` - Comprehensive RAG queries
- `agents/analyzer.py:260-325` - Single paper analysis
- `agents/analyzer.py:333-359` - Parallel processing with circuit breaker

---

### SynthesisAgent

**File**: `agents/synthesis.py`

**Core Responsibilities:**
1. Compare findings across all analyzed papers
2. Identify consensus points (where papers agree)
3. Identify contradictions (where papers disagree)
4. Identify research gaps (what's missing)
5. Generate executive summary addressing user's original query

**State Transformations:**

```python
# Input Keys
papers = state.get("papers", [])        # List[Paper]
analyses = state.get("analyses", [])    # List[Analysis] from AnalyzerAgent
query = state.get("query")              # Original user question

# Output Keys (added to state)
state["synthesis"] = SynthesisResult(
    consensus_points=[ConsensusPoint(...), ...],
    contradictions=[Contradiction(...), ...],
    research_gaps=["Gap 1", "Gap 2", ...],
    summary="Executive summary addressing user query...",
    papers_analyzed=["arxiv_id_1", "arxiv_id_2", ...],
    confidence_score=0.85
)
state["token_usage"]["input_tokens"] += 8000
state["token_usage"]["output_tokens"] += 2000
```

**Dependencies:**

```python
def __init__(
    self,
    rag_retriever,       # RAGRetriever (passed but not actively used)
    azure_openai_config: Dict[str, str],
    timeout: int = 90    # Longer timeout for synthesis (more complex task)
):
```

**Key Design Pattern: Cross-Paper Synthesis Prompt**

```python
# agents/synthesis.py:54-133
def _create_synthesis_prompt(
    self,
    query: str,
    papers: List[Paper],
    analyses: List[Analysis]
) -> str:
    """
    Create structured prompt for cross-paper synthesis.
    """
    # Format all analyses into structured summaries
    paper_summaries = []
    for paper, analysis in zip(papers, analyses):
        summary = f"""
[Paper {paper.arxiv_id}]
Title: {paper.title}
Authors: {', '.join(paper.authors[:3])}...
Published: {paper.published}

Methodology: {' | '.join(analysis.methodology[:3])}
Key Findings: {' | '.join(analysis.key_findings[:3])}
Conclusions: {' | '.join(analysis.conclusions[:2])}
Limitations: {' | '.join(analysis.limitations[:2])}
Contributions: {' | '.join(analysis.contributions[:2])}
"""
        paper_summaries.append(summary)

    # Synthesis prompt
    prompt = f"""
You are synthesizing findings from {len(papers)} research papers to answer this question:
"{query}"

# Paper Summaries
{chr(10).join(paper_summaries)}

# Task
Analyze the papers above and provide:

1. **Consensus Points**: What do multiple papers agree on?
   - For each consensus point, list supporting papers (use arxiv_id)
   - Provide evidence from the papers

2. **Contradictions**: Where do papers disagree or present conflicting findings?
   - Describe the contradiction clearly
   - List papers on each side (papers_a, papers_b)

3. **Research Gaps**: What questions remain unanswered? What future directions are suggested?

4. **Summary**: A concise executive summary (2-3 paragraphs) answering the user's original question

Return as JSON:
{{
    "consensus_points": [
        {{
            "point": "Description of consensus",
            "supporting_papers": ["arxiv_id_1", "arxiv_id_2"],
            "evidence": "Evidence from papers"
        }}
    ],
    "contradictions": [
        {{
            "description": "Description of contradiction",
            "papers_a": ["arxiv_id_1"],
            "papers_b": ["arxiv_id_2"],
            "context": "Additional context"
        }}
    ],
    "research_gaps": ["Gap 1", "Gap 2", ...],
    "summary": "Executive summary here...",
    "confidence_score": 0.85
}}
"""
    return prompt
```

**Why This Pattern?**
- **Structured Input**: LLM receives formatted summaries for all papers
- **Explicit Citations**: Requires grounding claims in specific papers
- **JSON Schema**: Forces structured output for Pydantic validation
- **Comprehensive Analysis**: Covers consensus, contradictions, gaps, and summary

**Key Design Pattern: Nested Data Normalization**

```python
# agents/synthesis.py:135-196
def _normalize_synthesis_response(self, data: dict) -> dict:
    """
    Normalize nested structures in synthesis response.

    Handles:
    - consensus_points[].supporting_papers (list)
    - consensus_points[].citations (list)
    - contradictions[].papers_a (list)
    - contradictions[].papers_b (list)
    - research_gaps (list)
    """
    def ensure_list_of_strings(value):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(item) for item in value if item]
        return [str(value)]

    normalized = {
        "consensus_points": [],
        "contradictions": [],
        "research_gaps": ensure_list_of_strings(data.get("research_gaps", [])),
        "summary": str(data.get("summary", "")),
        "confidence_score": float(data.get("confidence_score", 0.0))
    }

    # Normalize consensus points
    for cp in data.get("consensus_points", []):
        normalized["consensus_points"].append({
            "point": str(cp.get("point", "")),
            "supporting_papers": ensure_list_of_strings(cp.get("supporting_papers", [])),
            "evidence": str(cp.get("evidence", "")),
            "citations": ensure_list_of_strings(cp.get("citations", []))
        })

    # Normalize contradictions
    for contr in data.get("contradictions", []):
        normalized["contradictions"].append({
            "description": str(contr.get("description", "")),
            "papers_a": ensure_list_of_strings(contr.get("papers_a", [])),
            "papers_b": ensure_list_of_strings(contr.get("papers_b", [])),
            "context": str(contr.get("context", "")),
            "citations": ensure_list_of_strings(contr.get("citations", []))
        })

    return normalized
```

**Why This Pattern?**
- **Nested Schema Complexity**: ConsensusPoint and Contradiction have nested lists
- **LLM Inconsistency**: May return strings instead of lists for single items
- **Defensive Parsing**: Ensures Pydantic validation succeeds

**Error Handling Strategy:**

```python
# agents/synthesis.py:242-310
@observe(name="synthesis_agent_run", as_type="generation")
def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        papers = state.get("papers", [])
        analyses = state.get("analyses", [])
        query = state.get("query", "")

        # Handle paper count mismatch (defensive)
        if len(papers) != len(analyses):
            logger.warning(f"Paper count mismatch: {len(papers)} papers, {len(analyses)} analyses")
            # Use minimum length to avoid index errors
            min_len = min(len(papers), len(analyses))
            papers = papers[:min_len]
            analyses = analyses[:min_len]

        # Create synthesis prompt
        prompt = self._create_synthesis_prompt(query, papers, analyses)

        # Call LLM
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": "You are a research synthesis expert..."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=3000,
            timeout=self.timeout
        )

        # Parse and normalize
        data = json.loads(response.choices[0].message.content)
        normalized = self._normalize_synthesis_response(data)

        # Add paper IDs
        normalized["papers_analyzed"] = [p.arxiv_id for p in papers]

        # Create Pydantic model
        synthesis = SynthesisResult(**normalized)

        # Update state
        state["synthesis"] = synthesis
        state["token_usage"]["input_tokens"] += response.usage.prompt_tokens
        state["token_usage"]["output_tokens"] += response.usage.completion_tokens

        return state

    except Exception as e:
        logger.error(f"Synthesis failed: {str(e)}", exc_info=True)

        # Return minimal synthesis with confidence=0.0
        papers_analyzed = [p.arxiv_id for p in state.get("papers", [])]
        state["synthesis"] = SynthesisResult(
            consensus_points=[],
            contradictions=[],
            research_gaps=[],
            summary=f"Synthesis failed: {str(e)}",
            papers_analyzed=papers_analyzed,
            confidence_score=0.0
        )
        state["errors"].append(f"Synthesis failed: {str(e)}")
        return state
```

**Observability Integration:**

```python
@observe(name="synthesis_agent_run", as_type="generation")
```

- **Type**: `"generation"` (single LLM call for cross-paper analysis)
- **Trace Data**: Paper count, synthesis complexity, token usage
- **LangFuse View**: Shows synthesis LLM call with full prompt/completion

**Critical File Paths:**
- `agents/synthesis.py:54-133` - Synthesis prompt creation
- `agents/synthesis.py:135-196` - Nested data normalization
- `agents/synthesis.py:242-310` - Main `run()` method with error handling

---

### CitationAgent

**File**: `agents/citation.py`

**Core Responsibilities:**
1. Generate APA-formatted citations for all papers
2. Validate synthesis claims against source papers
3. Calculate cost estimates using dynamic pricing configuration
4. Create final ValidatedOutput with all metadata

**State Transformations:**

```python
# Input Keys
synthesis = state.get("synthesis")      # SynthesisResult from SynthesisAgent
papers = state.get("papers", [])        # List[Paper]
token_usage = state.get("token_usage", {})
model_desc = state.get("model_desc", {})

# Output Keys (added to state)
state["validated_output"] = ValidatedOutput(
    synthesis=synthesis,
    citations=[Citation(...), ...],
    retrieved_chunks=[chunk_id_1, chunk_id_2, ...],
    token_usage=token_usage,
    cost_estimate=0.0234,  # USD
    processing_time=12.5   # seconds
)
```

**Dependencies:**

```python
def __init__(
    self,
    rag_retriever  # RAGRetriever (injected but not actively used)
):
```

**Key Design Pattern: APA Citation Formatting**

```python
# agents/citation.py:31-61
def _format_apa_citation(self, paper: Paper) -> str:
    """
    Format paper in APA style.

    Format: Authors. (Year). Title. arXiv:ID. URL
    """
    # Handle different author counts
    if len(paper.authors) == 1:
        author_str = paper.authors[0]
    elif len(paper.authors) == 2:
        author_str = f"{paper.authors[0]} & {paper.authors[1]}"
    else:
        # 3+ authors: List all with ampersand before last
        author_str = ", ".join(paper.authors[:-1]) + f", & {paper.authors[-1]}"

    # Extract year from published date (format: "2024-01-15T10:30:00Z")
    year = paper.published.split("-")[0] if paper.published else "n.d."

    # Format citation
    citation = (
        f"{author_str}. ({year}). {paper.title}. "
        f"arXiv:{paper.arxiv_id}. {paper.arxiv_url}"
    )

    return citation
```

**Why This Pattern?**
- **Academic Standard**: APA is widely recognized format
- **Consistent Formatting**: Handles 1, 2, or many authors uniformly
- **Traceability**: Includes arXiv ID and URL for easy lookup

**Key Design Pattern: Synthesis Validation**

```python
# agents/citation.py:90-134
def validate_synthesis(
    self,
    synthesis: SynthesisResult,
    papers: List[Paper]
) -> Dict[str, Any]:
    """
    Validate synthesis claims against source papers.

    Returns:
    - total_consensus_points: int
    - total_contradictions: int
    - referenced_papers: List[str] (arxiv IDs mentioned)
    - chunk_ids: List[str] (chunks used for grounding)
    """
    validation_data = {
        "total_consensus_points": len(synthesis.consensus_points),
        "total_contradictions": len(synthesis.contradictions),
        "referenced_papers": [],
        "chunk_ids": []
    }

    # Collect all referenced paper IDs
    for cp in synthesis.consensus_points:
        validation_data["referenced_papers"].extend(cp.supporting_papers)
        validation_data["chunk_ids"].extend(cp.citations)

    for contr in synthesis.contradictions:
        validation_data["referenced_papers"].extend(contr.papers_a)
        validation_data["referenced_papers"].extend(contr.papers_b)
        validation_data["chunk_ids"].extend(contr.citations)

    # Deduplicate
    validation_data["referenced_papers"] = list(set(validation_data["referenced_papers"]))
    validation_data["chunk_ids"] = list(set(validation_data["chunk_ids"]))

    logger.info(
        f"Validation: {validation_data['total_consensus_points']} consensus points, "
        f"{validation_data['total_contradictions']} contradictions, "
        f"{len(validation_data['referenced_papers'])} papers referenced"
    )

    return validation_data
```

**Why This Pattern?**
- **Traceability**: Tracks which papers are actually cited
- **Metadata Extraction**: Chunk IDs for provenance tracking
- **Quality Signal**: High citation count indicates well-grounded synthesis

**Key Design Pattern: Dynamic Cost Calculation**

```python
# agents/citation.py:164-183
def calculate_cost(
    self,
    token_usage: Dict[str, int],
    model_desc: Dict[str, str]
) -> float:
    """
    Calculate cost estimate using dynamic pricing from config.
    """
    from utils.config import get_pricing_config

    pricing_config = get_pricing_config()

    # Get model-specific pricing
    llm_model = model_desc.get("llm_model", "gpt-4o-mini")
    embedding_model = model_desc.get("embedding_model", "text-embedding-3-small")

    llm_pricing = pricing_config.get_model_pricing(llm_model)
    embedding_pricing = pricing_config.get_embedding_pricing(embedding_model)

    # Calculate costs (pricing is per 1M tokens)
    input_cost = (token_usage.get("input_tokens", 0) / 1_000_000) * llm_pricing["input"]
    output_cost = (token_usage.get("output_tokens", 0) / 1_000_000) * llm_pricing["output"]
    embedding_cost = (token_usage.get("embedding_tokens", 0) / 1_000_000) * embedding_pricing

    total_cost = input_cost + output_cost + embedding_cost
    return round(total_cost, 4)
```

**Why This Pattern?**
- **Centralized Pricing**: Single source of truth in `utils/config.py`
- **Model Flexibility**: Supports any Azure OpenAI model (falls back to defaults)
- **Transparency**: Breaks down cost by operation type

**Error Handling Strategy:**

```python
# agents/citation.py:200-254
@observe(name="citation_agent_run", as_type="span")
def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Note: Citation agent rarely fails (pure data transformation).
    No complex error handling needed.
    """
    try:
        synthesis = state.get("synthesis")
        papers = state.get("papers", [])
        token_usage = state.get("token_usage", {})
        model_desc = state.get("model_desc", {})
        start_time = state.get("start_time", time.time())

        # Generate citations
        citations = []
        for paper in papers:
            citation_text = self._format_apa_citation(paper)
            citations.append(Citation(
                arxiv_id=paper.arxiv_id,
                citation_text=citation_text
            ))

        # Validate synthesis
        validation_data = self.validate_synthesis(synthesis, papers)

        # Calculate cost and timing
        cost_estimate = self.calculate_cost(token_usage, model_desc)
        processing_time = time.time() - start_time

        # Create final output
        validated_output = ValidatedOutput(
            synthesis=synthesis,
            citations=citations,
            retrieved_chunks=validation_data["chunk_ids"],
            token_usage=token_usage,
            cost_estimate=cost_estimate,
            processing_time=round(processing_time, 2)
        )

        state["validated_output"] = validated_output
        logger.info(
            f"Citation agent completed. Cost: ${cost_estimate:.4f}, "
            f"Time: {processing_time:.2f}s"
        )

        return state

    except Exception as e:
        logger.error(f"Citation agent failed: {str(e)}", exc_info=True)
        state["errors"].append(f"Citation failed: {str(e)}")
        return state
```

**Observability Integration:**

```python
@observe(name="citation_agent_run", as_type="span")
```

- **Type**: `"span"` (data processing only, no LLM calls)
- **Trace Data**: Citation count, cost estimate, processing time
- **LangFuse View**: Shows data transformation duration

**Critical File Paths:**
- `agents/citation.py:31-61` - APA citation formatting
- `agents/citation.py:90-134` - Synthesis validation
- `agents/citation.py:164-183` - Dynamic cost calculation
- `agents/citation.py:200-254` - Main `run()` method

---

## 4. Cross-Cutting Patterns

### State Management

#### AgentState TypedDict

All workflow state is managed through a strongly-typed dictionary defined in `utils/langgraph_state.py`:

```python
from typing import TypedDict, List, Dict, Optional, Any
from utils.schemas import Paper, PaperChunk, Analysis, SynthesisResult, ValidatedOutput

class AgentState(TypedDict, total=False):
    # Input fields (from user)
    query: str
    category: Optional[str]
    num_papers: int

    # Agent outputs
    papers: List[Paper]
    chunks: List[PaperChunk]
    analyses: List[Analysis]
    filtered_analyses: List[Analysis]  # After filter node
    synthesis: SynthesisResult
    validated_output: ValidatedOutput

    # Metadata
    errors: List[str]
    token_usage: Dict[str, int]  # {input_tokens, output_tokens, embedding_tokens}
    start_time: float
    processing_time: float
    model_desc: Dict[str, str]  # {llm_model, embedding_model}

    # Tracing
    trace_id: Optional[str]
    session_id: Optional[str]
    user_id: Optional[str]
```

**Key Benefits:**
- **Type Safety**: IDEs provide autocomplete and type checking
- **Documentation**: State shape is self-documenting
- **Validation**: LangGraph validates state structure at runtime

#### Serialization Requirements (msgpack)

**CRITICAL**: LangGraph uses msgpack for state checkpointing. Only these types are allowed in state:

**✅ Allowed:**
```python
# Primitives
state["query"] = "transformer architectures"  # str
state["num_papers"] = 5                       # int
state["processing_time"] = 12.5               # float
state["enabled"] = True                       # bool
state["optional_field"] = None                # None

# Collections
state["errors"] = ["Error 1", "Error 2"]      # list
state["token_usage"] = {"input": 1000}        # dict

# Pydantic models (via .model_dump())
state["papers"] = [paper.model_dump() for paper in papers]  # WRONG
state["papers"] = papers  # CORRECT (LangGraph serializes automatically)
```

**❌ Prohibited:**
```python
# Complex objects
state["progress"] = gr.Progress()        # ❌ Gradio components
state["file"] = open("data.txt")         # ❌ File handles
state["thread"] = threading.Thread()     # ❌ Thread objects
state["callback"] = lambda x: x          # ❌ Functions/callbacks
```

**Real Bug Example** (from `BUGFIX_MSGPACK_SERIALIZATION.md`):

```python
# BEFORE (broken)
def run_workflow(workflow_app, initial_state, config, progress):
    initial_state["progress"] = progress  # ❌ Non-serializable
    final_state = workflow_app.invoke(initial_state, config)
    # CRASH: TypeError: can't serialize gr.Progress

# AFTER (fixed)
def run_workflow(workflow_app, initial_state, config, progress):
    # Keep progress as local variable, NOT in state
    for event in workflow_app.stream(initial_state, config):
        # Update progress using local variable
        if progress:
            progress(0.5, desc="Processing...")
    return final_state
```

#### Token Usage Tracking Pattern

All agents update the shared `token_usage` dictionary:

```python
# Initialize in create_initial_state() (utils/langgraph_state.py:46-91)
initial_state["token_usage"] = {
    "input_tokens": 0,
    "output_tokens": 0,
    "embedding_tokens": 0
}

# RetrieverAgent updates embedding tokens
state["token_usage"]["embedding_tokens"] = len(chunks) * 300  # Estimate

# AnalyzerAgent updates LLM tokens (thread-safe)
with self.token_lock:
    self.total_input_tokens += response.usage.prompt_tokens
    self.total_output_tokens += response.usage.completion_tokens

# After all analyses
state["token_usage"]["input_tokens"] = self.total_input_tokens
state["token_usage"]["output_tokens"] = self.total_output_tokens

# SynthesisAgent accumulates (+=, not =)
state["token_usage"]["input_tokens"] += response.usage.prompt_tokens
state["token_usage"]["output_tokens"] += response.usage.completion_tokens

# CitationAgent reads final totals
cost_estimate = self.calculate_cost(state["token_usage"], model_desc)
```

**Why This Pattern?**
- **Centralized Tracking**: Single source of truth for token usage
- **Cost Transparency**: Users see exact token consumption
- **Performance Monitoring**: Track token usage trends over time

---

### Error Handling Philosophy

#### The Golden Rule: Never Raise from run()

**All agents follow this contract:**

```python
def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # Agent logic here
        return state
    except Exception as e:
        logger.error(f"Agent failed: {str(e)}", exc_info=True)
        state["errors"].append(f"Agent failed: {str(e)}")
        return state  # NEVER raise
```

**Why?**
- **Workflow Resilience**: One agent's failure doesn't crash entire pipeline
- **Partial Results**: Downstream agents can work with available data
- **Debugging**: Errors collected in state for tracing

#### Error Handling Strategies by Agent

**RetrieverAgent**: Fallback + Partial Success
```python
# Two-tier fallback for search
papers = self._search_with_fallback(query, max_results, category)
if not papers:
    state["errors"].append("No papers found")
    return state  # Early return, not exception

# Continue with partial results on download failures
for paper in papers:
    try:
        pdf_path = self._download_with_fallback(paper)
    except Exception as e:
        logger.warning(f"Skipping {paper.arxiv_id}: {str(e)}")
        continue  # Process other papers
```

**AnalyzerAgent**: Circuit Breaker + Minimal Analysis
```python
# Circuit breaker: Stop after 2 consecutive failures
if self.consecutive_failures >= self.max_consecutive_failures:
    logger.error("Circuit breaker triggered")
    break

# On failure, return minimal analysis with confidence=0.0
except Exception as e:
    return Analysis(
        arxiv_id=paper.arxiv_id,
        title=paper.title,
        methodology=[], key_findings=[], conclusions=[],
        limitations=[], contributions=[],
        confidence_score=0.0  # Signal failure
    )
```

**SynthesisAgent**: Paper Count Mismatch Handling
```python
# Defensive: Handle mismatched paper/analysis counts
if len(papers) != len(analyses):
    logger.warning(f"Count mismatch: {len(papers)} papers, {len(analyses)} analyses")
    min_len = min(len(papers), len(analyses))
    papers = papers[:min_len]
    analyses = analyses[:min_len]

# On failure, return empty synthesis with confidence=0.0
except Exception as e:
    state["synthesis"] = SynthesisResult(
        consensus_points=[], contradictions=[], research_gaps=[],
        summary=f"Synthesis failed: {str(e)}",
        papers_analyzed=[p.arxiv_id for p in papers],
        confidence_score=0.0
    )
```

**CitationAgent**: Rare Failures (Data Transformation Only)
```python
# Simpler error handling (no LLM, no external APIs)
try:
    # Pure data transformation
    citations = [self._format_apa_citation(p) for p in papers]
    cost = self.calculate_cost(token_usage, model_desc)
    return state
except Exception as e:
    logger.error(f"Citation failed: {str(e)}")
    state["errors"].append(f"Citation failed: {str(e)}")
    return state
```

#### Confidence Score as Quality Signal

All agents that can fail use `confidence_score` to indicate quality:

```python
# High confidence: Successful analysis with good context
Analysis(confidence_score=0.85, ...)

# Low confidence: Successful but limited context
Analysis(confidence_score=0.45, ...)

# Zero confidence: Failure (filter node removes these)
Analysis(confidence_score=0.0, ...)
```

**Filter Node** uses confidence scores to remove bad analyses:

```python
# orchestration/nodes.py:74-107
@observe(name="filter_low_confidence", as_type="span")
def filter_node(state: AgentState) -> AgentState:
    analyses = state.get("analyses", [])
    threshold = 0.7  # Configurable

    filtered = [a for a in analyses if a.confidence_score >= threshold]

    logger.info(
        f"Filtered {len(filtered)}/{len(analyses)} analyses "
        f"(threshold={threshold})"
    )

    state["filtered_analyses"] = filtered
    return state
```

---

### Observability Integration

#### Three-Tier Tracing Architecture

**Tier 1: Node-Level Tracing** (orchestration layer)

```python
# orchestration/nodes.py
@observe(name="analyzer_agent", as_type="span")
def analyzer_node(state: AgentState, analyzer_agent) -> AgentState:
    logger.info("Starting analyzer agent...")
    updated_state = analyzer_agent.run(state)
    logger.info(f"Analyzer completed. Analyses: {len(updated_state.get('analyses', []))}")
    return updated_state
```

**What's Captured:**
- Node execution duration
- Input/output state snapshots
- Errors caught by node wrapper

**Tier 2: Agent-Level Tracing** (agent logic)

```python
# agents/analyzer.py
@observe(name="analyzer_agent_run", as_type="generation")
def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
    # Agent logic...
    return state
```

**What's Captured:**
- Agent execution duration
- State transformations
- Agent-specific metadata (paper count, analysis count)

**Tier 3: LLM-Level Tracing** (automatic instrumentation)

```python
# utils/langfuse_client.py:74-94
from langfuse.openai import openai

def instrument_openai():
    """
    Instrument Azure OpenAI client for automatic tracing.
    All chat.completions.create() calls are automatically traced.
    """
    langfuse_client = get_langfuse_client()
    if langfuse_client:
        openai.langfuse_auth(langfuse_client)
```

**What's Captured:**
- Full prompt (system + user messages)
- Full completion (response text)
- Token usage (prompt_tokens, completion_tokens)
- Model metadata (model, temperature, max_tokens)
- Latency (time to first token, total time)
- Cost (calculated from token usage)

#### @observe Decorator Patterns

**Generation Type** (for LLM-heavy agents):
```python
@observe(name="analyzer_agent_run", as_type="generation")
def run(self, state):
    # Marks this as an LLM generation task
    # LangFuse shows token usage, cost, latency
    pass
```

**Span Type** (for data processing):
```python
@observe(name="filter_low_confidence", as_type="span")
def filter_node(state):
    # Marks this as a processing step
    # LangFuse shows duration, input/output
    pass
```

**Nested Tracing** (automatic):
```python
retriever_node()  # Creates span "retriever_agent"
  └─ retriever_agent.run()  # Creates generation "retriever_agent_run"
      └─ embedding_generator.generate_batch()  # Creates generation "embeddings"
          └─ Azure OpenAI API call  # Automatic instrumentation
```

#### Session and Trace ID Tracking

```python
# app.py:421-434
import uuid

# Generate unique session ID per workflow execution
session_id = f"session-{uuid.uuid4().hex[:8]}"

initial_state = create_initial_state(
    query=query,
    category=category,
    num_papers=num_papers,
    model_desc=model_desc,
    start_time=start_time,
    session_id=session_id,
    user_id=None  # Optional: for multi-user tracking
)
```

**Use Cases:**
- **Session Grouping**: Group all traces from single workflow execution
- **User Tracking**: Analyze behavior across multiple sessions
- **Debugging**: Find all traces for failed session

#### Graceful Degradation When LangFuse Unavailable

```python
# utils/langfuse_client.py:97-138
def observe(name: str = None, as_type: str = "span", **kwargs):
    """
    Wrapper for @observe decorator with graceful degradation.

    If LangFuse not configured, returns identity decorator.
    """
    langfuse_client = get_langfuse_client()

    if langfuse_client is None:
        # Return no-op decorator
        def identity_decorator(func):
            return func
        return identity_decorator

    # Return actual LangFuse decorator
    from langfuse.decorators import observe as langfuse_observe
    return langfuse_observe(name=name, as_type=as_type, **kwargs)
```

**Why This Pattern?**
- **Optional Observability**: App works without LangFuse configured
- **No Import Errors**: Doesn't fail if `langfuse` package missing
- **Zero Code Changes**: Same decorator usage regardless of config

---

### Performance Optimizations

#### Parallel Processing in AnalyzerAgent

```python
# agents/analyzer.py:333-359
from concurrent.futures import ThreadPoolExecutor, as_completed

def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
    papers = state.get("papers", [])
    analyses = []

    # Parallel analysis with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
        # Submit all papers for analysis
        future_to_paper = {
            executor.submit(self.analyze_paper, paper): paper
            for paper in papers
        }

        # Collect results as they complete
        for future in as_completed(future_to_paper):
            paper = future_to_paper[future]
            try:
                analysis = future.result()
                analyses.append(analysis)
            except Exception as e:
                logger.error(f"Failed to analyze {paper.arxiv_id}: {str(e)}")
```

**Performance Impact:**
- **Serial**: 5 papers × 60s = 300s (5 minutes)
- **Parallel (4 workers)**: ~75s (80% reduction)

**Thread Safety:**
```python
# agents/analyzer.py:48-51
import threading

def __init__(self, ...):
    self.token_lock = threading.Lock()
    self.total_input_tokens = 0
    self.total_output_tokens = 0

# In analyze_paper() method
with self.token_lock:
    self.total_input_tokens += response.usage.prompt_tokens
    self.total_output_tokens += response.usage.completion_tokens
```

#### Circuit Breaker Pattern

```python
# agents/analyzer.py:54-57
def __init__(self, ...):
    self.consecutive_failures = 0
    self.max_consecutive_failures = 2

# In run() method
for future in as_completed(future_to_paper):
    # Check circuit breaker BEFORE processing next result
    if self.consecutive_failures >= self.max_consecutive_failures:
        logger.error(f"Circuit breaker triggered after {self.consecutive_failures} failures")
        break

    # Process result
    if analysis.confidence_score > 0:
        self.consecutive_failures = 0  # Reset on success
    else:
        self.consecutive_failures += 1
```

**Why Circuit Breaker?**
- **Fail Fast**: Stops after 2 failures instead of wasting 3 more LLM calls
- **Cost Savings**: Prevents runaway API usage on systemic failures
- **User Experience**: Faster failure feedback

#### Batch Operations

**Embedding Generation** (RetrieverAgent):
```python
# rag/embeddings.py
def generate_batch(self, chunks: List[PaperChunk]) -> List[List[float]]:
    """
    Generate embeddings for multiple chunks in a single API call.

    Azure OpenAI supports batch size up to 2048 inputs.
    """
    texts = [chunk.content for chunk in chunks]

    # Single API call for all chunks
    response = self.client.embeddings.create(
        model=self.deployment_name,
        input=texts  # List of strings
    )

    return [item.embedding for item in response.data]
```

**Performance Impact:**
- **Serial**: 100 chunks × 50ms = 5000ms
- **Batch**: 1 call × 200ms = 200ms (96% reduction)

#### Timeout Configuration

Different agents have different timeout needs:

```python
# AnalyzerAgent (60s) - moderate timeout
self.client.chat.completions.create(
    timeout=60  # Analyzing single paper
)

# SynthesisAgent (90s) - longer timeout
self.client.chat.completions.create(
    timeout=90  # Cross-paper synthesis more complex
)
```

**Why Different Timeouts?**
- **Synthesis is slower**: Processes all papers simultaneously, larger context
- **Prevents premature failures**: Allows complex reasoning to complete
- **Still bounded**: Avoids infinite hangs

---

## 5. Workflow Orchestration

### LangGraph Workflow Structure

The workflow is defined in `orchestration/workflow_graph.py`:

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from utils.langgraph_state import AgentState

def create_workflow_graph(
    retriever_agent,
    analyzer_agent,
    synthesis_agent,
    citation_agent,
    use_checkpointing: bool = True
) -> Any:
    """
    Create LangGraph workflow with all agents and conditional routing.
    """
    # Create state graph
    workflow = StateGraph(AgentState)

    # Add nodes (lambda binds agent instances)
    workflow.add_node("retriever", lambda state: retriever_node(state, retriever_agent))
    workflow.add_node("analyzer", lambda state: analyzer_node(state, analyzer_agent))
    workflow.add_node("filter", filter_node)
    workflow.add_node("synthesis", lambda state: synthesis_node(state, synthesis_agent))
    workflow.add_node("citation", lambda state: citation_node(state, citation_agent))
    workflow.add_node("finalize", finalize_node)

    # Set entry point
    workflow.set_entry_point("retriever")

    # Add conditional edges
    workflow.add_conditional_edges(
        "retriever",
        should_continue_after_retriever,
        {
            "continue": "analyzer",
            "end": END
        }
    )

    workflow.add_conditional_edges(
        "filter",
        should_continue_after_filter,
        {
            "continue": "synthesis",
            "end": END
        }
    )

    # Add standard edges
    workflow.add_edge("analyzer", "filter")
    workflow.add_edge("synthesis", "citation")
    workflow.add_edge("citation", "finalize")
    workflow.add_edge("finalize", END)

    # Compile with checkpointing
    if use_checkpointing:
        checkpointer = MemorySaver()
        return workflow.compile(checkpointer=checkpointer)
    else:
        return workflow.compile()
```

**Complete Workflow Flow:**

```
START
  ↓
retriever
  ↓
[Check: papers found?]
  ├─ No → END
  └─ Yes → analyzer
            ↓
          filter
            ↓
          [Check: valid analyses?]
            ├─ No → END
            └─ Yes → synthesis
                      ↓
                    citation
                      ↓
                    finalize
                      ↓
                    END
```

### Node Wrapper Pattern

**Purpose of Node Wrappers:**

1. **Orchestration Concerns**: Tracing, logging, error handling
2. **Agent Logic Isolation**: Keeps agents pure and testable
3. **Consistent Interface**: All nodes follow same pattern

**Standard Node Wrapper Template:**

```python
from langfuse.decorators import observe
from utils.langgraph_state import AgentState
import logging

logger = logging.getLogger(__name__)

@observe(name="<agent_name>", as_type="<span|generation>")
def <agent>_node(state: AgentState, agent_instance) -> AgentState:
    """
    Node wrapper for <AgentName>.

    Responsibilities:
    - LangFuse tracing (via @observe)
    - Structured logging
    - Error handling
    - State transformation delegation
    """
    logger.info("Starting <agent_name> agent...")

    try:
        # Delegate to agent's run() method
        updated_state = agent_instance.run(state)

        # Log completion with metrics
        logger.info(f"<Agent> completed. <Metric>: {len(updated_state.get('<key>', []))}")
        return updated_state

    except Exception as e:
        # Catch-all error handling
        logger.error(f"<Agent> node failed: {str(e)}", exc_info=True)
        state["errors"].append(f"<Agent> failed: {str(e)}")
        return state  # Return original state on failure
```

**Example: FilterNode** (standalone logic, no agent instance)

```python
# orchestration/nodes.py:74-107
@observe(name="filter_low_confidence", as_type="span")
def filter_node(state: AgentState) -> AgentState:
    """
    Filter out low-confidence analyses.

    Note: This is NOT an agent wrapper - it's standalone logic.
    """
    analyses = state.get("analyses", [])
    threshold = 0.7

    # Filter logic
    filtered = [a for a in analyses if a.confidence_score >= threshold]

    logger.info(
        f"Filtered {len(filtered)}/{len(analyses)} analyses "
        f"(threshold={threshold})"
    )

    state["filtered_analyses"] = filtered
    return state
```

### Conditional Routing

**Two Routing Decision Points:**

**1. After Retriever: Check if Papers Found**

```python
# orchestration/nodes.py:168-179
def should_continue_after_retriever(state: AgentState) -> str:
    """
    Route based on paper retrieval success.

    Returns:
        "continue": Papers found, proceed to analyzer
        "end": No papers found, terminate workflow
    """
    papers = state.get("papers", [])

    if len(papers) == 0:
        logger.warning("No papers retrieved. Ending workflow.")
        return "end"

    logger.info(f"Retrieved {len(papers)} papers. Continuing to analyzer.")
    return "continue"
```

**Why Early Termination?**
- **Cost Savings**: No point running LLM analysis if no papers
- **User Experience**: Immediate feedback that search failed
- **Error Clarity**: Clear error message vs generic "no results"

**2. After Filter: Check if Valid Analyses Remain**

```python
# orchestration/nodes.py:182-193
def should_continue_after_filter(state: AgentState) -> str:
    """
    Route based on filter results.

    Returns:
        "continue": Valid analyses exist, proceed to synthesis
        "end": All analyses filtered out, terminate workflow
    """
    filtered = state.get("filtered_analyses", [])

    if len(filtered) == 0:
        logger.warning("No valid analyses after filtering. Ending workflow.")
        return "end"

    logger.info(f"{len(filtered)} valid analyses. Continuing to synthesis.")
    return "continue"
```

**Why Filter Check?**
- **Quality Gate**: Prevents synthesis on all-failed analyses
- **Confidence Threshold**: Only synthesizes high-quality analyses (>0.7)
- **Cost Savings**: Avoids synthesis LLM call on garbage data

### Checkpointing and State Persistence

**MemorySaver Checkpointer:**

```python
# orchestration/workflow_graph.py:120-126
from langgraph.checkpoint.memory import MemorySaver

if use_checkpointing:
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)
```

**What Gets Checkpointed:**
- **State after each node**: Full AgentState dictionary
- **Serialized to msgpack**: Efficient binary format
- **Stored in memory**: Checkpointer holds state history

**Use Cases:**

**1. Workflow Resumption:**
```python
# Get state at specific point
thread_id = "thread-abc123"
state = workflow.get_state(thread_id, checkpoint_id="checkpoint-5")

# Resume from that state
final_state = workflow.invoke(state, config={"thread_id": thread_id})
```

**2. Debugging:**
```python
# Inspect state after analyzer node
state_after_analyzer = workflow.get_state(thread_id, checkpoint_id="after-analyzer")
print(f"Analyses: {state_after_analyzer['analyses']}")
```

**3. Time Travel (Replay):**
```python
# Re-run from specific checkpoint with different parameters
state["num_papers"] = 10  # Change parameter
workflow.invoke(state, config={"thread_id": thread_id})
```

**Configuration:**

```python
# app.py:431-436 -- run_workflow() builds the config dict internally from thread_id
final_state = run_workflow(
    app=workflow_app,
    initial_state=initial_state,
    thread_id=session_id,  # Unique per execution
    use_streaming=False,
)
```

**Why Checkpointing?**
- **Resilience**: Can resume on crashes
- **Debugging**: Inspect intermediate state
- **Experimentation**: Replay from checkpoints with different configs

---

## 6. Building New Agents

### Step-by-Step Development Guide

Follow this workflow to create a new agent that integrates seamlessly with the system:

#### Step 1: Define Agent Responsibilities

**Questions to Answer:**
- What specific task does this agent perform?
- What are its inputs (which state keys)?
- What are its outputs (which state keys added/modified)?
- Does it call external APIs or LLMs?
- Can it fail? How should it degrade gracefully?

**Example: SummarizerAgent**
- **Task**: Generate concise summaries for each paper
- **Inputs**: `papers`, `chunks`
- **Outputs**: `summaries` (List[PaperSummary])
- **External Calls**: Azure OpenAI (LLM)
- **Failure Mode**: Return empty summary with confidence=0.0

#### Step 2: Create Pydantic Schemas

Add output schemas to `utils/schemas.py`:

```python
# utils/schemas.py
from pydantic import BaseModel, Field
from typing import List

class PaperSummary(BaseModel):
    """Summary of a single paper."""
    arxiv_id: str = Field(..., description="arXiv ID of the paper")
    title: str = Field(..., description="Paper title")
    summary: str = Field(..., description="3-4 sentence summary")
    key_points: List[str] = Field(default_factory=list, description="Bullet points")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Summary quality")
```

#### Step 3: Implement Agent Class

Create `agents/summarizer.py`:

```python
from typing import Dict, Any, List
import logging
import json
from openai import AzureOpenAI
from utils.schemas import Paper, PaperChunk, PaperSummary
from langfuse.decorators import observe

logger = logging.getLogger(__name__)

class SummarizerAgent:
    """Generates concise summaries for each paper."""

    def __init__(
        self,
        azure_openai_config: Dict[str, str],
        max_summary_tokens: int = 500,
        timeout: int = 30
    ):
        """
        Initialize SummarizerAgent.

        Args:
            azure_openai_config: Azure OpenAI credentials
            max_summary_tokens: Max tokens for summary generation
            timeout: LLM call timeout in seconds
        """
        self.deployment_name = azure_openai_config["deployment_name"]
        self.max_summary_tokens = max_summary_tokens
        self.timeout = timeout

        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=azure_openai_config["api_key"],
            api_version=azure_openai_config.get("api_version", "2024-02-01"),
            azure_endpoint=azure_openai_config["endpoint"]
        )

    def _create_summary_prompt(self, paper: Paper, chunks: List[PaperChunk]) -> str:
        """Create prompt for summarization."""
        # Get abstract and introduction chunks
        relevant_chunks = [
            c for c in chunks
            if c.paper_id == paper.arxiv_id and c.section in ["abstract", "introduction"]
        ][:5]  # First 5 chunks

        context = "\n\n".join([c.content for c in relevant_chunks])

        prompt = f"""
Summarize this research paper concisely.

Title: {paper.title}
Authors: {', '.join(paper.authors[:3])}

Paper Content:
{context}

Provide:
1. A 3-4 sentence summary
2. 3-5 key points (bullet list)

Return as JSON:
{{
    "summary": "3-4 sentence summary here...",
    "key_points": ["Point 1", "Point 2", ...],
    "confidence_score": 0.85
}}
"""
        return prompt

    def _normalize_summary_response(self, data: dict, paper: Paper) -> dict:
        """Normalize LLM response to match Pydantic schema."""
        def ensure_string(value):
            return str(value) if value else ""

        def ensure_list_of_strings(value):
            if isinstance(value, list):
                return [str(item) for item in value if item]
            return [str(value)] if value else []

        return {
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "summary": ensure_string(data.get("summary", "")),
            "key_points": ensure_list_of_strings(data.get("key_points", [])),
            "confidence_score": float(data.get("confidence_score", 0.0))
        }

    def summarize_paper(self, paper: Paper, chunks: List[PaperChunk]) -> PaperSummary:
        """
        Summarize a single paper.

        Args:
            paper: Paper metadata
            chunks: All chunks (filtered to this paper in method)

        Returns:
            PaperSummary with summary, key points, confidence
        """
        try:
            # Create prompt
            prompt = self._create_summary_prompt(paper, chunks)

            # Call LLM
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a research paper summarizer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,  # Deterministic
                response_format={"type": "json_object"},
                max_tokens=self.max_summary_tokens,
                timeout=self.timeout
            )

            # Parse and normalize
            data = json.loads(response.choices[0].message.content)
            normalized = self._normalize_summary_response(data, paper)

            # Create Pydantic model
            summary = PaperSummary(**normalized)

            logger.info(f"Summarized {paper.arxiv_id} (confidence={summary.confidence_score:.2f})")
            return summary

        except Exception as e:
            logger.error(f"Failed to summarize {paper.arxiv_id}: {str(e)}", exc_info=True)

            # Return minimal summary with confidence=0.0
            return PaperSummary(
                arxiv_id=paper.arxiv_id,
                title=paper.title,
                summary="",
                key_points=[],
                confidence_score=0.0
            )

    @observe(name="summarizer_agent_run", as_type="generation")
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run summarizer agent on all papers.

        Args:
            state: Workflow state (requires 'papers' and 'chunks' keys)

        Returns:
            Updated state with 'summaries' key added
        """
        try:
            papers = state.get("papers", [])
            chunks = state.get("chunks", [])

            if not papers:
                logger.warning("No papers to summarize")
                state["summaries"] = []
                return state

            # Summarize each paper
            summaries = []
            for paper in papers:
                summary = self.summarize_paper(paper, chunks)
                summaries.append(summary)

            # Update state
            state["summaries"] = summaries

            logger.info(f"Summarized {len(summaries)} papers")
            return state

        except Exception as e:
            logger.error(f"Summarizer agent failed: {str(e)}", exc_info=True)
            state["errors"].append(f"Summarizer failed: {str(e)}")
            state["summaries"] = []
            return state  # Never raise
```

#### Step 4: Add Observability Decorators

Already added in Step 3:

```python
@observe(name="summarizer_agent_run", as_type="generation")
def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
    # Agent logic...
```

**Decorator Type Selection:**
- Use `as_type="generation"` if agent calls LLM
- Use `as_type="span"` if agent only processes data
- Decorator is automatically no-op if LangFuse not configured

#### Step 5: Create Node Wrapper

Add to `orchestration/nodes.py`:

```python
# orchestration/nodes.py
from langfuse.decorators import observe
from utils.langgraph_state import AgentState
import logging

logger = logging.getLogger(__name__)

@observe(name="summarizer_agent", as_type="span")
def summarizer_node(state: AgentState, summarizer_agent) -> AgentState:
    """
    Node wrapper for SummarizerAgent.

    Responsibilities:
    - LangFuse tracing
    - Structured logging
    - Error handling
    """
    logger.info("Starting summarizer agent...")

    try:
        updated_state = summarizer_agent.run(state)

        summaries = updated_state.get("summaries", [])
        logger.info(f"Summarizer completed. Summaries: {len(summaries)}")

        return updated_state

    except Exception as e:
        logger.error(f"Summarizer node failed: {str(e)}", exc_info=True)
        state["errors"].append(f"Summarizer node failed: {str(e)}")
        return state
```

#### Step 6: Add to Workflow Graph

Update `orchestration/workflow_graph.py`:

```python
def create_workflow_graph(
    retriever_agent,
    analyzer_agent,
    summarizer_agent,  # NEW: Add parameter
    synthesis_agent,
    citation_agent,
    use_checkpointing: bool = True
):
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("retriever", lambda state: retriever_node(state, retriever_agent))
    workflow.add_node("analyzer", lambda state: analyzer_node(state, analyzer_agent))
    workflow.add_node("summarizer", lambda state: summarizer_node(state, summarizer_agent))  # NEW
    workflow.add_node("filter", filter_node)
    workflow.add_node("synthesis", lambda state: synthesis_node(state, synthesis_agent))
    workflow.add_node("citation", lambda state: citation_node(state, citation_agent))
    workflow.add_node("finalize", finalize_node)

    # Set entry point
    workflow.set_entry_point("retriever")

    # Add edges (NEW: Insert summarizer between retriever and analyzer)
    workflow.add_edge("retriever", "summarizer")  # NEW
    workflow.add_edge("summarizer", "analyzer")   # NEW
    # workflow.add_edge("retriever", "analyzer")  # REMOVE: Old direct edge

    workflow.add_edge("analyzer", "filter")
    workflow.add_edge("filter", "synthesis")
    workflow.add_edge("synthesis", "citation")
    workflow.add_edge("citation", "finalize")
    workflow.add_edge("finalize", END)

    # Compile with checkpointing
    if use_checkpointing:
        checkpointer = MemorySaver()
        return workflow.compile(checkpointer=checkpointer)
    else:
        return workflow.compile()
```

#### Step 7: Update Conditional Routing (if needed)

If your agent can fail and should terminate the workflow:

```python
# orchestration/nodes.py
def should_continue_after_summarizer(state: AgentState) -> str:
    """
    Route based on summarizer success.

    Returns:
        "continue": Summaries generated, proceed
        "end": All summaries failed, terminate
    """
    summaries = state.get("summaries", [])

    # Filter successful summaries (confidence > 0)
    valid_summaries = [s for s in summaries if s.confidence_score > 0]

    if len(valid_summaries) == 0:
        logger.warning("No valid summaries. Ending workflow.")
        return "end"

    logger.info(f"{len(valid_summaries)} valid summaries. Continuing.")
    return "continue"

# In workflow graph
workflow.add_conditional_edges(
    "summarizer",
    should_continue_after_summarizer,
    {
        "continue": "analyzer",
        "end": END
    }
)
```

#### Step 8: Initialize Agent in app.py

```python
# app.py
from agents.summarizer import SummarizerAgent

class ResearchPaperAnalyzer:
    def __init__(self):
        # ... existing initialization ...

        # Initialize new agent
        self.summarizer_agent = SummarizerAgent(
            azure_openai_config=azure_config,
            max_summary_tokens=500,
            timeout=30
        )

        # Create workflow with new agent
        self.workflow_app = create_workflow_graph(
            retriever_agent=self.retriever_agent,
            analyzer_agent=self.analyzer_agent,
            summarizer_agent=self.summarizer_agent,  # NEW
            synthesis_agent=self.synthesis_agent,
            citation_agent=self.citation_agent
        )
```

#### Step 9: Update AgentState TypedDict

Add new state keys to `utils/langgraph_state.py`:

```python
# utils/langgraph_state.py
from typing import TypedDict, List, Optional
from utils.schemas import Paper, PaperChunk, Analysis, PaperSummary  # NEW import

class AgentState(TypedDict, total=False):
    # ... existing fields ...

    # Agent outputs
    papers: List[Paper]
    chunks: List[PaperChunk]
    summaries: List[PaperSummary]  # NEW: Summaries from SummarizerAgent
    analyses: List[Analysis]
    filtered_analyses: List[Analysis]
    synthesis: SynthesisResult
    validated_output: ValidatedOutput

    # ... rest of fields ...
```

---

### Minimal Agent Template

Use this template as a starting point for new agents:

```python
# agents/template_agent.py
from typing import Dict, Any
import logging
from langfuse.decorators import observe

logger = logging.getLogger(__name__)

class TemplateAgent:
    """
    [Description of what this agent does]
    """

    def __init__(self, dependency1, dependency2, **kwargs):
        """
        Initialize TemplateAgent.

        Args:
            dependency1: Description
            dependency2: Description
            **kwargs: Additional configuration
        """
        self.dependency1 = dependency1
        self.dependency2 = dependency2
        # Initialize any other state

    def _helper_method(self, input_data):
        """Helper method for internal processing."""
        # Implementation...
        pass

    @observe(name="template_agent_run", as_type="span")  # or "generation" if LLM
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform workflow state.

        Args:
            state: Current workflow state

        Returns:
            Updated state with new keys added
        """
        try:
            # 1. Read inputs from state
            input_data = state.get("input_key", [])

            if not input_data:
                logger.warning("No input data found")
                state["output_key"] = []
                return state

            # 2. Process data
            results = self._helper_method(input_data)

            # 3. Update state with outputs
            state["output_key"] = results

            # 4. Log completion
            logger.info(f"TemplateAgent completed. Results: {len(results)}")

            return state

        except Exception as e:
            # 5. Error handling: NEVER raise, always return state
            logger.error(f"TemplateAgent failed: {str(e)}", exc_info=True)
            state["errors"].append(f"TemplateAgent failed: {str(e)}")
            state["output_key"] = []  # Provide default/empty output
            return state
```

**Node Wrapper Template:**

```python
# orchestration/nodes.py
@observe(name="template_agent", as_type="span")
def template_node(state: AgentState, template_agent) -> AgentState:
    """Node wrapper for TemplateAgent."""
    logger.info("Starting template agent...")

    try:
        updated_state = template_agent.run(state)
        logger.info(f"Template agent completed.")
        return updated_state
    except Exception as e:
        logger.error(f"Template node failed: {str(e)}", exc_info=True)
        state["errors"].append(f"Template node failed: {str(e)}")
        return state
```

---

### Testing Patterns

Create `tests/test_template_agent.py`:

```python
import pytest
from unittest.mock import Mock, patch
from agents.template_agent import TemplateAgent

class TestTemplateAgent:
    """Test suite for TemplateAgent."""

    @pytest.fixture
    def mock_dependency(self):
        """Mock external dependencies."""
        mock_dep = Mock()
        mock_dep.some_method.return_value = ["result1", "result2"]
        return mock_dep

    @pytest.fixture
    def agent(self, mock_dependency):
        """Create TemplateAgent instance with mocked dependencies."""
        return TemplateAgent(
            dependency1=mock_dependency,
            dependency2=Mock()
        )

    def test_run_success(self, agent):
        """Test successful agent execution."""
        # Arrange
        state = {
            "input_key": ["data1", "data2"],
            "errors": []
        }

        # Act
        result = agent.run(state)

        # Assert
        assert "output_key" in result
        assert len(result["output_key"]) > 0
        assert len(result["errors"]) == 0

    def test_run_empty_input(self, agent):
        """Test agent handles empty input gracefully."""
        # Arrange
        state = {
            "input_key": [],
            "errors": []
        }

        # Act
        result = agent.run(state)

        # Assert
        assert result["output_key"] == []
        assert len(result["errors"]) == 0

    def test_run_missing_input_key(self, agent):
        """Test agent handles missing state keys."""
        # Arrange
        state = {"errors": []}

        # Act
        result = agent.run(state)

        # Assert
        assert result["output_key"] == []
        assert len(result["errors"]) == 0

    def test_run_dependency_failure(self, agent, mock_dependency):
        """Test agent handles dependency failures gracefully."""
        # Arrange
        mock_dependency.some_method.side_effect = Exception("API error")
        state = {
            "input_key": ["data1"],
            "errors": []
        }

        # Act
        result = agent.run(state)

        # Assert
        assert result["output_key"] == []  # Empty on failure
        assert len(result["errors"]) > 0  # Error logged
        assert "TemplateAgent failed" in result["errors"][0]

    def test_state_not_mutated(self, agent):
        """Test agent doesn't mutate input state."""
        # Arrange
        original_state = {
            "input_key": ["data1"],
            "errors": []
        }
        state_copy = original_state.copy()

        # Act
        result = agent.run(state_copy)

        # Assert
        assert "output_key" not in original_state  # Original unchanged
        assert "output_key" in result  # Result has new key
```

**Run Tests:**

```bash
# Run all tests for this agent
pytest tests/test_template_agent.py -v

# Run with coverage
pytest tests/test_template_agent.py --cov=agents.template_agent -v

# Run single test
pytest tests/test_template_agent.py::TestTemplateAgent::test_run_success -v
```

---

### Best Practices Checklist

Use this checklist when building or reviewing agent code:

**Agent Design:**
- [ ] Agent has single, clear responsibility
- [ ] Agent implements `run(state) -> state` interface
- [ ] Dependencies injected via constructor
- [ ] No instance state between invocations (stateless)

**State Management:**
- [ ] Reads inputs using `state.get(key, default)`
- [ ] Adds new keys to state (doesn't overwrite critical keys)
- [ ] Returns modified state (doesn't mutate in-place)
- [ ] All state values are msgpack-serializable (no Gradio components, file handles, etc.)

**Error Handling:**
- [ ] Never raises exceptions from `run()`
- [ ] Catches all exceptions and logs with `exc_info=True`
- [ ] Appends errors to `state["errors"]`
- [ ] Returns partial/degraded results on failure
- [ ] Uses confidence scores to signal quality

**Pydantic Schemas:**
- [ ] Output data modeled with Pydantic classes
- [ ] Schema includes validation (Field with constraints)
- [ ] Normalization method handles malformed LLM responses
- [ ] Schema added to `utils/schemas.py` and imported in `AgentState`

**LLM Configuration (if applicable):**
- [ ] Uses `temperature=0.0` for deterministic outputs
- [ ] Uses `response_format={"type": "json_object"}` for structured data
- [ ] Sets appropriate `timeout` (60s for analysis, 90s for synthesis)
- [ ] Sets appropriate `max_tokens` limit
- [ ] Tracks token usage in `state["token_usage"]`

**Observability:**
- [ ] `run()` method decorated with `@observe`
- [ ] Uses `as_type="generation"` for LLM calls, `as_type="span"` for data processing
- [ ] Structured logging with INFO/WARNING/ERROR levels
- [ ] Logs start/completion with metrics (count, duration, etc.)

**Performance:**
- [ ] Uses parallel processing if applicable (ThreadPoolExecutor)
- [ ] Implements circuit breaker if making repeated external calls
- [ ] Uses batch operations where possible (embeddings, database)
- [ ] Appropriate timeout configuration

**Testing:**
- [ ] Test suite in `tests/test_<agent_name>.py`
- [ ] Tests cover: success, empty input, missing keys, dependency failures
- [ ] Uses mocks for external dependencies
- [ ] Tests verify state transformations
- [ ] Tests verify error handling (no exceptions raised)

**Integration:**
- [ ] Node wrapper created in `orchestration/nodes.py`
- [ ] Agent added to workflow graph in `orchestration/workflow_graph.py`
- [ ] Conditional routing added if needed
- [ ] Agent initialized in `app.py`
- [ ] AgentState TypedDict updated with new state keys

**Documentation:**
- [ ] Docstrings for class and all public methods
- [ ] Type hints for all parameters and returns
- [ ] Comments for complex logic
- [ ] Example added to AGENTS.md (this document)

---

## 7. Agent Comparison Reference

Quick reference table comparing all agents:

| Aspect | RetrieverAgent | AnalyzerAgent | SynthesisAgent | CitationAgent |
|--------|---------------|---------------|----------------|---------------|
| **File** | `agents/retriever.py` | `agents/analyzer.py` | `agents/synthesis.py` | `agents/citation.py` |
| **Primary Task** | Search arXiv, download PDFs, chunk, embed | Analyze individual papers with RAG | Cross-paper synthesis | Generate citations, validate, cost calculation |
| **Input State Keys** | `query`, `category`, `num_papers` | `papers` | `papers`, `analyses`, `query` | `synthesis`, `papers`, `token_usage`, `model_desc` |
| **Output State Keys** | `papers`, `chunks`, `token_usage[embedding_tokens]` | `analyses`, `token_usage[input/output_tokens]` | `synthesis`, `token_usage[input/output_tokens]` | `validated_output` |
| **External APIs** | arXiv API (or MCP), Azure OpenAI (embeddings) | Azure OpenAI (LLM) | Azure OpenAI (LLM) | None |
| **LLM Calls** | No (only embeddings) | Yes (one per paper) | Yes (one for all papers) | No |
| **Model** | text-embedding-3-small | gpt-4o-mini (configurable) | gpt-4o-mini (configurable) | N/A |
| **Temperature** | N/A | 0.0 | 0.0 | N/A |
| **Timeout** | 30s (download), 60s (embedding) | 60s | 90s | N/A |
| **Parallel Processing** | No | Yes (ThreadPoolExecutor, 4 workers) | No | No |
| **Observability Type** | `generation` (includes embeddings) | `generation` (LLM-heavy) | `generation` (LLM) | `span` (data only) |
| **Error Handling** | Two-tier fallback, partial success | Circuit breaker, minimal analysis (confidence=0.0) | Paper count mismatch, empty synthesis (confidence=0.0) | Rare failures (pure data transformation) |
| **Confidence Scoring** | N/A | Based on RAG context quality | Based on synthesis completeness | N/A |
| **Main Dependencies** | ArxivClient, PDFProcessor, EmbeddingGenerator, VectorStore | RAGRetriever, AzureOpenAI | RAGRetriever, AzureOpenAI | PricingConfig |
| **Failure Mode** | Returns empty papers/chunks, appends errors | Returns confidence=0.0 analyses | Returns empty synthesis, confidence=0.0 | Returns original state, appends errors |
| **Cost Impact** | Embedding tokens (~$0.01 per 100k tokens) | Input/output tokens (~$0.15-$0.60 per 1M tokens) | Input/output tokens (~$0.15-$0.60 per 1M tokens) | None (calculates cost, doesn't incur) |
| **Typical Duration** | 5-15s (download + embed) | 30-60s (parallel, 4 papers) | 10-20s (single synthesis) | <1s |
| **State Mutation** | Adds `papers`, `chunks` | Adds `analyses` | Adds `synthesis` | Adds `validated_output` |
| **Thread Safety** | N/A | Yes (token_lock for shared counter) | N/A | N/A |
| **Deterministic** | Yes (fixed search results, deterministic embeddings) | Yes (temperature=0) | Yes (temperature=0) | Yes |

---

## 8. Troubleshooting and Debugging

### Common Issues and Solutions

#### Issue 1: msgpack Serialization Error

**Symptom:**
```
TypeError: can't serialize <class 'gradio.Progress'>
```

**Cause:** Non-serializable object added to state (Gradio Progress, file handles, callbacks)

**Solution:**
1. **Never** add complex objects to state
2. Keep them as local variables instead
3. See `BUGFIX_MSGPACK_SERIALIZATION.md` for detailed fix

**Example Fix:**
```python
# WRONG
def run_workflow(workflow_app, initial_state, config, progress):
    initial_state["progress"] = progress  # ❌
    return workflow_app.invoke(initial_state, config)

# CORRECT
def run_workflow(workflow_app, initial_state, config, progress):
    # Keep progress as local variable
    for event in workflow_app.stream(initial_state, config):
        if progress:
            progress(0.5, desc="Processing...")  # ✅
    return final_state
```

---

#### Issue 2: All Analyses Filtered Out

**Symptom:**
```
WARNING: No valid analyses after filtering. Ending workflow.
```

**Cause:** All analyses have confidence_score < 0.7 (filter threshold)

**Root Causes:**
- RAG retrieval failed (no chunks found)
- LLM returned malformed JSON repeatedly
- Circuit breaker triggered after 2 failures

**Debugging Steps:**

1. **Check LangFuse traces:** See which papers failed
   ```python
   from observability import TraceReader

   reader = TraceReader()
   traces = reader.get_traces(session_id="session-abc123")
   # Pass trace_id when known -- an unscoped call scans by name across the
   # whole project's history and can be slow on a high-volume agent name.
   analyzer_spans = reader.filter_by_agent("analyzer_agent", trace_id=traces[0].id)

   for span in analyzer_spans:
       print(f"Paper: {span.metadata.get('arxiv_id')}")
       print(f"Confidence: {span.metadata.get('confidence_score')}")
   ```

2. **Check RAG retrieval:** Verify chunks were found
   ```python
   # In analyzer_agent.py, add logging
   logger.info(f"Retrieved {len(unique_chunks)} chunks for {paper.arxiv_id}")
   ```

3. **Lower filter threshold temporarily:**
   ```python
   # orchestration/nodes.py:77
   threshold = 0.5  # Lower from 0.7 to accept more analyses
   ```

4. **Check circuit breaker:**
   ```python
   # agents/analyzer.py
   logger.error(f"Circuit breaker triggered after {self.consecutive_failures} failures")
   # If you see this, investigate first 2 failures
   ```

---

#### Issue 3: Retriever Returns No Papers

**Symptom:**
```
WARNING: No papers retrieved. Ending workflow.
```

**Cause:** arXiv search returned no results (or primary/fallback clients both failed)

**Debugging Steps:**

1. **Check query and category:**
   ```python
   logger.info(f"Searching arXiv: query='{query}', category='{category}'")
   # Verify query is reasonable and category is valid (e.g., 'cs.AI', not 'AI')
   ```

2. **Test arXiv search manually:**
   ```bash
   # In terminal
   python -c "import arxiv; print(list(arxiv.Search('transformer').results())[:3])"
   ```

3. **Check fallback client:**
   ```python
   # agents/retriever.py:69-97
   logger.warning(f"Primary client failed: {str(e)}, trying fallback...")
   # If you see this, primary client (MCP) is failing
   ```

4. **Disable MCP temporarily:**
   ```bash
   # .env
   USE_MCP_ARXIV=false  # Force direct arXiv API
   ```

---

#### Issue 4: Synthesis Returns Empty Results

**Symptom:**
```json
{
    "consensus_points": [],
    "contradictions": [],
    "research_gaps": [],
    "summary": ""
}
```

**Cause:** LLM returned empty synthesis (or normalization stripped all data)

**Debugging Steps:**

1. **Check LangFuse trace for synthesis LLM call:**
   - View full prompt sent to LLM
   - View full completion received
   - Check if completion was actually empty or normalization failed

2. **Verify paper summaries in prompt:**
   ```python
   # agents/synthesis.py:54-133
   logger.debug(f"Synthesis prompt:\n{prompt}")
   # Check if paper summaries are actually populated
   ```

3. **Check normalization:**
   ```python
   # agents/synthesis.py:135-196
   logger.debug(f"Raw LLM response: {data}")
   logger.debug(f"Normalized response: {normalized}")
   # Verify normalization isn't stripping valid data
   ```

4. **Increase max_tokens:**
   ```python
   # agents/synthesis.py:280
   max_tokens=3000  # Increase from default if synthesis is cut off
   ```

---

#### Issue 5: Cost Estimate is $0.00

**Symptom:**
```
Cost: $0.0000
```

**Cause:** Token usage not tracked properly

**Debugging Steps:**

1. **Check token_usage in state:**
   ```python
   logger.info(f"Token usage: {state['token_usage']}")
   # Should show non-zero input_tokens, output_tokens, embedding_tokens
   ```

2. **Verify agents are updating token_usage:**
   ```python
   # AnalyzerAgent should do:
   state["token_usage"]["input_tokens"] = self.total_input_tokens

   # SynthesisAgent should do:
   state["token_usage"]["input_tokens"] += response.usage.prompt_tokens
   ```

3. **Check pricing configuration:**
   ```python
   from utils.config import get_pricing_config

   pricing = get_pricing_config()
   print(pricing.get_model_pricing("gpt-4o-mini"))
   # Should return {"input": 0.15, "output": 0.60} per 1M tokens
   ```

---

### Reading LangFuse Traces

**Accessing LangFuse:**

1. **Web UI:** https://cloud.langfuse.com (or self-hosted URL)
2. **Python API:**
   ```python
   from observability import TraceReader

   reader = TraceReader()
   traces = reader.get_traces(limit=10)
   ```

**Trace Structure:**

```
Trace (session-abc123)
│
├─ Span: retriever_agent
│  ├─ Generation: retriever_agent_run
│  └─ Generation: embeddings (Azure OpenAI)
│
├─ Span: analyzer_agent
│  ├─ Generation: analyzer_agent_run
│  ├─ Generation: LLM Call 1 (paper 1)
│  ├─ Generation: LLM Call 2 (paper 2)
│  └─ Span: rag_retrieve
│
├─ Span: filter_low_confidence
│
├─ Span: synthesis_agent
│  ├─ Generation: synthesis_agent_run
│  └─ Generation: LLM Call (synthesis)
│
└─ Span: citation_agent
   └─ Span: citation_agent_run
```

**What to Look For:**

**1. Execution Duration:**
- Span duration = total time including child spans
- Generation duration = time for single LLM call
- Look for slow spans (>60s) indicating bottlenecks

**2. Token Usage:**
- Generations show `usage.prompt_tokens` and `usage.completion_tokens`
- High token usage = higher cost
- Unusually low tokens may indicate truncation

**3. Errors:**
- Spans with `level: ERROR` indicate failures
- Check `metadata.error` for exception details
- Trace errors back to specific papers/operations

**4. LLM Prompts/Completions:**
- Click on Generation to see full prompt and completion
- Verify prompt includes expected context
- Check if completion is valid JSON

**Example Query:**

```python
from observability import TraceReader, AgentPerformanceAnalyzer

reader = TraceReader()
analyzer = AgentPerformanceAnalyzer()

# Get failed traces
traces = reader.get_traces(limit=100)
failed_traces = [t for t in traces if t.status == "ERROR"]

print(f"Failed traces: {len(failed_traces)}/{len(traces)}")

# Analyze analyzer latency
stats = analyzer.agent_latency_stats("analyzer_agent", days=7)
print(f"Analyzer P95 latency: {stats.p95_latency_ms:.2f}ms")

# Check error rates
error_rates = analyzer.error_rates(days=7)
for agent, rate in error_rates.items():
    print(f"{agent}: {rate:.1%} error rate")
```

---

### State Inspection Techniques

**During Development (in agent code):**

```python
# agents/analyzer.py
def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
    # Print state keys at entry
    logger.debug(f"State keys: {state.keys()}")

    # Print specific values
    papers = state.get("papers", [])
    logger.debug(f"Received {len(papers)} papers: {[p.arxiv_id for p in papers]}")

    # ... agent logic ...

    # Print state changes before return
    logger.debug(f"Returning {len(state.get('analyses', []))} analyses")

    return state
```

**In Gradio UI (during workflow execution):**

```python
# app.py
final_state = run_workflow(app=workflow_app, initial_state=initial_state, thread_id=session_id)

# Inspect final state
print(f"Final state keys: {final_state.keys()}")
print(f"Papers: {len(final_state.get('papers', []))}")
print(f"Analyses: {len(final_state.get('analyses', []))}")
print(f"Errors: {final_state.get('errors', [])}")
print(f"Token usage: {final_state.get('token_usage', {})}")
```

**Using Checkpointer (post-execution):**

```python
# Get state at specific checkpoint
from orchestration.workflow_graph import get_workflow_state

thread_id = "session-abc123"
state_after_analyzer = get_workflow_state(workflow_app, thread_id, checkpoint_id="after-analyzer")

print(f"Analyses after analyzer: {len(state_after_analyzer.get('analyses', []))}")

# Compare with state after filter
state_after_filter = get_workflow_state(workflow_app, thread_id, checkpoint_id="after-filter")
print(f"Analyses after filter: {len(state_after_filter.get('filtered_analyses', []))}")
print(f"Filtered out: {len(state_after_analyzer['analyses']) - len(state_after_filter['filtered_analyses'])}")
```

---

### Log Analysis Patterns

**Log Levels:**
- **INFO**: Normal workflow progress (agent start/completion, counts)
- **WARNING**: Recoverable issues (fallback triggered, empty results, low confidence)
- **ERROR**: Failures (exceptions caught, agent failures, API errors)
- **DEBUG**: Detailed debugging (state contents, intermediate values)

**Useful Log Patterns:**

**1. Track Workflow Progress:**
```bash
# In terminal, tail logs and grep for agent completions
tail -f app.log | grep "completed"

# Output:
# INFO: Retriever completed. Papers: 5, Chunks: 237
# INFO: Analyzer completed. Analyses: 5
# INFO: Filter completed. Valid: 4/5
# INFO: Synthesis completed. Consensus: 3, Contradictions: 1
# INFO: Citation completed. Cost: $0.0234
```

**2. Identify Failures:**
```bash
# Grep for ERROR logs
grep "ERROR" app.log | tail -20

# Analyze common failure patterns
grep "ERROR" app.log | cut -d':' -f4- | sort | uniq -c | sort -rn
```

**3. Track Fallback Usage:**
```bash
# Check how often fallback client is used
grep "trying fallback" app.log | wc -l
grep "Searching with fallback client" app.log | wc -l
```

**4. Monitor Circuit Breaker:**
```bash
# Check if circuit breaker is triggering
grep "Circuit breaker triggered" app.log

# If found, investigate what caused consecutive failures
grep "consecutive_failures" app.log
```

**5. Analyze Token Usage:**
```bash
# Extract token usage from logs
grep "Token usage" app.log | tail -10

# Calculate total cost
grep "Cost:" app.log | awk '{sum+=$NF} END {print "Total: $"sum}'
```

---

## Appendix: File Reference

**Agent Implementations:**
- `agents/retriever.py` - RetrieverAgent with fallback mechanisms
- `agents/analyzer.py` - AnalyzerAgent with parallel processing and circuit breaker
- `agents/synthesis.py` - SynthesisAgent with cross-paper analysis
- `agents/citation.py` - CitationAgent with APA formatting and cost calculation

**Orchestration:**
- `orchestration/__init__.py` - Module exports
- `orchestration/nodes.py` - Node wrappers with tracing and error handling
- `orchestration/workflow_graph.py` - LangGraph workflow builder and execution

**State Management:**
- `utils/langgraph_state.py` - AgentState TypedDict and initialization helpers
- `utils/schemas.py` - Pydantic models for all data structures

**Observability:**
- `utils/langfuse_client.py` - LangFuse client initialization and @observe decorator
- `observability/trace_reader.py` - Trace querying and export API
- `observability/analytics.py` - Performance analytics and trajectory analysis

**Configuration:**
- `utils/config.py` - Pricing configuration and environment variables
- `.env.example` - Environment variable template

**Documentation:**
- `CLAUDE.md` - Comprehensive system-wide developer guide
- `AGENTS.md` - This document (agent architecture deep-dive)
- `REFACTORING_SUMMARY.md` - LangGraph + LangFuse refactoring details
- `BUGFIX_MSGPACK_SERIALIZATION.md` - msgpack serialization fix
- `observability/README.md` - Observability documentation

---

## Document Maintenance

**Last Updated:** 2025-12-20

**Version:** 1.0

**Authors:** Claude Sonnet 4.5 (auto-generated from codebase exploration)

**Changelog:**
- 2025-12-20: Initial creation with comprehensive agent documentation

**Contributing:**
- When adding new agents, update Section 3 (Individual Agent Deep Dives)
- When adding new patterns, update Section 4 (Cross-Cutting Patterns)
- When modifying workflow, update Section 5 (Workflow Orchestration)
- Keep Agent Comparison Reference (Section 7) in sync with agent changes

---

**End of AGENTS.md**
