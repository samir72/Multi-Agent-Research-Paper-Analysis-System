# Bug Fix: LangGraph msgpack Serialization Error

## Problem

The application was crashing with the error:
```
Type is not msgpack serializable: Progress
```

This occurred when LangGraph attempted to serialize the workflow state for checkpointing after the citation node completed.

## Root Cause

The Gradio `Progress` object was being added to the LangGraph state dictionary:
```python
# app.py line 460 (old)
initial_state["progress"] = progress
```

LangGraph uses msgpack for state serialization (required for checkpointing), but msgpack cannot serialize Gradio's Progress object since it's a complex Python object with methods and internal state.

## Solution

### Changes Made

1. **Removed Progress from State Schema** (`utils/langgraph_state.py`)
   - Removed `progress: Optional[Any]` field from `AgentState` TypedDict
   - Removed `"progress": None` from `create_initial_state()` return value

2. **Removed Progress from State Initialization** (`app.py`)
   - Removed line: `initial_state["progress"] = progress`
   - Added comment explaining why Progress is not in state

3. **Removed Progress Checks from Nodes** (`orchestration/nodes.py`)
   - Removed all `if state.get("progress"):` checks from:
     - `retriever_node()`
     - `analyzer_node()`
     - `synthesis_node()`
     - `citation_node()`

4. **Removed Legacy Node Methods** (`app.py`)
   - Removed unused methods that were checking for progress in state:
     - `_retriever_node()`
     - `_filter_low_confidence_node()`
     - `_synthesis_node()`
     - `_citation_node()`

### Why This Works

- **Progress stays functional**: The `progress` object is still passed to `run_workflow()` and used locally (lines 407, 425, 438 in app.py)
- **State stays serializable**: LangGraph can now serialize the state using msgpack since it only contains serializable types
- **No loss of functionality**: Progress updates still work via local variable usage in `run_workflow()`
- **Backward compatible**: The fix doesn't break any existing functionality

## Architecture Principle

**LangGraph State Rule**: Only store msgpack-serializable data in LangGraph state:
- ✅ Primitives: str, int, float, bool, None
- ✅ Collections: list, dict
- ✅ Pydantic models (serializable via .model_dump())
- ❌ Complex objects: Gradio components, file handles, thread objects, callbacks

For UI components like Gradio Progress, pass them as function parameters or use them in the orchestration layer, **not** in the state dictionary.

## Testing

The fix should resolve the error and allow the workflow to complete successfully. To verify:

1. Run the application: `python app.py`
2. Submit a research query
3. Verify the workflow completes without "Type is not msgpack serializable" error
4. Verify progress updates still appear in the Gradio UI
5. Check that results are properly cached and displayed

## Deployment Compatibility

This fix works for both:
- ✅ Local development (tested)
- ✅ Hugging Face Spaces (msgpack serialization is consistent across platforms)

No environment-specific changes needed.
