---
name: verify-trace
description: Live end-to-end LangFuse trace verification for this repo
---

1. Print the installed `langfuse` version (`pip show langfuse`) and confirm it satisfies `>=3.10.0,<4.0.0` — this repo was previously broken by a silent v2→v3 API surface change (`client.get_traces()` etc. don't exist on v3; use `client.api.trace.get/list`, `client.api.observations.get_many`).
2. Fire a real request through the LangGraph workflow (`run_workflow()` in `orchestration/workflow_graph.py`, or via `app.py`) with a known `session_id`, so `workflow_trace()` pre-generates a deterministic `trace_id` via `Langfuse.create_trace_id(seed=session_id)`.
3. Query LangFuse for that `trace_id` using `observability.TraceReader` (the fixed v3 query surface) — confirm the root span exists, every node from `retriever` through `finalize` nested under it, and that each analyzer `Send` branch produced its own span with the correct `user_id` (from `request.session_hash`).
4. Also grep application logs for `[trace_id=<id>]` to confirm `utils/trace_context.py`'s log correlation is working end to end.
5. Report the raw span list and matching log lines — do not claim success based on code inspection or unit tests alone.
