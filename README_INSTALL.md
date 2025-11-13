# Installation Instructions

## Issue: MCP Dependency Conflict

Some dependencies (particularly `spaces` from Gradio) try to downgrade `mcp` to version 1.10.1, which conflicts with `fastmcp` that requires `mcp>=1.17.0`.

## Solution

Use the constraints file when installing dependencies:

```bash
pip install -r pre-requirements.txt
pip install -c constraints.txt -r requirements.txt
```

The `-c constraints.txt` flag enforces the mcp version and prevents downgrades.

## For Hugging Face Spaces

If deploying to Hugging Face Spaces, ensure the installation command uses constraints:
```bash
pip install -c constraints.txt -r requirements.txt
```
