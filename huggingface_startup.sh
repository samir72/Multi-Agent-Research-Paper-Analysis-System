#!/bin/bash
# Hugging Face Spaces startup script
# This runs after pip install to fix the mcp dependency conflict

echo "Fixing MCP dependency conflict..."
pip install --force-reinstall --no-deps mcp==1.17.0
echo "MCP version fixed!"
pip show mcp | grep Version

# Start the application
python app.py
