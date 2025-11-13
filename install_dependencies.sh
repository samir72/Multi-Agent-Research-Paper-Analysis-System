#!/bin/bash
# Installation script to handle MCP dependency conflicts

set -e  # Exit on error

echo "Step 1: Installing pre-requirements..."
pip install -r pre-requirements.txt

echo "Step 2: Installing fastmcp and mcp first..."
pip install fastmcp==2.13.0.2

echo "Step 3: Installing remaining requirements..."
pip install -r requirements.txt --no-deps || true

echo "Step 4: Installing all requirements with dependencies (mcp will be preserved)..."
pip install -r requirements.txt

echo "Step 5: Reinstalling mcp to ensure correct version..."
pip install --force-reinstall --no-deps mcp==1.17.0

echo "Installation complete!"
echo "Verifying mcp version..."
pip show mcp | grep Version
