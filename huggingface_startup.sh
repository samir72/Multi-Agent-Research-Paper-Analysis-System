#!/bin/bash
# Hugging Face Spaces startup script
# This runs after pip install to fix the mcp dependency conflict

echo "🔧 Fixing MCP dependency conflict..."
pip install --force-reinstall --no-deps mcp==1.17.0
echo "✅ MCP version fixed!"
pip show mcp | grep Version

# Check if required environment variables are set
echo ""
echo "🔍 Checking environment variables..."

required_vars=("AZURE_OPENAI_ENDPOINT" "AZURE_OPENAI_API_KEY" "AZURE_OPENAI_DEPLOYMENT_NAME" "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
missing_vars=()

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        missing_vars+=("$var")
        echo "❌ Missing: $var"
    else
        echo "✅ Found: $var"
    fi
done

if [ ${#missing_vars[@]} -ne 0 ]; then
    echo ""
    echo "⚠️  ERROR: Missing required environment variables!"
    echo "Please set the following in HuggingFace Spaces Settings > Repository secrets:"
    for var in "${missing_vars[@]}"; do
        echo "  - $var"
    done
    echo ""
    echo "See .env.example for the complete list of required variables."
    exit 1
fi

echo ""
echo "✅ All required environment variables are set!"
echo ""

# Start the application
echo "🚀 Starting application..."
python app.py
