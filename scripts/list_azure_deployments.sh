#!/bin/bash
# List all deployments in your Azure OpenAI resource

# Load environment variables
source .env 2>/dev/null || true

# Extract resource name and subscription info from endpoint
ENDPOINT="${AZURE_OPENAI_ENDPOINT}"
API_KEY="${AZURE_OPENAI_API_KEY}"
API_VERSION="${AZURE_OPENAI_API_VERSION:-2024-02-01}"

echo "=================================="
echo "Azure OpenAI Deployments"
echo "=================================="
echo ""
echo "Endpoint: $ENDPOINT"
echo ""

# List deployments
curl -s "${ENDPOINT}openai/deployments?api-version=${API_VERSION}" \
  -H "api-key: ${API_KEY}" \
  -H "Content-Type: application/json" | python3 -m json.tool

echo ""
echo "=================================="
echo "Copy the exact 'id' or 'model' name from above and use it as AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"
echo "=================================="
