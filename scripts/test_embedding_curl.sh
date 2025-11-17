#!/bin/bash
# Test Azure OpenAI embedding deployment directly via curl

# Load environment variables
set -a
source .env 2>/dev/null || true
set +a

ENDPOINT="${AZURE_OPENAI_ENDPOINT}"
API_KEY="${AZURE_OPENAI_API_KEY}"
DEPLOYMENT_NAME="${AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME}"
API_VERSION="${AZURE_OPENAI_API_VERSION:-2024-02-01}"

echo "=================================="
echo "Testing Azure OpenAI Embedding Deployment"
echo "=================================="
echo ""
echo "Endpoint: $ENDPOINT"
echo "Deployment: $DEPLOYMENT_NAME"
echo "API Version: $API_VERSION"
echo ""
echo "Sending test request..."
echo ""

# Make the embedding request
curl -X POST "${ENDPOINT}openai/deployments/${DEPLOYMENT_NAME}/embeddings?api-version=${API_VERSION}" \
  -H "Content-Type: application/json" \
  -H "api-key: ${API_KEY}" \
  -d '{
    "input": "This is a test embedding request"
  }' 2>&1 | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'error' in data:
        print('❌ ERROR:')
        print(json.dumps(data, indent=2))
        sys.exit(1)
    elif 'data' in data:
        embedding_dim = len(data['data'][0]['embedding'])
        print('✅ SUCCESS!')
        print(f'   Embedding dimension: {embedding_dim}')
        print(f'   Model: {data.get(\"model\", \"unknown\")}')
        print(f'   Usage tokens: {data.get(\"usage\", {}).get(\"total_tokens\", 0)}')
        sys.exit(0)
except Exception as e:
    print(f'❌ Failed to parse response: {e}')
    sys.exit(1)
"

echo ""
echo "=================================="
