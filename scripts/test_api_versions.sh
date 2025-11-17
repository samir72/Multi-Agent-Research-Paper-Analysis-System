#!/bin/bash
# Test different API versions to find which one works with your deployment

set -a
source .env 2>/dev/null || true
set +a

ENDPOINT="${AZURE_OPENAI_ENDPOINT}"
API_KEY="${AZURE_OPENAI_API_KEY}"
DEPLOYMENT_NAME="${AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME}"

# Common API versions to test
API_VERSIONS=(
    "2024-02-01"
    "2024-05-01-preview"
    "2023-12-01-preview"
    "2023-05-15"
    "2023-03-15-preview"
    "2022-12-01"
)

echo "=================================="
echo "Testing API Versions for Embedding Deployment"
echo "=================================="
echo ""
echo "Endpoint: $ENDPOINT"
echo "Deployment: $DEPLOYMENT_NAME"
echo ""

for API_VERSION in "${API_VERSIONS[@]}"; do
    echo "Testing API version: $API_VERSION"

    RESPONSE=$(curl -s -w "\n%{http_code}" -X POST \
        "${ENDPOINT}openai/deployments/${DEPLOYMENT_NAME}/embeddings?api-version=${API_VERSION}" \
        -H "Content-Type: application/json" \
        -H "api-key: ${API_KEY}" \
        -d '{"input": "test"}' 2>&1)

    HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
    BODY=$(echo "$RESPONSE" | sed '$d')

    if [ "$HTTP_CODE" = "200" ]; then
        echo "  ✅ SUCCESS! HTTP $HTTP_CODE"
        echo "  Use this in your .env: AZURE_OPENAI_API_VERSION=$API_VERSION"
        echo ""
        echo "  Response sample:"
        echo "$BODY" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'data' in data:
        dim = len(data['data'][0]['embedding'])
        print(f'    Embedding dimension: {dim}')
        print(f'    Model: {data.get(\"model\", \"unknown\")}')
except:
    pass
" 2>/dev/null
        echo ""
        echo "=================================="
        echo "✅ FOUND WORKING API VERSION: $API_VERSION"
        echo "=================================="
        exit 0
    else
        ERROR_MSG=$(echo "$BODY" | python3 -c "import sys, json; print(json.load(sys.stdin).get('error', {}).get('message', 'Unknown error'))" 2>/dev/null || echo "Unknown error")
        echo "  ❌ FAILED: HTTP $HTTP_CODE - $ERROR_MSG"
    fi
    echo ""
done

echo "=================================="
echo "❌ No working API version found"
echo "=================================="
echo ""
echo "This suggests a different issue. Please check:"
echo "  1. The deployment name is EXACTLY: $DEPLOYMENT_NAME (case-sensitive)"
echo "  2. The deployment is in the same resource as: $ENDPOINT"
echo "  3. The deployment status is 'Succeeded' in Azure Portal"
exit 1
