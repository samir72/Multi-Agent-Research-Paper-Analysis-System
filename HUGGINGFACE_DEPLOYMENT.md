# HuggingFace Spaces Deployment Guide

This guide explains how to deploy the Multi-Agent Research Paper Analysis System to HuggingFace Spaces.

## Prerequisites

1. **HuggingFace Account**: Create an account at [huggingface.co](https://huggingface.co)
2. **Azure OpenAI Resource**: You need an active Azure OpenAI resource with:
   - A deployed LLM model (e.g., `gpt-4o-mini`)
   - A deployed embedding model (e.g., `text-embedding-3-small`)

## Required Environment Variables

You **MUST** configure the following environment variables in HuggingFace Spaces Settings > Repository secrets:

### Azure OpenAI Configuration (REQUIRED)

| Variable Name | Description | Example |
|--------------|-------------|---------|
| `AZURE_OPENAI_ENDPOINT` | Your Azure OpenAI resource endpoint | `https://your-resource.openai.azure.com/` |
| `AZURE_OPENAI_API_KEY` | Your Azure OpenAI API key | `abc123...` |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Your LLM deployment name | `gpt-4o-mini` |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME` | Your embedding deployment name | `text-embedding-3-small` |
| `AZURE_OPENAI_API_VERSION` | Azure OpenAI API version | `2024-05-01-preview` |

### LangFuse Observability (Optional)

| Variable Name | Description | Default |
|--------------|-------------|---------|
| `LANGFUSE_ENABLED` | Enable/disable LangFuse tracing | `true` |
| `LANGFUSE_PUBLIC_KEY` | LangFuse public key | (required if enabled) |
| `LANGFUSE_SECRET_KEY` | LangFuse secret key | (required if enabled) |
| `LANGFUSE_HOST` | LangFuse host URL | `https://cloud.langfuse.com` |

### MCP Configuration (Optional)

| Variable Name | Description | Default |
|--------------|-------------|---------|
| `USE_MCP_ARXIV` | Use MCP for arXiv access | `false` |
| `USE_LEGACY_MCP` | Use legacy MCP instead of FastMCP | `false` |
| `MCP_ARXIV_STORAGE_PATH` | MCP server storage path | `./data/mcp_papers/` |
| `FASTMCP_SERVER_PORT` | FastMCP server port | `5555` |

## Common Deployment Issues

### 1. 404 Error: "Resource not found"

**Symptoms:**
```
Error code: 404 - {'error': {'code': '404', 'message': 'Resource not found'}}
```

**Cause:** Missing or incorrect `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME` variable.

**Solution:**
1. Go to HuggingFace Spaces Settings > Repository secrets
2. Add `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME` with your embedding deployment name
3. Verify the deployment exists in your Azure OpenAI resource

### 2. Missing Environment Variables

**Symptoms:**
```
ValueError: Missing required environment variables: AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME
```

**Solution:**
The app will now validate all required variables on startup. Follow the error message to set missing variables in HuggingFace Spaces secrets.

### 3. MCP Dependency Conflicts

**Symptoms:**
```
ImportError: cannot import name 'FastMCP'
```

**Solution:**
The `huggingface_startup.sh` script automatically fixes MCP version conflicts. Ensure this script is configured as the startup command in your Space's settings.

## Deployment Steps

### 1. Create a New Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Select "Gradio" as the SDK
4. Choose Python 3.10 as the version
5. Set the Space name and visibility

### 2. Configure Repository Secrets

1. Go to your Space's Settings
2. Scroll to "Repository secrets"
3. Add all required environment variables listed above
4. Click "Save" after adding each variable

### 3. Configure Startup Command

In your Space's README.md, ensure the startup command uses the custom script:

```yaml
---
title: Multi-Agent Research Paper Analysis
emoji: 📚
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.11.0
python_version: 3.10
app_file: app.py
startup_duration_timeout: 5m
---
```

In your Space settings, set the startup command to:
```bash
bash huggingface_startup.sh
```

### 4. Push Your Code

```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
git push hf main
```

### 5. Monitor Deployment

1. Watch the build logs in HuggingFace Spaces
2. Look for the environment variable check output:
   ```
   🔍 Checking environment variables...
   ✅ Found: AZURE_OPENAI_ENDPOINT
   ✅ Found: AZURE_OPENAI_API_KEY
   ✅ Found: AZURE_OPENAI_DEPLOYMENT_NAME
   ✅ Found: AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME
   ```
3. If any variables are missing, the deployment will fail with clear instructions

## Verifying Deployment

Once deployed, test your Space:

1. Open the Space URL
2. Enter a research query (e.g., "transformer architectures in NLP")
3. Select an arXiv category
4. Click "Analyze Papers"
5. Verify that papers are retrieved and analyzed successfully

## Troubleshooting

### Check Logs

View real-time logs in HuggingFace Spaces:
1. Go to your Space
2. Click on "Logs" tab
3. Look for error messages or warnings

### Validate Azure OpenAI Deployments

Ensure your deployments exist:
1. Go to [portal.azure.com](https://portal.azure.com)
2. Navigate to your Azure OpenAI resource
3. Click "Model deployments"
4. Verify both LLM and embedding deployments are listed and active

### Test Locally First

Before deploying to HuggingFace Spaces:
1. Copy `.env.example` to `.env`
2. Fill in your Azure OpenAI credentials
3. Run `python app.py` locally
4. Verify everything works
5. Then push to HuggingFace Spaces

## Performance Considerations

- **Cold Start**: First load may take 1-2 minutes as dependencies initialize
- **Memory**: Recommended minimum 4GB RAM
- **Storage**: ~500MB for dependencies + downloaded papers
- **Timeout**: Set `startup_duration_timeout: 5m` in README.md

## Security Best Practices

1. **Never commit API keys** to the repository
2. **Use HuggingFace Spaces secrets** for all sensitive variables
3. **Rotate keys regularly** in both Azure and HuggingFace
4. **Monitor usage** in Azure OpenAI to prevent unexpected costs
5. **Set rate limits** in Azure to prevent abuse

## Cost Management

- **Embedding costs**: ~$0.02 per 1M tokens
- **LLM costs**: ~$0.15-$0.60 per 1M tokens (depending on model)
- **Typical analysis**: 5 papers costs ~$0.10-$0.50
- **Monitor usage**: Use Azure OpenAI metrics dashboard
- **LangFuse observability**: Track token usage and costs per request

## Support

For issues specific to:
- **This application**: Open an issue on GitHub
- **HuggingFace Spaces**: Check [HuggingFace Docs](https://huggingface.co/docs/hub/spaces)
- **Azure OpenAI**: Consult [Azure OpenAI Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
