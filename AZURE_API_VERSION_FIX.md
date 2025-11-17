# Azure OpenAI API Version Fix

## Problem

**Error**: `Error code: 404 - {'error': {'code': '404', 'message': 'Resource not found'}}`

**Root Cause**: The `AZURE_OPENAI_API_VERSION` environment variable was set to `2024-02-01`, which is outdated and not supported by the Azure OpenAI service.

## Solution

Update the `AZURE_OPENAI_API_VERSION` to a supported version.

### Recommended API Version

```bash
AZURE_OPENAI_API_VERSION=2024-07-18
```

### Alternative Supported Versions

- `2024-08-01-preview` (latest preview)
- `2024-06-01`
- `2024-05-01-preview`
- `2024-02-15-preview`

## Configuration

### Local Development

Update your `.env` file:

```bash
# Change from:
AZURE_OPENAI_API_VERSION=2024-02-01

# To:
AZURE_OPENAI_API_VERSION=2024-07-18
```

### HuggingFace Spaces Deployment

1. Go to your Space settings
2. Navigate to "Repository secrets"
3. Update or add: `AZURE_OPENAI_API_VERSION=2024-07-18`
4. Factory reboot the Space to apply changes

## Validation

### Step 1: Validate Locally

Run the diagnostic script to verify your configuration:

```bash
python scripts/validate_azure_embeddings.py
```

**Expected Output**:
```
✅ AZURE_OPENAI_API_VERSION: 2024-07-18
✅ SUCCESS: Embedding generated successfully!
✅ All checks passed! Your Azure OpenAI embeddings configuration is correct.
```

### Step 2: Test the Application

```bash
python app.py
```

Navigate to http://localhost:7860 and test with a query to ensure no 404 errors occur.

### Step 3: Verify HuggingFace Deployment

1. Update the `AZURE_OPENAI_API_VERSION` secret in HuggingFace Spaces
2. Restart the Space
3. Monitor logs for successful startup
4. Test a query to confirm the fix

## Required Environment Variables

Ensure all Azure OpenAI variables are properly configured:

```bash
# Core Azure OpenAI (all required)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini
AZURE_OPENAI_API_VERSION=2024-07-18  # UPDATED

# Embeddings deployment (CRITICAL)
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-small
```

## Additional Notes

### Checking API Version Support

To verify which API versions are supported for your Azure OpenAI resource:

1. Visit the [Azure OpenAI API Version Reference](https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation)
2. Check for deprecation notices
3. Use the latest stable version for best compatibility

### Impact of API Version

The API version determines:
- Available features and endpoints
- Request/response schemas
- Model availability
- Rate limits and quotas

Using an outdated or unsupported API version will result in 404 errors even if your deployment names are correct.

## Prevention

### For Future Deployments

1. **Always validate before deploying**:
   ```bash
   python scripts/validate_azure_embeddings.py
   ```

2. **Keep API version up to date**: Check Azure documentation quarterly for deprecations

3. **Document your configuration**: Maintain a record of your Azure OpenAI setup

4. **Test after updates**: Always test locally before deploying to production

## Testing Checklist

- [ ] Updated `AZURE_OPENAI_API_VERSION` to `2024-07-18` in `.env`
- [ ] Run `python scripts/validate_azure_embeddings.py` → Success
- [ ] Test local app with `python app.py` → No 404 errors
- [ ] Updated HuggingFace Spaces secret
- [ ] Restarted HuggingFace Space
- [ ] Verified no 404 errors in production logs
- [ ] Tested query in deployed Space → Success

## Related Files

- `.env.example` - Environment variable template
- `scripts/validate_azure_embeddings.py` - Configuration validation script
- `CLAUDE.md` - Development guide
- `README.md` - Project documentation
