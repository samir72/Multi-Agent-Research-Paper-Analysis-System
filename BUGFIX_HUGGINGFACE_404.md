# Bug Fix: HuggingFace Spaces 404 Error for Embeddings

## Issue Summary

**Date**: 2025-11-17
**Environment**: HuggingFace Spaces deployment
**Severity**: Critical (blocks deployment)
**Status**: ✅ Fixed

### Error Log
```
2025-11-17 08:46:13,968 - rag.embeddings - ERROR - Error generating embedding: Error code: 404 - {'error': {'code': '404', 'message': 'Resource not found'}}
2025-11-17 08:46:22,171 - __main__ - ERROR - Workflow error: RetryError[<Future at 0x7fc76c42fcd0 state=finished raised NotFoundError>]
```

## Root Cause

The error occurred because the **`AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME`** environment variable was **not set** in HuggingFace Spaces, causing the Azure OpenAI API to return a 404 error when trying to generate embeddings.

### Why This Happened

1. **Inconsistent variable name in `.env.example`**: The example file had the variable commented out and named differently:
   ```bash
   # .env.example (OLD - BROKEN)
   # AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small  # Wrong name!
   ```

2. **No validation on startup**: The app did not validate that all required environment variables were set before attempting to use them.

3. **Unclear error messages**: The 404 error from Azure OpenAI didn't clearly indicate which deployment was missing.

## The Fix

### 1. Fixed `.env.example` (lines 7-8)

**Before:**
```bash
# Optional: Embedding model deployment name (if different)
# AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
```

**After:**
```bash
# REQUIRED: Embedding model deployment name
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-small
```

**Changes:**
- ✅ Uncommented the variable (it's required, not optional)
- ✅ Fixed variable name: `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME` (matches code)
- ✅ Added `AZURE_OPENAI_API_VERSION=2024-05-01-preview` for completeness

### 2. Added Environment Validation in `app.py` (lines 43-75)

```python
def validate_environment():
    """Validate that all required environment variables are set."""
    required_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_DEPLOYMENT_NAME",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"  # Now validated!
    ]

    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if not value or value.strip() == "":
            missing_vars.append(var)

    if missing_vars:
        error_msg = (
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            f"Please set them in your .env file or HuggingFace Spaces secrets.\n"
            f"See .env.example for reference."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Log configuration (masked)
    logger.info(f"Azure OpenAI Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
    logger.info(f"LLM Deployment: {os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')}")
    logger.info(f"Embedding Deployment: {os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME')}")
    logger.info(f"API Version: {os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-01')}")

# Validate environment before importing other modules
validate_environment()
```

**Benefits:**
- ✅ Fails fast with clear error message at startup
- ✅ Shows which variables are missing
- ✅ Logs configuration for debugging
- ✅ Prevents cryptic 404 errors later in pipeline

### 3. Enhanced Error Messages in `rag/embeddings.py` (lines 37-64, 99-109, 164-174)

**Added deployment name validation in `__init__`:**
```python
# Validate configuration
if not self.embedding_model:
    raise ValueError(
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME environment variable is not set. "
        "This is required for generating embeddings. Please set it in your .env file."
    )
```

**Added better error handling for 404 errors:**
```python
except Exception as e:
    error_msg = str(e)
    if "404" in error_msg or "Resource not found" in error_msg:
        logger.error(
            f"Embedding deployment '{self.embedding_model}' not found. "
            f"Please verify that this deployment exists in your Azure OpenAI resource. "
            f"Original error: {error_msg}"
        )
    else:
        logger.error(f"Error generating embedding: {error_msg}")
    raise
```

**Benefits:**
- ✅ Clear error message pointing to missing deployment
- ✅ Guides user to check Azure OpenAI resource
- ✅ Applied to both single and batch embedding methods

### 4. Updated HuggingFace Startup Script (lines 10-40)

```bash
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
```

**Benefits:**
- ✅ Validates environment variables before starting Python app
- ✅ Shows clear ✅/❌ status for each variable
- ✅ Fails early with deployment instructions
- ✅ Prevents wasted time debugging Python errors

### 5. Created Comprehensive Deployment Guide

**New file:** `HUGGINGFACE_DEPLOYMENT.md`

**Contents:**
- Complete list of required environment variables
- Step-by-step deployment instructions
- Common issues and solutions (including this 404 error)
- Azure OpenAI deployment verification steps
- Performance and cost considerations
- Security best practices

### 6. Updated README.md (lines 662-685)

Added prominent link to deployment guide and highlighted the **required** embedding deployment variable:

```markdown
**Required**: Add the following secrets in Space settings → Repository secrets:
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME` (e.g., `text-embedding-3-small`) ⚠️ **Required!**
```

## Testing

All fixes were tested locally:

1. ✅ Environment variable validation detects missing `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME`
2. ✅ EmbeddingGenerator raises clear error when deployment name is missing
3. ✅ App startup logs show all configuration values
4. ✅ Startup script validates environment variables before running Python

## How to Deploy the Fix to HuggingFace Spaces

### Option 1: Automated Deployment (Recommended)
```bash
git add .
git commit -m "Fix: Add missing AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME validation"
git push origin main
```
The GitHub Actions workflow will automatically sync to HuggingFace Spaces.

### Option 2: Manual Deployment
1. Push changes to your HuggingFace Space repository
2. **Critical**: Add the missing secret in HuggingFace Spaces:
   - Go to your Space → Settings → Repository secrets
   - Add new secret: `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME` = `text-embedding-3-small`
   - (Or whatever your actual Azure deployment name is)
3. The Space will rebuild and start successfully

## Verification

After deploying, you should see in the logs:

```
🔍 Checking environment variables...
✅ Found: AZURE_OPENAI_ENDPOINT
✅ Found: AZURE_OPENAI_API_KEY
✅ Found: AZURE_OPENAI_DEPLOYMENT_NAME
✅ Found: AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME
✅ All required environment variables are set!

🚀 Starting application...
2025-11-17 00:00:00,000 - app - INFO - Azure OpenAI Endpoint: https://your-resource.openai.azure.com/
2025-11-17 00:00:00,000 - app - INFO - LLM Deployment: gpt-4o-mini
2025-11-17 00:00:00,000 - app - INFO - Embedding Deployment: text-embedding-3-small
2025-11-17 00:00:00,000 - app - INFO - API Version: 2024-05-01-preview
```

## Prevention Measures

This fix includes multiple layers of defense to prevent similar issues:

1. **Example file accuracy**: `.env.example` now matches actual required variables
2. **Startup validation**: App fails fast with clear error message
3. **Component validation**: EmbeddingGenerator validates its own requirements
4. **Shell-level validation**: Startup script checks before Python runs
5. **Documentation**: Comprehensive deployment guide with troubleshooting
6. **Error messages**: 404 errors now explain which deployment is missing

## Files Modified

- ✅ `.env.example` - Fixed variable name and uncommented
- ✅ `app.py` - Added `validate_environment()` function
- ✅ `rag/embeddings.py` - Enhanced error messages and validation
- ✅ `huggingface_startup.sh` - Added environment variable checks
- ✅ `README.md` - Updated deployment section with required variables
- ✅ `HUGGINGFACE_DEPLOYMENT.md` - Created comprehensive guide (new file)
- ✅ `BUGFIX_HUGGINGFACE_404.md` - This document (new file)

## Related Issues

- This bug **only affected HuggingFace Spaces** deployment
- **Local development worked** because `.env` had the correct variable set
- The issue would have been **caught immediately** with these validation layers

## Lessons Learned

1. **Always validate environment on startup** - fail fast with clear errors
2. **Keep `.env.example` in sync** - it's the source of truth for deployments
3. **Multi-layer validation** - shell + Python + component level
4. **Better error messages** - 404 should explain what's missing
5. **Comprehensive documentation** - deployment guides prevent issues
