#!/usr/bin/env python3
"""
Diagnostic script to validate Azure OpenAI embeddings deployment.

This script helps diagnose 404 errors related to embedding deployments.
Run this before deploying to HuggingFace Spaces to ensure configuration is correct.

Usage:
    python scripts/validate_azure_embeddings.py
"""
import os
import sys
from pathlib import Path
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def validate_azure_config():
    """Validate Azure OpenAI configuration."""
    print("=" * 80)
    print("Azure OpenAI Embeddings Deployment Validator")
    print("=" * 80)
    print()

    # Check required environment variables
    required_vars = {
        "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    }

    print("1. Checking environment variables...")
    print("-" * 80)
    missing_vars = []
    for var_name, var_value in required_vars.items():
        if var_value:
            # Mask sensitive values
            if "KEY" in var_name:
                display_value = f"{var_value[:10]}...{var_value[-4:]}" if len(var_value) > 14 else "***"
            else:
                display_value = var_value
            print(f"✅ {var_name}: {display_value}")
        else:
            print(f"❌ {var_name}: NOT SET")
            missing_vars.append(var_name)

    print()

    if missing_vars:
        print(f"ERROR: Missing required environment variables: {', '.join(missing_vars)}")
        print()
        print("Fix: Add these variables to your .env file or HuggingFace Spaces secrets")
        return False

    print("2. Testing embeddings deployment...")
    print("-" * 80)

    try:
        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            api_key=required_vars["AZURE_OPENAI_API_KEY"],
            api_version=required_vars["AZURE_OPENAI_API_VERSION"],
            azure_endpoint=required_vars["AZURE_OPENAI_ENDPOINT"]
        )

        deployment_name = required_vars["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"]
        print(f"Testing deployment: {deployment_name}")
        print()

        # Try to generate a test embedding
        test_text = "This is a test embedding."
        response = client.embeddings.create(
            input=test_text,
            model=deployment_name
        )

        embedding = response.data[0].embedding
        embedding_dim = len(embedding)

        print(f"✅ SUCCESS: Embedding generated successfully!")
        print(f"   Embedding dimension: {embedding_dim}")
        print(f"   Model used: {deployment_name}")
        print()
        print("=" * 80)
        print("✅ All checks passed! Your Azure OpenAI embeddings configuration is correct.")
        print("=" * 80)
        return True

    except Exception as e:
        error_msg = str(e)
        print(f"❌ ERROR: Failed to generate embedding")
        print()
        print(f"Error message: {error_msg}")
        print()

        # Provide helpful diagnostics
        if "404" in error_msg or "Resource not found" in error_msg:
            print("DIAGNOSIS: Deployment not found (404 error)")
            print()
            print("Possible causes:")
            print("  1. Deployment name is incorrect")
            print("  2. Deployment doesn't exist in your Azure OpenAI resource")
            print("  3. Deployment is in a different Azure region/resource")
            print()
            print("How to fix:")
            print("  Option A: Create the deployment in Azure Portal")
            print("    1. Go to https://portal.azure.com")
            print("    2. Navigate to your Azure OpenAI resource")
            print("    3. Go to 'Model deployments' → 'Manage Deployments'")
            print("    4. Create a new deployment:")
            print(f"       - Model: text-embedding-3-small (or text-embedding-ada-002)")
            print(f"       - Deployment name: {deployment_name}")
            print()
            print("  Option B: Use existing deployment")
            print("    1. Check what embedding deployments you already have in Azure Portal")
            print("    2. Update AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME to match existing deployment")
            print("    3. Common deployment names:")
            print("       - text-embedding-3-small")
            print("       - text-embedding-ada-002")
            print("       - embedding")
            print()

        elif "401" in error_msg or "Unauthorized" in error_msg:
            print("DIAGNOSIS: Authentication failed (401 error)")
            print()
            print("How to fix:")
            print("  1. Verify AZURE_OPENAI_API_KEY is correct")
            print("  2. Check that the key hasn't expired")
            print("  3. Ensure the key matches the Azure OpenAI resource")
            print()

        elif "InvalidRequestError" in error_msg:
            print("DIAGNOSIS: Invalid request to Azure OpenAI API")
            print()
            print("How to fix:")
            print("  1. Check AZURE_OPENAI_API_VERSION (try '2024-02-01' or '2024-05-01-preview')")
            print("  2. Verify AZURE_OPENAI_ENDPOINT format (should end with '/')")
            print()

        print("=" * 80)
        print("❌ Configuration validation FAILED")
        print("=" * 80)
        return False


def list_common_deployment_names():
    """List common embedding deployment names."""
    print()
    print("Common embedding deployment names to try:")
    print("  - text-embedding-3-small (recommended, most cost-effective)")
    print("  - text-embedding-3-large (higher quality, more expensive)")
    print("  - text-embedding-ada-002 (legacy, widely supported)")
    print("  - embedding (generic name, check your Azure portal)")
    print()


if __name__ == "__main__":
    print()
    success = validate_azure_config()

    if not success:
        list_common_deployment_names()
        sys.exit(1)

    print()
    print("Next steps:")
    print("  1. If deploying to HuggingFace Spaces:")
    print("     - Add all Azure OpenAI secrets to HuggingFace Spaces settings")
    print("     - Ensure AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME matches your Azure deployment")
    print("  2. Run the application:")
    print("     python app.py")
    print()
    sys.exit(0)
