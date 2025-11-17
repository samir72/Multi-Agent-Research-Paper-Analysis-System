#!/usr/bin/env python3
"""
Test Azure OpenAI LLM deployment with current API version.
"""
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

def test_llm_deployment():
    """Test LLM deployment with current API version."""
    print("=" * 80)
    print("Testing Azure OpenAI LLM Deployment")
    print("=" * 80)
    print()

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

    print(f"Endpoint: {endpoint}")
    print(f"Deployment: {deployment_name}")
    print(f"API Version: {api_version}")
    print()
    print("Sending test request...")
    print()

    try:
        client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )

        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello, world!' if you can read this."}
            ],
            temperature=0,
            max_tokens=50
        )

        message = response.choices[0].message.content
        tokens_used = response.usage.total_tokens

        print(f"✅ SUCCESS: LLM responded successfully!")
        print(f"   Response: {message}")
        print(f"   Model: {deployment_name}")
        print(f"   Tokens used: {tokens_used}")
        print(f"   API Version: {api_version}")
        print()
        print("=" * 80)
        print("✅ LLM deployment works with API version:", api_version)
        print("=" * 80)
        return True

    except Exception as e:
        error_msg = str(e)
        print(f"❌ ERROR: LLM request failed")
        print()
        print(f"Error message: {error_msg}")
        print()

        if "404" in error_msg or "Resource not found" in error_msg:
            print("DIAGNOSIS: Deployment not found with API version", api_version)
            print()
            print("Possible solutions:")
            print("  1. Your LLM deployment might require a different API version")
            print("  2. Try API version 2024-07-18 for gpt-4o-mini")
            print("  3. You may need separate API versions for LLM vs embeddings")
            print()
        elif "401" in error_msg:
            print("DIAGNOSIS: Authentication failed")
            print()

        print("=" * 80)
        print("❌ LLM deployment test FAILED")
        print("=" * 80)
        return False

if __name__ == "__main__":
    test_llm_deployment()
