#!/usr/bin/env python3
"""
Validate that the configured Azure OpenAI resource supports the Responses
API before enabling USE_RESPONSES_API=true (see .env.example and
agents/analyzer.py / agents/synthesis.py).

Responses API requires a specific minimum preview api-version and has
historically had narrower deployment/region availability than Chat
Completions -- this is a required go/no-go check, not an assumption
(this codebase has been burned before by unverified API-version
assumptions; see AZURE_API_VERSION_FIX.md and CLAUDE.md's version history).

Run: python scripts/validate_responses_api.py
"""
import os
import sys

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# Ordered oldest -> newest. Azure OpenAI's own error message names the exact
# minimum version it requires, so this list only needs to bracket that value
# closely enough to report it -- it is not meant to be exhaustive.
CANDIDATE_VERSIONS = [
    "2024-05-01-preview",
    "2024-08-01-preview",
    "2024-10-01-preview",
    "2024-12-01-preview",
    "2025-01-01-preview",
    "2025-02-01-preview",
    "2025-03-01-preview",
    "2025-04-01-preview",
]


def _try_responses_call(client: AzureOpenAI, deployment: str) -> tuple[bool, str]:
    try:
        response = client.responses.create(
            model=deployment,
            instructions="You are a helpful assistant.",
            input="Say hello in one word.",
            max_output_tokens=20,
        )
        return True, response.output_text
    except Exception as e:
        return False, str(e)


def main() -> bool:
    print("=" * 80)
    print("Validating Azure OpenAI Responses API availability")
    print("=" * 80)
    print()

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    configured_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

    print(f"Endpoint: {endpoint}")
    print(f"Deployment: {deployment}")
    print(f"Currently configured AZURE_OPENAI_API_VERSION: {configured_version}")
    print()

    # Step 1: does the *currently configured* version work?
    client = AzureOpenAI(api_key=api_key, api_version=configured_version, azure_endpoint=endpoint)
    ok, detail = _try_responses_call(client, deployment)
    if ok:
        print(f"✅ Responses API works with your CURRENT AZURE_OPENAI_API_VERSION ({configured_version}).")
        print(f"   Sample output: {detail!r}")
        print()
        print("Safe to set USE_RESPONSES_API=true without changing AZURE_OPENAI_API_VERSION.")
        return True

    print(f"❌ Responses API failed at your current API version ({configured_version}):")
    print(f"   {detail[:200]}")
    print()
    print("Probing newer API versions to find the minimum that works...")
    print()

    working_version = None
    for v in CANDIDATE_VERSIONS:
        client = AzureOpenAI(api_key=api_key, api_version=v, azure_endpoint=endpoint)
        ok, detail = _try_responses_call(client, deployment)
        status = "✅ SUCCESS" if ok else "❌ failed"
        print(f"  {v}: {status}" + ("" if ok else f" ({detail[:120]})"))
        if ok:
            working_version = v
            break

    print()
    if not working_version:
        print("=" * 80)
        print("❌ Responses API is NOT available on this resource/deployment/region")
        print("   at any of the probed API versions.")
        print("   USE_RESPONSES_API=true would fall back to Chat Completions on every")
        print("   call (the code degrades gracefully), but that means no benefit from")
        print("   enabling the flag -- leave it false until this is resolved with Azure.")
        print("=" * 80)
        return False

    print("=" * 80)
    print(f"✅ Responses API works starting at api-version {working_version}")
    print()
    print("Before enabling USE_RESPONSES_API=true, also confirm the embeddings call")
    print("(rag/embeddings.py) and Chat Completions fallback path still work if you")
    print("bump AZURE_OPENAI_API_VERSION -- both currently read the same env var.")
    print("=" * 80)

    # Step 2: confirm embeddings and Chat Completions still work at the
    # newer version, since rag/embeddings.py reads the SAME
    # AZURE_OPENAI_API_VERSION env var (falls back to "2024-02-01" only if
    # unset) -- bumping it for Responses API is a global change, not scoped
    # to the analyzer/synthesis agents.
    embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
    client = AzureOpenAI(api_key=api_key, api_version=working_version, azure_endpoint=endpoint)

    if embedding_deployment:
        try:
            resp = client.embeddings.create(input="test", model=embedding_deployment)
            print(f"✅ Embeddings still work at {working_version} (dim={len(resp.data[0].embedding)})")
        except Exception as e:
            print(f"⚠️  Embeddings FAILED at {working_version}: {str(e)[:150]}")
            print("   Do not bump AZURE_OPENAI_API_VERSION globally without resolving this --")
            print("   consider a separate api_version for rag/embeddings.py instead.")

    try:
        resp = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": "Say hi in one word."}],
            temperature=0,
            max_tokens=10,
        )
        print(f"✅ Chat Completions fallback path still works at {working_version}")
    except Exception as e:
        print(f"⚠️  Chat Completions FAILED at {working_version}: {str(e)[:150]}")

    print()
    print(f"Next step: set AZURE_OPENAI_API_VERSION={working_version} in .env, then")
    print("USE_RESPONSES_API=true.")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
