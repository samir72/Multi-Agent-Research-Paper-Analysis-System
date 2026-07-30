#!/usr/bin/env python3
"""
Diagnostic script to validate LangFuse API key credentials.

Run this before starting the app to confirm LANGFUSE_PUBLIC_KEY /
LANGFUSE_SECRET_KEY in your .env actually authenticate against LangFuse.

Usage:
    python scripts/validate_langfuse_keys.py
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Allow running as `python scripts/validate_langfuse_keys.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.langfuse_client import check_langfuse_auth


def validate_langfuse_config() -> bool:
    """Validate LangFuse configuration and credentials."""
    print("=" * 80)
    print("LangFuse Key Validator")
    print("=" * 80)
    print()

    enabled = os.getenv("LANGFUSE_ENABLED", "true").lower() == "true"
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
    host = os.getenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")

    print("1. Checking environment variables...")
    print("-" * 80)
    print(f"✅ LANGFUSE_ENABLED: {enabled}")
    print(f"{'✅' if public_key else '❌'} LANGFUSE_PUBLIC_KEY: {public_key or 'NOT SET'}")
    secret_display = f"{secret_key[:6]}...{secret_key[-4:]}" if len(secret_key) > 10 else ("***" if secret_key else "NOT SET")
    print(f"{'✅' if secret_key else '❌'} LANGFUSE_SECRET_KEY: {secret_display}")
    print(f"✅ LANGFUSE_BASE_URL: {host}")
    print()

    if not enabled:
        print("LangFuse is disabled via LANGFUSE_ENABLED=false — nothing to validate.")
        return True

    if not public_key or not secret_key:
        print("ERROR: Missing required LangFuse API keys.")
        print()
        print("Fix: Add LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to your .env file.")
        print("     Get them from https://cloud.langfuse.com -> Project Settings -> API Keys")
        return False

    print("2. Testing authentication against LangFuse API...")
    print("-" * 80)
    ok, message = check_langfuse_auth()

    if ok:
        print(f"✅ SUCCESS: {message}")
        print()
        print("=" * 80)
        print("✅ All checks passed! Your LangFuse configuration is correct.")
        print("=" * 80)
        return True

    print(f"❌ ERROR: {message}")
    print()

    if "401" in message or "Unauthorized" in message:
        print("DIAGNOSIS: Authentication failed (401)")
        print("  1. Verify LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are correct and match the same project")
        print("  2. Check for extra whitespace or truncated copy/paste")
    elif "403" in message:
        print("DIAGNOSIS: Access denied (403)")
        print("  Keys parsed but access was refused — check project/org permissions in LangFuse.")
    elif any(s in message for s in ("Connection", "timeout", "resolve", "Timeout", "Name or service")):
        print("DIAGNOSIS: Cannot reach LangFuse host")
        print(f"  1. Verify LANGFUSE_BASE_URL ({host}) is correct")
        print("  2. If self-hosting, confirm the server is running and reachable")
    else:
        print("DIAGNOSIS: See error message above for details.")

    print()
    print("=" * 80)
    print("❌ Configuration validation FAILED")
    print("=" * 80)
    return False


if __name__ == "__main__":
    print()
    success = validate_langfuse_config()
    print()
    if not success:
        sys.exit(1)
    print("Next steps:")
    print("  python app.py")
    print()
    sys.exit(0)
