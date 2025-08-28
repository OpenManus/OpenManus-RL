#!/usr/bin/env python3
"""Test script to verify API connectivity and configuration."""

import os
import sys
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / '.env')

def test_env_variables():
    """Test that environment variables are loaded correctly."""
    print("=== Environment Variables ===")
    
    # Check which variables are set
    api_key = os.getenv('OPENAI_API_KEY') or os.getenv('OAI_KEY')
    api_base = os.getenv('OPENAI_API_BASE') or os.getenv('OAI_ENDPOINT')
    
    print(f"OPENAI_API_KEY: {'✓ Set' if os.getenv('OPENAI_API_KEY') else '✗ Not set'}")
    print(f"OPENAI_API_BASE: {os.getenv('OPENAI_API_BASE', 'Not set')}")
    print(f"OAI_KEY: {'✓ Set' if os.getenv('OAI_KEY') else '✗ Not set'}")
    print(f"OAI_ENDPOINT: {os.getenv('OAI_ENDPOINT', 'Not set')}")
    
    if api_key:
        print(f"Using API key: {api_key[:10]}...")
    if api_base:
        print(f"Using API base: {api_base}")
    
    return api_key, api_base

def test_openai_api(api_key, api_base):
    """Test OpenAI API connection."""
    print("\n=== Testing OpenAI API ===")
    
    if not api_key:
        print("❌ No API key found")
        return False
    
    if not api_base:
        api_base = "https://api.openai.com"
    
    url = f"{api_base}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": "Say 'API test successful' if you can read this."}
        ],
        "max_tokens": 50,
        "temperature": 0
    }
    
    try:
        print(f"Connecting to: {url}")
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            print(f"✅ API Response: {content}")
            return True
        else:
            print(f"❌ API Error: {response.text[:200]}")
            return False
            
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection Error: {e}")
        return False
    except Exception as e:
        print(f"❌ API Exception: {e}")
        return False

def test_custom_endpoint(endpoint):
    """Test custom API endpoint."""
    print(f"\n=== Testing Custom Endpoint: {endpoint} ===")
    
    try:
        # Test basic connectivity
        response = requests.get(endpoint, timeout=5)
        print(f"✅ Endpoint reachable, status: {response.status_code}")
        return True
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection refused to {endpoint}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all API tests."""
    print("OpenManus-RL API Connectivity Test")
    print("=" * 40)
    
    # Test environment variables
    api_key, api_base = test_env_variables()
    
    success = False
    
    # Test OpenAI API if configured
    if api_key and api_base and 'openai.com' in api_base:
        success = test_openai_api(api_key, api_base)
    
    # Test custom endpoint if configured
    custom_endpoint = os.getenv('OAI_ENDPOINT')
    if custom_endpoint:
        success = test_custom_endpoint(custom_endpoint) or success
    
    print(f"\n=== Test Summary ===")
    if success:
        print("✅ At least one API endpoint is working")
        return 0
    else:
        print("❌ No working API endpoints found")
        print("\nSuggestions:")
        print("1. Check your .env file has valid OPENAI_API_KEY")
        print("2. Ensure OPENAI_API_BASE is set to https://api.openai.com")
        print("3. If using custom endpoint, ensure it's running and accessible")
        return 1

if __name__ == "__main__":
    sys.exit(main())