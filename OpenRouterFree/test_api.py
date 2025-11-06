"""
Test script to verify OpenRouter API connection and functionality.
This script makes a simple test call to the API to ensure everything is configured correctly.
"""

import os
import requests
import json

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
YOUR_SITE_URL = "http://localhost"
YOUR_SITE_NAME = "PyCharmMiscProject"

def test_api_connection():
    """Test the OpenRouter API connection with a simple prompt."""

    print("=" * 70)
    print("OpenRouter API Connection Test")
    print("=" * 70)
    print()

    # Check API key
    if not OPENROUTER_API_KEY:
        print("[ERROR] OPENROUTER_API_KEY not found!")
        print("[INFO] Please set it using: set OPENROUTER_API_KEY=your_api_key")
        return False

    print(f"[OK] API Key found: {OPENROUTER_API_KEY[:10]}...")
    print()

    # Test with a simple model
    test_model = "meta-llama/llama-3.3-8b-instruct:free"
    test_prompt = "What is 2+2? Answer with just the number."

    print(f"[INFO] Testing with model: {test_model}")
    print(f"[INFO] Test prompt: {test_prompt}")
    print()

    try:
        # Prepare API request
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": YOUR_SITE_URL,
            "X-Title": YOUR_SITE_NAME,
        }

        data = {
            "model": test_model,
            "messages": [
                {
                    "role": "user",
                    "content": test_prompt
                }
            ]
        }

        print("[INFO] Sending request to OpenRouter API...")
        response = requests.post(
            url=OPENROUTER_API_URL,
            headers=headers,
            data=json.dumps(data),
            timeout=30
        )

        response.raise_for_status()
        result = response.json()

        # Extract answer
        if 'choices' in result and len(result['choices']) > 0:
            answer = result['choices'][0]['message']['content'].strip()
            print("[OK] API call successful!")
            print()
            print(f"Response: {answer}")
            print()

            # Show usage stats if available
            if 'usage' in result:
                usage = result['usage']
                print("[STATS] Token usage:")
                print(f"  Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
                print(f"  Completion tokens: {usage.get('completion_tokens', 'N/A')}")
                print(f"  Total tokens: {usage.get('total_tokens', 'N/A')}")

            print()
            print("=" * 70)
            print("[SUCCESS] API connection test passed!")
            print("=" * 70)
            return True
        else:
            print("[ERROR] Unexpected API response format")
            print(f"Response: {result}")
            return False

    except requests.exceptions.Timeout:
        print("[ERROR] Request timeout - the API is taking too long to respond")
        return False
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] API request failed: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"[ERROR] API error details: {json.dumps(error_detail, indent=2)}")
            except:
                print(f"[ERROR] Response status code: {e.response.status_code}")
                print(f"[ERROR] Response text: {e.response.text}")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_api_connection()
    if success:
        print()
        print("[INFO] You can now run the main script using: execute.bat")
    else:
        print()
        print("[INFO] Please fix the errors above before running the main script")

    print()
    input("Press Enter to exit...")

