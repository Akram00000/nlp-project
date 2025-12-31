"""Test Ollama API directly to diagnose 400 error."""
import httpx
import json

base_url = "http://127.0.0.1:11434/v1"

# Simple test payload
payload = {
    "model": "qwen/qwen2.5-vl-7b",
    "messages": [
        {"role": "user", "content": "Hello, how are you?"}
    ]
}

print(f"Testing: {base_url}/chat/completions")
print(f"Payload: {json.dumps(payload, indent=2)}")
print()

try:
    response = httpx.post(
        f"{base_url}/chat/completions",
        json=payload,
        timeout=60.0
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text[:500]}")
except Exception as e:
    print(f"Error: {e}")
