import httpx
import json

base_url = "http://127.0.0.1:11434/v1"
try:
    response = httpx.get(f"{base_url}/models")
    print(f"Status Code: {response.status_code}")
    print("Models Response:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Failed to probe models: {e}")

try:
    # Try Ollama native tags too
    native_response = httpx.get("http://127.0.0.1:11434/api/tags")
    print(f"\nOllama Native Response ({native_response.status_code}):")
    print(json.dumps(native_response.json(), indent=2))
except Exception as e:
    print(f"Failed to probe native tags: {e}")
