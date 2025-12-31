import httpx
import json

def probe():
    base_url = "http://127.0.0.1:11434"
    try:
        # Try OpenAI models endpoint
        r = httpx.get(f"{base_url}/v1/models")
        print("--- OpenAI-Compatible Models ---")
        if r.status_code == 200:
            data = r.json()
            for m in data.get('data', []):
                print(f"ID: {m.get('id')}")
        else:
            print(f"Status: {r.status_code}")
    except Exception as e:
        print(f"Error querying /v1/models: {e}")

    try:
        # Try Ollama native endpoint
        r = httpx.get(f"{base_url}/api/tags")
        print("\n--- Ollama Native Models ---")
        if r.status_code == 200:
            data = r.json()
            for m in data.get('models', []):
                print(f"Name: {m.get('name')}")
        else:
            print(f"Status: {r.status_code}")
    except Exception as e:
        print(f"Error querying /api/tags: {e}")

if __name__ == "__main__":
    probe()
