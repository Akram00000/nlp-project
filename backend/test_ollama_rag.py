"""Test Ollama with a larger context similar to RAG."""
import httpx
import json

base_url = "http://127.0.0.1:11434/v1"

# Simulating the RAG system prompt and longer context
system_prompt = """أنت عالم إسلامي متخصص. أجب على الأسئلة بناءً على المصادر المقدمة فقط.
You are an Islamic scholar. Answer based only on the provided sources."""

user_prompt = """السؤال / Question:
ما حكم صلاة الجماعة؟

المصادر / Sources:
[1] حكم صلاة الجماعة سنة مؤكدة...
[2] قال الإمام النووي في المجموع...

الجواب / Answer:"""

payload = {
    "model": "qwen/qwen2.5-vl-7b",
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    "max_tokens": 2000,
    "temperature": 0.7
}

print(f"Testing with RAG-like payload...")
print(f"Model: {payload['model']}")
print()

try:
    response = httpx.post(
        f"{base_url}/chat/completions",
        json=payload,
        timeout=120.0
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
        print(f"Response: {content[:300]}...")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Exception: {e}")
