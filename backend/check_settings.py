"""Check what model the settings are actually loading."""
from rag_service.config.settings import settings

print(f"Default Provider: {settings.default_provider}")
print(f"LM Studio Base URL: {settings.lmstudio.base_url}")
print(f"LM Studio Model: {settings.lmstudio.model}")
print(f"LM Studio Max Tokens: {settings.lmstudio.max_tokens}")
print(f"LM Studio Temperature: {settings.lmstudio.temperature}")
