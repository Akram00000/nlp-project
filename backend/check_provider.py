"""Check what model the provider is actually using at runtime."""
from rag_service.providers.factory import get_provider

provider = get_provider()
print(f"Provider: {provider.name}")
print(f"Model from config: {provider.config.model}")
print(f"Model property: {provider.model}")
print(f"Base URL: {provider.base_url}")
