"""
LM Studio Provider Implementation.

Connects to local LM Studio instance using OpenAI-compatible API.
Best for: Local development, privacy, offline usage.
"""

import httpx
from typing import AsyncIterator, Optional, Any
from loguru import logger

from .base import BaseLLMProvider, ProviderConfig, LLMResponse, Message, ProviderType


class LMStudioProvider(BaseLLMProvider):
    """
    LM Studio LLM provider using OpenAI-compatible API.
    
    LM Studio runs locally and exposes an OpenAI-compatible API endpoint.
    Default URL: http://localhost:1234/v1
    """
    
    def __init__(self, config: Optional[ProviderConfig] = None):
        if config is None:
            config = ProviderConfig(
                provider_type=ProviderType.LMSTUDIO,
                base_url="http://localhost:1234/v1",
                model="local-model",
            )
        super().__init__(config)
        
        self.base_url = config.base_url or "http://localhost:1234/v1"
        self._client = httpx.Client(timeout=config.timeout)
        self._async_client = httpx.AsyncClient(timeout=config.timeout)
    
    def generate(
        self,
        messages: list[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using LM Studio."""
        
        payload = self._build_payload(messages, max_tokens, temperature, **kwargs)
        logger.debug(f"LM Studio payload: model={payload.get('model')}, messages={len(payload.get('messages', []))}")
        
        try:
            response = self._client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            data = response.json()
            
            return self._parse_response(data)
            
        except httpx.HTTPStatusError as e:
            # Log the actual error response from the server
            error_body = e.response.text if e.response else "No response body"
            logger.error(f"LM Studio request failed: {e}. Response: {error_body}")
            raise RuntimeError(f"LM Studio generation failed: {e}. Server response: {error_body}") from e
        except httpx.HTTPError as e:
            logger.error(f"LM Studio request failed: {e}")
            raise RuntimeError(f"LM Studio generation failed: {e}") from e
    
    async def generate_async(
        self,
        messages: list[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """Asynchronously generate response using LM Studio."""
        
        payload = self._build_payload(messages, max_tokens, temperature, **kwargs)
        
        try:
            response = await self._async_client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            data = response.json()
            
            return self._parse_response(data)
            
        except httpx.HTTPError as e:
            logger.error(f"LM Studio async request failed: {e}")
            raise RuntimeError(f"LM Studio generation failed: {e}") from e
    
    async def stream(
        self,
        messages: list[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream response tokens from LM Studio."""
        
        payload = self._build_payload(messages, max_tokens, temperature, stream=True, **kwargs)
        
        try:
            async with self._async_client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix
                        if data.strip() == "[DONE]":
                            break
                        
                        try:
                            import json
                            chunk = json.loads(data)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
                            
        except httpx.HTTPError as e:
            logger.error(f"LM Studio stream failed: {e}")
            raise RuntimeError(f"LM Studio streaming failed: {e}") from e
    
    def health_check(self) -> bool:
        """Check if LM Studio is running and responding."""
        try:
            response = self._client.get(f"{self.base_url}/models", timeout=5.0)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"LM Studio health check failed: {e}")
            return False
    
    def _build_payload(
        self,
        messages: list[Message],
        max_tokens: Optional[int],
        temperature: Optional[float],
        stream: bool = False,
        **kwargs
    ) -> dict:
        """Build request payload for LM Studio API."""
        
        payload = {
            "model": self.model,
            "messages": self._format_messages(messages),
            "stream": stream,
        }
        
        # Add optional parameters only if they are not None
        max_tokens_val = self._get_param("max_tokens", max_tokens)
        if max_tokens_val is not None:
            payload["max_tokens"] = max_tokens_val
            
        temp_val = self._get_param("temperature", temperature)
        if temp_val is not None:
            payload["temperature"] = temp_val
            
        # Add any other kwargs
        payload.update({k: v for k, v in kwargs.items() if v is not None})
        
        return payload
    
    def _parse_response(self, data: dict) -> LLMResponse:
        """Parse LM Studio API response."""
        
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = data.get("usage", {})
        
        return LLMResponse(
            content=message.get("content", ""),
            model=data.get("model", self.config.model),
            provider=self.name,
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            finish_reason=choice.get("finish_reason"),
            raw_response=data
        )
    
    def __del__(self):
        """Cleanup HTTP clients."""
        try:
            self._client.close()
        except Exception:
            pass
