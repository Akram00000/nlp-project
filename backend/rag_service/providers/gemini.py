"""
Google Gemini Provider Implementation.

Uses Google's Generative AI SDK for Gemini models.
Best for: Good Arabic support, cost-effective, large context windows.
"""

from typing import AsyncIterator, Optional
from loguru import logger

from .base import BaseLLMProvider, ProviderConfig, LLMResponse, Message, ProviderType


class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini LLM provider.
    
    Requires GEMINI_API_KEY environment variable or config.
    Supports: gemini-1.5-flash, gemini-1.5-pro, gemini-pro, etc.
    """
    
    def __init__(self, config: Optional[ProviderConfig] = None):
        if config is None:
            config = ProviderConfig(
                provider_type=ProviderType.GEMINI,
                model="gemini-1.5-flash",
            )
        elif config.model == "default":
            # Override default model with valid Gemini model
            config.model = "gemini-1.5-flash"
        super().__init__(config)
        
        self._client = None
        self._model = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Gemini client."""
        try:
            # Try new google-genai SDK first (v1.x)
            try:
                from google import genai
                from google.genai import types
                
                logger.debug(f"Gemini config.api_key: {self.config.api_key[:20] if self.config.api_key else 'None'}...")
                
                if not self.config.api_key:
                    import os
                    # Try both environment variable formats
                    api_key = os.getenv("RAG__GEMINI__API_KEY") or os.getenv("GEMINI_API_KEY")
                    if not api_key:
                        logger.warning("GEMINI_API_KEY not set. Gemini provider will not work.")
                        return
                else:
                    api_key = self.config.api_key
                
                # New API: Create client directly
                self._client = genai.Client(api_key=api_key)
                self._model_name = self.config.model
                self._initialized = True
                logger.info(f"Gemini provider initialized with model: {self.config.model} (google-genai v1.x)")
                
            except (ImportError, AttributeError):
                # Fallback to google-generativeai (older SDK)
                import google.generativeai as genai
                
                logger.debug(f"Gemini config.api_key: {self.config.api_key[:20] if self.config.api_key else 'None'}...")
                
                if not self.config.api_key:
                    import os
                    api_key = os.getenv("RAG__GEMINI__API_KEY") or os.getenv("GEMINI_API_KEY")
                    if not api_key:
                        logger.warning("GEMINI_API_KEY not set. Gemini provider will not work.")
                        return
                else:
                    api_key = self.config.api_key
                
                genai.configure(api_key=api_key)
                self._client = genai
                self._model = genai.GenerativeModel(self.config.model)
                self._initialized = True
                logger.info(f"Gemini provider initialized with model: {self.config.model} (google-generativeai)")
            
        except ImportError as e:
            logger.error(f"Google Generative AI package not installed: {e}")
            raise ImportError("Please install: pip install google-genai OR pip install google-generativeai")
    
    def generate(
        self,
        messages: list[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Gemini."""
        
        if not self._initialized:
            raise RuntimeError("Gemini provider not initialized. Check API key.")
        
        # Convert messages to Gemini format
        gemini_messages = self._convert_messages(messages)
        
        generation_config = self._build_generation_config(max_tokens, temperature, **kwargs)
        
        try:
            # Handle system prompt
            system_instruction = None
            for msg in messages:
                if msg.role == "system":
                    system_instruction = msg.content
                    break
            
            # Check which SDK is being used
            if hasattr(self._client, 'models'):
                # New google-genai SDK (v1.x)
                from google.genai import types
                
                config_dict = {}
                if max_tokens:
                    config_dict['max_output_tokens'] = max_tokens
                if temperature is not None:
                    config_dict['temperature'] = temperature
                if system_instruction:
                    config_dict['system_instruction'] = system_instruction
                
                generate_config = types.GenerateContentConfig(**config_dict) if config_dict else None
                
                # Convert messages to string format for new SDK
                # New SDK expects just the content string(s), not dict format
                content_parts = []
                for msg in gemini_messages:
                    if isinstance(msg, dict) and 'parts' in msg:
                        content_parts.extend(msg['parts'])
                    elif isinstance(msg, str):
                        content_parts.append(msg)
                
                # For single user message, pass as string. For multi-turn, use list format
                if len(content_parts) == 1:
                    contents = content_parts[0]
                else:
                    contents = "\n\n".join(content_parts)
                
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=contents,
                    config=generate_config
                )
            else:
                # Old google-generativeai SDK
                if system_instruction:
                    model = self._client.GenerativeModel(
                        self.config.model,
                        system_instruction=system_instruction
                    )
                else:
                    model = self._model
                
                response = model.generate_content(
                    gemini_messages,
                    generation_config=generation_config
                )
            
            return self._parse_response(response)
            
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise RuntimeError(f"Gemini generation failed: {e}") from e
    
    async def generate_async(
        self,
        messages: list[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """Asynchronously generate response using Gemini."""
        
        if not self._initialized:
            raise RuntimeError("Gemini provider not initialized. Check API key.")
        
        gemini_messages = self._convert_messages(messages)
        generation_config = self._build_generation_config(max_tokens, temperature, **kwargs)
        
        try:
            system_instruction = None
            for msg in messages:
                if msg.role == "system":
                    system_instruction = msg.content
                    break
            
            if system_instruction:
                model = self._client.GenerativeModel(
                    self.config.model,
                    system_instruction=system_instruction
                )
            else:
                model = self._model
            
            response = await model.generate_content_async(
                gemini_messages,
                generation_config=generation_config
            )
            
            return self._parse_response(response)
            
        except Exception as e:
            logger.error(f"Gemini async generation failed: {e}")
            raise RuntimeError(f"Gemini generation failed: {e}") from e
    
    async def stream(
        self,
        messages: list[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream response tokens from Gemini."""
        
        if not self._initialized:
            raise RuntimeError("Gemini provider not initialized. Check API key.")
        
        gemini_messages = self._convert_messages(messages)
        generation_config = self._build_generation_config(max_tokens, temperature, **kwargs)
        
        try:
            system_instruction = None
            for msg in messages:
                if msg.role == "system":
                    system_instruction = msg.content
                    break
            
            if system_instruction:
                model = self._client.GenerativeModel(
                    self.config.model,
                    system_instruction=system_instruction
                )
            else:
                model = self._model
            
            response = await model.generate_content_async(
                gemini_messages,
                generation_config=generation_config,
                stream=True
            )
            
            async for chunk in response:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            logger.error(f"Gemini stream failed: {e}")
            raise RuntimeError(f"Gemini streaming failed: {e}") from e
    
    def health_check(self) -> bool:
        """Check if Gemini API is accessible."""
        if not self._initialized:
            return False
        
        try:
            # Try listing models as a health check
            models = list(self._client.list_models())
            return len(models) > 0
        except Exception as e:
            logger.warning(f"Gemini health check failed: {e}")
            return False
    
    def _convert_messages(self, messages: list[Message]) -> list:
        """Convert messages to Gemini format."""
        gemini_messages = []
        
        for msg in messages:
            if msg.role == "system":
                # System messages handled separately via system_instruction
                continue
            elif msg.role == "user":
                gemini_messages.append({"role": "user", "parts": [msg.content]})
            elif msg.role == "assistant":
                gemini_messages.append({"role": "model", "parts": [msg.content]})
        
        # If only system message, create a user message from context
        if not gemini_messages:
            for msg in messages:
                if msg.role == "system":
                    gemini_messages.append({"role": "user", "parts": [msg.content]})
                    break
        
        return gemini_messages
    
    def _build_generation_config(
        self,
        max_tokens: Optional[int],
        temperature: Optional[float],
        **kwargs
    ) -> dict:
        """Build Gemini generation config."""
        
        return {
            "max_output_tokens": self._get_param("max_tokens", max_tokens),
            "temperature": self._get_param("temperature", temperature),
            **kwargs
        }
    
    def _parse_response(self, response) -> LLMResponse:
        """Parse Gemini API response."""
        
        # Extract text from response
        try:
            content = response.text
        except (AttributeError, ValueError):
            content = ""
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content.parts:
                    content = candidate.content.parts[0].text
        
        # Try to get usage metadata
        usage = {}
        if hasattr(response, 'usage_metadata'):
            usage = {
                "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
                "total_tokens": getattr(response.usage_metadata, 'total_token_count', 0),
            }
        
        return LLMResponse(
            content=content,
            model=self.config.model,
            provider=self.name,
            usage=usage,
            finish_reason="stop",
            raw_response=response
        )
