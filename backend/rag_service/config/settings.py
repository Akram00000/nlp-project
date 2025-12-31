"""
Configuration management for RAG Service using Pydantic Settings.

Supports environment variables and .env files for configuration.
"""

from typing import Optional, Literal
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class QdrantSettings(BaseModel):
    """Qdrant vector database settings."""
    
    host: str = Field(default="localhost", description="Qdrant server host")
    port: int = Field(default=6333, description="Qdrant server port")
    collection_name: str = Field(default="fatwas", description="Collection name for fatwas")
    prefer_grpc: bool = Field(default=False, description="Use gRPC instead of HTTP")
    api_key: Optional[str] = Field(default=None, description="Qdrant API key if using cloud")


class LMStudioSettings(BaseModel):
    """LM Studio provider settings."""
    
    base_url: str = Field(default="http://localhost:1234/v1", description="LM Studio API URL")
    model: str = Field(default="local-model", description="Model identifier")
    max_tokens: int = Field(default=4096, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, description="Generation temperature")
    timeout: float = Field(default=120.0, description="Request timeout in seconds")


class GeminiSettings(BaseModel):
    """Google Gemini provider settings."""
    
    api_key: Optional[str] = Field(default=None, description="Gemini API key")
    model: str = Field(default="gemini-1.5-flash", description="Gemini model name")
    max_tokens: int = Field(default=8192, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, description="Generation temperature")


class OpenAICompatibleSettings(BaseModel):
    """OpenAI-compatible API settings (for various providers)."""
    
    api_key: Optional[str] = Field(default=None, description="API key")
    base_url: str = Field(default="https://api.openai.com/v1", description="API base URL")
    model: str = Field(default="gpt-4", description="Model name")
    max_tokens: int = Field(default=4096, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, description="Generation temperature")
    timeout: float = Field(default=120.0, description="Request timeout in seconds")


class HuggingFaceSettings(BaseModel):
    """Hugging Face provider settings."""
    
    api_key: Optional[str] = Field(default=None, description="Hugging Face API token")
    base_url: str = Field(default="https://api-inference.huggingface.co/v1/", description="Hugging Face API base URL")
    model: str = Field(default="meta-llama/Llama-3.2-3B-Instruct", description="Model name")
    max_tokens: int = Field(default=4096, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, description="Generation temperature")
    timeout: float = Field(default=120.0, description="Request timeout in seconds")


class EmbeddingSettings(BaseModel):
    """Embedding model settings."""
    
    model_name: str = Field(
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        description="Sentence transformer model for embeddings"
    )
    device: str = Field(default="cpu", description="Device for embedding model (cpu/cuda)")
    batch_size: int = Field(default=32, description="Batch size for embedding")


class RetrievalSettings(BaseModel):
    """Retrieval configuration."""
    
    top_k: int = Field(default=5, description="Number of documents to retrieve")
    similarity_threshold: float = Field(default=0.15, description="Minimum similarity score")
    rerank: bool = Field(default=True, description="Enable re-ranking of results")
    rerank_model: Optional[str] = Field(default=None, description="Re-ranker model path")
    hybrid_search: bool = Field(default=False, description="Enable hybrid search (dense + sparse)")


class CitationSettings(BaseModel):
    """Citation formatting settings."""
    
    format: Literal["islamic_scholarly", "simple", "detailed"] = Field(
        default="islamic_scholarly",
        description="Citation format style"
    )
    include_inline: bool = Field(default=True, description="Include inline citations in response")
    include_references: bool = Field(default=True, description="Include reference list at end")
    include_confidence: bool = Field(default=True, description="Include confidence scores")


class Settings(BaseSettings):
    """Main settings for RAG Service."""
    
    # Provider selection
    default_provider: Literal["lmstudio", "gemini", "openai", "huggingface"] = Field(
        default="lmstudio",
        description="Default LLM provider to use"
    )
    
    # Sub-settings with defaults
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    lmstudio: LMStudioSettings = Field(default_factory=LMStudioSettings)
    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    openai: OpenAICompatibleSettings = Field(default_factory=OpenAICompatibleSettings)
    huggingface: HuggingFaceSettings = Field(default_factory=HuggingFaceSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    citation: CitationSettings = Field(default_factory=CitationSettings)
    
    # Paths
    data_dir: Path = Field(default=Path("data"), description="Data directory path")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Safety
    enable_safety_checks: bool = Field(default=True, description="Enable content safety checks")
    
    def __init__(self, **kwargs):
        """Initialize settings and manually override nested models from environment variables."""
        super().__init__(**kwargs)
        
        import os
        
        # Manually populate nested settings from environment variables
        # This is a workaround for Pydantic's nested settings not reading env vars properly
        
        # Gemini settings
        if 'RAG__GEMINI__API_KEY' in os.environ:
            self.gemini.api_key = os.environ['RAG__GEMINI__API_KEY']
        if 'RAG__GEMINI__MODEL' in os.environ:
            self.gemini.model = os.environ['RAG__GEMINI__MODEL']
        if 'RAG__GEMINI__MAX_TOKENS' in os.environ:
            self.gemini.max_tokens = int(os.environ['RAG__GEMINI__MAX_TOKENS'])
        if 'RAG__GEMINI__TEMPERATURE' in os.environ:
            self.gemini.temperature = float(os.environ['RAG__GEMINI__TEMPERATURE'])
            
        # LM Studio settings  
        if 'RAG__LMSTUDIO__BASE_URL' in os.environ:
            self.lmstudio.base_url = os.environ['RAG__LMSTUDIO__BASE_URL']
        elif 'LMSTUDIO_BASE_URL' in os.environ:
            self.lmstudio.base_url = os.environ['LMSTUDIO_BASE_URL']
            
        if 'RAG__LMSTUDIO__MODEL' in os.environ:
            self.lmstudio.model = os.environ['RAG__LMSTUDIO__MODEL']
        elif 'LMSTUDIO_MODEL' in os.environ:
            self.lmstudio.model = os.environ['LMSTUDIO_MODEL']
        if 'RAG__LMSTUDIO__MAX_TOKENS' in os.environ:
            self.lmstudio.max_tokens = int(os.environ['RAG__LMSTUDIO__MAX_TOKENS'])
        if 'RAG__LMSTUDIO__TEMPERATURE' in os.environ:
            self.lmstudio.temperature = float(os.environ['RAG__LMSTUDIO__TEMPERATURE'])
        if 'RAG__LMSTUDIO__TIMEOUT' in os.environ:
            self.lmstudio.timeout = float(os.environ['RAG__LMSTUDIO__TIMEOUT'])
            
        # Hugging Face settings
        if 'RAG__HUGGINGFACE__API_KEY' in os.environ:
            self.huggingface.api_key = os.environ['RAG__HUGGINGFACE__API_KEY']
        if 'RAG__HUGGINGFACE__BASE_URL' in os.environ:
            self.huggingface.base_url = os.environ['RAG__HUGGINGFACE__BASE_URL']
        if 'RAG__HUGGINGFACE__MODEL' in os.environ:
            self.huggingface.model = os.environ['RAG__HUGGINGFACE__MODEL']
        if 'RAG__HUGGINGFACE__MAX_TOKENS' in os.environ:
            self.huggingface.max_tokens = int(os.environ['RAG__HUGGINGFACE__MAX_TOKENS'])
        if 'RAG__HUGGINGFACE__TEMPERATURE' in os.environ:
            self.huggingface.temperature = float(os.environ['RAG__HUGGINGFACE__TEMPERATURE'])
        if 'RAG__HUGGINGFACE__TIMEOUT' in os.environ:
            self.huggingface.timeout = float(os.environ['RAG__HUGGINGFACE__TIMEOUT'])
            
        # Qdrant settings
        if 'RAG__QDRANT__HOST' in os.environ:
            self.qdrant.host = os.environ['RAG__QDRANT__HOST']
        if 'RAG__QDRANT__PORT' in os.environ:
            self.qdrant.port = int(os.environ['RAG__QDRANT__PORT'])
        if 'RAG__QDRANT__COLLECTION_NAME' in os.environ:
            self.qdrant.collection_name = os.environ['RAG__QDRANT__COLLECTION_NAME']
        if 'RAG__QDRANT__API_KEY' in os.environ:
            self.qdrant.api_key = os.environ['RAG__QDRANT__API_KEY']
            
        # Embedding settings
        if 'RAG__EMBEDDING__MODEL_NAME' in os.environ:
            self.embedding.model_name = os.environ['RAG__EMBEDDING__MODEL_NAME']
        if 'RAG__EMBEDDING__DEVICE' in os.environ:
            self.embedding.device = os.environ['RAG__EMBEDDING__DEVICE']
        if 'RAG__EMBEDDING__BATCH_SIZE' in os.environ:
            self.embedding.batch_size = int(os.environ['RAG__EMBEDDING__BATCH_SIZE'])
            
        # Retrieval settings
        if 'RAG__RETRIEVAL__TOP_K' in os.environ:
            self.retrieval.top_k = int(os.environ['RAG__RETRIEVAL__TOP_K'])
        if 'RAG__RETRIEVAL__SIMILARITY_THRESHOLD' in os.environ:
            self.retrieval.similarity_threshold = float(os.environ['RAG__RETRIEVAL__SIMILARITY_THRESHOLD'])
        if 'RAG__RETRIEVAL__RERANK' in os.environ:
            self.retrieval.rerank = os.environ['RAG__RETRIEVAL__RERANK'].lower() in ('true', '1', 'yes')
        if 'RAG__RETRIEVAL__RERANK_MODEL' in os.environ:
            self.retrieval.rerank_model = os.environ['RAG__RETRIEVAL__RERANK_MODEL']
            
        # Citation settings
        if 'RAG__CITATION__FORMAT' in os.environ:
            self.citation.format = os.environ['RAG__CITATION__FORMAT']
        if 'RAG__CITATION__INCLUDE_INLINE' in os.environ:
            self.citation.include_inline = os.environ['RAG__CITATION__INCLUDE_INLINE'].lower() in ('true', '1', 'yes')
        if 'RAG__CITATION__INCLUDE_REFERENCES' in os.environ:
            self.citation.include_references = os.environ['RAG__CITATION__INCLUDE_REFERENCES'].lower() in ('true', '1', 'yes')
        if 'RAG__CITATION__INCLUDE_CONFIDENCE' in os.environ:
            self.citation.include_confidence = os.environ['RAG__CITATION__INCLUDE_CONFIDENCE'].lower() in ('true', '1', 'yes')
    
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=str(Path(__file__).parent.parent / ".env"),
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        nested_model_default_partial_update=True,  # Allow nested models to update from env vars
        case_sensitive=False,
        extra="ignore"
    )


# Load .env file explicitly before creating settings instance
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_file, override=False)  # Only load if not already set

# Global settings instance
settings = Settings()
