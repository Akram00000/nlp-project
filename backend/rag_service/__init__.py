"""
RAG Service - Custom Retrieval-Augmented Generation for Islamic Fatwa Chatbot

A provider-agnostic RAG system with integrated citation tracking and translation support.
"""

from .core.rag_engine import RAGEngine
from .config.settings import settings

__version__ = "0.1.0"
__all__ = ["RAGEngine", "settings"]
