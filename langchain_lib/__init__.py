"""
Enterprise RAG system - OpenAI + Chroma with flexible data sources.
Streamlined implementation for intelligent JIRA ticket search and analysis.
"""

from .config import Config
from .enterprise_rag import EnterpriseRAG

__all__ = ["Config", "EnterpriseRAG"]
__version__ = "3.0.0" 