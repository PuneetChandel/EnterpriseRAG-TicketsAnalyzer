"""
Enterprise RAG Configuration - OpenAI + Chroma with flexible data sources
Streamlined configuration with only essential options
"""

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Streamlined config - OpenAI + Chroma only, flexible data sources"""
    
    # Essential settings only
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    vector_store_path: str = os.getenv("VECTOR_STORE_PATH", "./vectorstore")
    
    # Namespace configuration
    namespace: str = os.getenv("NAMESPACE", "default")
    collection_name: str = os.getenv("COLLECTION_NAME", "enterprise-rag")
    
    # Data source configuration
    data_source: str = os.getenv("DATA_SOURCE", "csv")  # csv, s3, confluence, mixed
    csv_path: str = os.getenv("CSV_PATH", "./data/org_Jira.csv")
    
    # S3 settings
    s3_bucket: str = os.getenv("S3_BUCKET", "")
    s3_key: str = os.getenv("S3_KEY", "")
    
    # Confluence settings
    confluence_url: str = os.getenv("CONFLUENCE_URL", "")
    confluence_username: str = os.getenv("CONFLUENCE_USERNAME", "")
    confluence_api_token: str = os.getenv("CONFLUENCE_API_TOKEN", "")
    confluence_space_key: str = os.getenv("CONFLUENCE_SPACE_KEY", "")
    
    # OpenAI settings
    llm_model: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    
    # Text processing
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Batch processing
    batch_size: int = int(os.getenv("BATCH_SIZE", "100"))  # Documents per batch for vector DB operations
    batch_mode: bool = os.getenv("BATCH_MODE", "true").lower() == "true"  # Enable batch processing for large datasets
    
    def __post_init__(self):
        """Basic validation"""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required")
        
        # Validate data source
        if self.data_source == "csv" and not Path(self.csv_path).exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        if self.data_source == "s3" and (not self.s3_bucket or not self.s3_key):
            raise ValueError("S3_BUCKET and S3_KEY are required for S3 data source")
        
        if self.data_source == "confluence" and not all([
            self.confluence_url, self.confluence_username, 
            self.confluence_api_token, self.confluence_space_key
        ]):
            raise ValueError("Confluence URL, username, API token, and space key are required")
        
        # Create vector store directory
        Path(self.vector_store_path).mkdir(parents=True, exist_ok=True) 