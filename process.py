#!/usr/bin/env python3
"""
Enterprise RAG Data Processing - OpenAI + Chroma with flexible data sources
Process data from CSV, S3, or Confluence into vector embeddings
"""

import argparse
import sys
from pathlib import Path

from langchain_lib.enterprise_rag import EnterpriseRAG
from langchain_lib.config import Config


def main():
    """Enterprise RAG processing with multiple data sources"""
    parser = argparse.ArgumentParser(description="Process data with Enterprise RAG")
    parser.add_argument("--data-source", choices=["csv", "s3", "confluence", "mixed"], 
                       help="Data source type")
    parser.add_argument("--csv-path", help="Path to CSV file")
    parser.add_argument("--s3-bucket", help="S3 bucket name")
    parser.add_argument("--s3-key", help="S3 key/path")
    parser.add_argument("--confluence-space", help="Confluence space key")
    parser.add_argument("--namespace", help="Vector store namespace")
    parser.add_argument("--batch-size", type=int, help="Batch size for processing documents")
    parser.add_argument("--no-batch", action="store_true", help="Disable batch processing")
    parser.add_argument("--force-reload", action="store_true", help="Force reload data")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    try:
        # Load base configuration
        config = Config()
        
        # Override with command line arguments
        if args.data_source:
            config.data_source = args.data_source
        if args.csv_path:
            config.csv_path = args.csv_path
        if args.s3_bucket:
            config.s3_bucket = args.s3_bucket
        if args.s3_key:
            config.s3_key = args.s3_key
        if args.confluence_space:
            config.confluence_space_key = args.confluence_space
        if args.namespace:
            config.namespace = args.namespace
        if args.batch_size:
            config.batch_size = args.batch_size
        if args.no_batch:
            config.batch_mode = False
        
        # Initialize and setup
        print("🚀 Initializing Enterprise RAG System")
        print("=" * 50)
        
        rag = EnterpriseRAG(config)
        
        # Display configuration
        print(f"📊 Configuration:")
        print(f"   Data Source: {config.data_source}")
        print(f"   LLM: {config.llm_model}")
        print(f"   Embeddings: {config.embedding_model}")
        print(f"   Vector Store: Chroma ({config.vector_store_path})")
        print(f"   Namespace: {config.namespace}")
        print(f"   Batch Mode: {'Enabled' if config.batch_mode else 'Disabled'}")
        if config.batch_mode:
            print(f"   Batch Size: {config.batch_size}")
        
        if config.data_source == "csv":
            print(f"   CSV Path: {config.csv_path}")
        elif config.data_source == "s3":
            print(f"   S3: s3://{config.s3_bucket}/{config.s3_key}")
        elif config.data_source == "confluence":
            print(f"   Confluence Space: {config.confluence_space_key}")
        elif config.data_source == "mixed":
            print(f"   Mixed sources: CSV, S3, Confluence (as available)")
        
        # Process data
        print("\n📄 Processing Data...")
        rag.setup_system(force_reload=args.force_reload)
        
        # Show results
        stats = rag.get_stats()
        print("\n📊 Processing Complete!")
        print(f"   Documents processed: {stats['total_documents']}")
        print(f"   Data source: {stats['data_source']}")
        print(f"   Model: {stats['llm_model']}")
        
        print("\n✅ Ready for queries!")
        print("Next steps:")
        print("1. Run: python chat.py")
        print("2. Or use programmatically:")
        print("   from langchain_lib.enterprise_rag import EnterpriseRAG")
        print("   rag = EnterpriseRAG()")
        print("   result = rag.query('your question')")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        
        print("\n💡 Troubleshooting:")
        print("1. Check your .env file configuration")
        print("2. Ensure data source is accessible")
        print("3. Verify API keys are correct")
        print("4. Install dependencies: pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main() 