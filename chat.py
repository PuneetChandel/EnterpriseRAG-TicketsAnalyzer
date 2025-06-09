#!/usr/bin/env python3
"""
Enterprise RAG Chat Interface - OpenAI + Chroma with flexible data sources
Streamlined chat interface focused on core functionality
"""

from langchain_lib.enterprise_rag import EnterpriseRAG
from langchain_lib.config import Config


def main():
    """Enterprise RAG chat interface"""
    try:
        print("🚀 Initializing Enterprise RAG System...")
        
        # Simple initialization
        config = Config()
        rag = EnterpriseRAG(config)
        rag.setup_system()
        
        # Show stats
        stats = rag.get_stats()
        print(f"\n📊 Ready! Loaded {stats['total_documents']} documents")
        print(f"🤖 Using {stats['llm_model']} with {stats['embedding_model']}")
        print(f"📊 Data source: {stats['data_source']}")
        print("\n💬 Ask questions about your JIRA tickets")
        print("Commands: 'clear', 'stats', 'namespaces', 'switch <namespace>', 'quit'\n")
        
        # Chat loop
        while True:
            question = input("Q: ").strip()
            
            if not question:
                continue
            elif question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif question.lower() == 'clear':
                rag.clear_memory()
                continue
            elif question.lower() == 'stats':
                stats = rag.get_stats()
                print(f"📊 Stats: {stats}")
                continue
            elif question.lower() == 'namespaces':
                namespaces = rag.list_namespaces()
                if namespaces:
                    print(f"📂 Available namespaces: {', '.join(namespaces)}")
                    print(f"📍 Current namespace: {rag.config.namespace}")
                else:
                    print("📂 No namespaces found")
                continue
            elif question.lower().startswith('switch '):
                namespace = question[7:].strip()
                if rag.switch_namespace(namespace):
                    # Reinitialize chains with new namespace
                    rag.setup_chains()
                continue
            
            # Query the system
            try:
                result = rag.query(question)
                print(f"\nA: {result['answer']}")
                
                if result.get('source_documents'):
                    sources = [doc.metadata.get('ticket_id', 'Unknown') 
                             for doc in result['source_documents'][:3]]
                    print(f"📄 Sources: {', '.join(sources)}")
                
                mode = "conversation" if result.get('conversation_mode') else "simple"
                print(f"🔧 Mode: {mode}\n")
                
            except Exception as e:
                print(f"❌ Error: {e}\n")
    
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        print("\nCheck:")
        print("1. OPENAI_API_KEY in .env")
        print("2. Data source is accessible")
        print("3. Dependencies installed: pip install -r requirements.txt")


if __name__ == "__main__":
    main() 