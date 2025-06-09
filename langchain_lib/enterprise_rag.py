"""
Enterprise RAG System - OpenAI + Chroma with flexible data sources
Streamlined implementation without provider abstraction layers
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import time

# LangChain Core
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Fixed providers - no abstraction needed
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

from .config import Config
from .schema import DocumentMetadata, VectorStoreSchema


class EnterpriseRAG:
    """Streamlined RAG system - OpenAI + Chroma, flexible data sources"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        
        # Initialize fixed components (no provider switching)
        self.llm = ChatOpenAI(
            model=self.config.llm_model,
            temperature=self.config.temperature,
            api_key=self.config.openai_api_key
        )
        
        self.embeddings = OpenAIEmbeddings(
            model=self.config.embedding_model,
            api_key=self.config.openai_api_key
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Will be initialized later
        self.vectorstore = None
        self.qa_chain = None
        self.conversation_chain = None
    
    def load_csv_data(self) -> List[Document]:
        """Load JIRA CSV data"""
        print(f"📊 Loading data from {self.config.csv_path}...")
        df = pd.read_csv(self.config.csv_path)
        print(f"📊 Loaded {len(df)} tickets from CSV")
        
        return self._create_documents_from_dataframe(df, source="csv")
    
    def load_data(self) -> List[Document]:
        """Load data from configured source"""
        if self.config.data_source == "csv":
            return self.load_csv_data()
        elif self.config.data_source == "s3":
            return self.load_s3_data()
        elif self.config.data_source == "confluence":
            return self.load_confluence_data()
        elif self.config.data_source == "mixed":
            return self.load_mixed_data()
        else:
            raise ValueError(f"Unsupported data source: {self.config.data_source}")
    
    def load_s3_data(self) -> List[Document]:
        """Load JIRA data from S3"""
        try:
            import boto3
            from io import StringIO
        except ImportError:
            raise ImportError("S3 support requires: pip install boto3")
        
        print(f"☁️ Loading from S3: s3://{self.config.s3_bucket}/{self.config.s3_key}")
        
        try:
            s3 = boto3.client('s3')
            response = s3.get_object(Bucket=self.config.s3_bucket, Key=self.config.s3_key)
            csv_content = response['Body'].read().decode('utf-8')
            
            # Parse CSV content
            df = pd.read_csv(StringIO(csv_content))
            print(f"📊 Loaded {len(df)} tickets from S3")
            
            # Convert to documents using same logic as CSV
            return self._create_documents_from_dataframe(df, source="s3")
            
        except Exception as e:
            raise ConnectionError(f"Failed to load from S3: {e}")
    
    def load_confluence_data(self) -> List[Document]:
        """Load data from Confluence"""
        try:
            from langchain_community.document_loaders import ConfluenceLoader
        except ImportError:
            raise ImportError("Confluence support requires: pip install atlassian-python-api beautifulsoup4")
        
        print(f"📄 Loading from Confluence space: {self.config.confluence_space_key}")
        
        try:
            loader = ConfluenceLoader(
                url=self.config.confluence_url,
                username=self.config.confluence_username,
                api_key=self.config.confluence_api_token,
                space_key=self.config.confluence_space_key,
                include_attachments=False
            )
            
            documents = loader.load()
            
            if not documents:
                print("⚠️ No documents found in Confluence space")
                return []
            
            print(f"📊 Loaded {len(documents)} documents from Confluence")
            
            # Add confluence-specific metadata using schema
            processed_documents = []
            for i, doc in enumerate(documents):
                doc_metadata = DocumentMetadata.from_confluence_doc(
                    doc, 
                    self.config.confluence_space_key,
                    chunk_index=i,
                    total_chunks=len(documents)
                )
                doc.metadata = doc_metadata.to_dict()
                processed_documents.append(doc)
            
            documents = processed_documents
            
            return documents
            
        except Exception as e:
            error_msg = str(e).lower()
            if "authentication" in error_msg or "401" in error_msg:
                raise ConnectionError("Confluence authentication failed. Check your username and API token.")
            elif "not found" in error_msg or "404" in error_msg:
                raise ConnectionError(f"Confluence space '{self.config.confluence_space_key}' not found.")
            else:
                raise ConnectionError(f"Confluence error: {e}")
    
    def load_mixed_data(self) -> List[Document]:
        """Load data from multiple sources with graceful fallbacks"""
        all_documents = []
        sources_loaded = []
        
        # Try CSV first
        if self.config.csv_path and Path(self.config.csv_path).exists():
            try:
                print("📊 Loading CSV data...")
                csv_docs = self.load_csv_data()
                all_documents.extend(csv_docs)
                sources_loaded.append("CSV")
                print(f"✅ Loaded {len(csv_docs)} CSV documents")
            except Exception as e:
                print(f"⚠️ Failed to load CSV: {e}")
        
        # Try S3
        if self.config.s3_bucket and self.config.s3_key:
            try:
                print("☁️ Loading S3 data...")
                s3_docs = self.load_s3_data()
                all_documents.extend(s3_docs)
                sources_loaded.append("S3")
                print(f"✅ Loaded {len(s3_docs)} S3 documents")
            except Exception as e:
                print(f"⚠️ Failed to load S3: {e}")
        
        # Try Confluence
        if all([self.config.confluence_url, self.config.confluence_username, 
                self.config.confluence_api_token, self.config.confluence_space_key]):
            try:
                print("📄 Loading Confluence data...")
                confluence_docs = self.load_confluence_data()
                all_documents.extend(confluence_docs)
                sources_loaded.append("Confluence")
                print(f"✅ Loaded {len(confluence_docs)} Confluence documents")
            except Exception as e:
                print(f"⚠️ Failed to load Confluence: {e}")
        
        if not all_documents:
            raise ValueError("No data sources could be loaded successfully")
        
        print(f"📊 Total: {len(all_documents)} documents from {', '.join(sources_loaded)}")
        return all_documents
    
    def _create_documents_from_dataframe(self, df: pd.DataFrame, source: str = "csv") -> List[Document]:
        """Helper method to create documents from DataFrame with schema"""
        documents = []
        for _, row in df.iterrows():
            # Enhanced content creation with more fields
            content = f"""
            Ticket: {row.get('Issue key', 'Unknown')}
            Summary: {row.get('Summary', 'No summary')}
            Description: {row.get('Description', 'No description')}
            Status: {row.get('Status', 'Unknown')}
            Priority: {row.get('Priority', 'Unknown')}
            Issue Type: {row.get('Issue Type', 'Unknown')}
            Assignee: {row.get('Assignee', 'Unassigned')}
            Reporter: {row.get('Reporter', 'Unknown')}
            Component: {row.get('Component/s', 'N/A')}
            Version: {row.get('Affects Version/s', 'N/A')}
            Fix Version: {row.get('Fix Version/s', 'N/A')}
            Labels: {row.get('Labels', 'N/A')}
            Created: {row.get('Created', 'N/A')}
            Updated: {row.get('Updated', 'N/A')}
            """
            
            # Use schema for structured metadata
            doc_metadata = DocumentMetadata.from_jira_row(row.to_dict(), source)
            
            documents.append(Document(
                page_content=content.strip(),
                metadata=doc_metadata.to_dict()
            ))
        
        return documents

    def _batch_add_documents(self, documents: List[Document], collection_name: str) -> None:
        """Add documents to vectorstore in batches for better performance"""
        start_time = time.time()
        
        if not self.config.batch_mode or len(documents) <= self.config.batch_size:
            # Process all at once if batch mode disabled or small dataset
            split_docs = self.text_splitter.split_documents(documents)
            print(f"📄 Processing {len(split_docs)} chunks in single batch")
            
            self.vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=self.embeddings,
                collection_name=collection_name,
                persist_directory=self.config.vector_store_path
            )
            elapsed = time.time() - start_time
            print(f"⏱️  Completed in {elapsed:.1f}s ({len(split_docs)/elapsed:.1f} chunks/sec)")
            return
        
        # Batch processing for large datasets
        print(f"📦 Batch processing enabled: {self.config.batch_size} documents per batch")
        print(f"📊 Processing {len(documents)} documents...")
        
        # Process documents in batches
        total_batches = (len(documents) + self.config.batch_size - 1) // self.config.batch_size
        all_chunks = []
        
        for i in range(0, len(documents), self.config.batch_size):
            batch_num = (i // self.config.batch_size) + 1
            batch = documents[i:i + self.config.batch_size]
            
            print(f"📄 Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
            
            # Split documents in this batch
            batch_chunks = self.text_splitter.split_documents(batch)
            all_chunks.extend(batch_chunks)
            
            print(f"   → Generated {len(batch_chunks)} chunks from batch {batch_num}")
        
        print(f"📄 Total chunks to embed: {len(all_chunks)}")
        
        # Now batch the embedding process
        embed_start = time.time()
        
        if len(all_chunks) <= self.config.batch_size:
            # Embed all chunks at once if within batch size
            print("🔄 Creating vectorstore with all chunks...")
            self.vectorstore = Chroma.from_documents(
                documents=all_chunks,
                embedding=self.embeddings,
                collection_name=collection_name,
                persist_directory=self.config.vector_store_path
            )
        else:
            # Embed in batches
            print(f"🔄 Embedding {len(all_chunks)} chunks in batches of {self.config.batch_size}...")
            
            # Create initial vectorstore with first batch
            first_batch = all_chunks[:self.config.batch_size]
            self.vectorstore = Chroma.from_documents(
                documents=first_batch,
                embedding=self.embeddings,
                collection_name=collection_name,
                persist_directory=self.config.vector_store_path
            )
            print(f"✅ Created vectorstore with first batch ({len(first_batch)} chunks)")
            
            # Add remaining batches
            remaining_chunks = all_chunks[self.config.batch_size:]
            total_embed_batches = (len(remaining_chunks) + self.config.batch_size - 1) // self.config.batch_size
            
            for i in range(0, len(remaining_chunks), self.config.batch_size):
                batch_num = (i // self.config.batch_size) + 1
                batch = remaining_chunks[i:i + self.config.batch_size]
                
                print(f"🔄 Adding embedding batch {batch_num}/{total_embed_batches} ({len(batch)} chunks)")
                self.vectorstore.add_documents(batch)
                
                # Progress update
                completed_chunks = self.config.batch_size + (i + len(batch))
                progress = (completed_chunks / len(all_chunks)) * 100
                elapsed = time.time() - embed_start
                rate = completed_chunks / elapsed if elapsed > 0 else 0
                print(f"   📊 Progress: {progress:.1f}% ({rate:.1f} chunks/sec)")
                
        total_elapsed = time.time() - start_time
        final_rate = len(all_chunks) / total_elapsed if total_elapsed > 0 else 0
        print(f"✅ Batch processing complete: {len(all_chunks)} chunks embedded")
        print(f"⏱️  Total time: {total_elapsed:.1f}s ({final_rate:.1f} chunks/sec)")

    def setup_vectorstore(self, force_reload: bool = False):
        """Initialize Chroma vectorstore with namespace support and batch processing"""
        # Create namespaced collection name
        collection_name = f"{self.config.collection_name}_{self.config.namespace}"
        
        # Check if vectorstore exists
        if not force_reload and Path(self.config.vector_store_path).exists():
            self.vectorstore = Chroma(
                collection_name=collection_name,
                persist_directory=self.config.vector_store_path,
                embedding_function=self.embeddings
            )
            
            if self.vectorstore._collection.count() > 0:
                print(f"✅ Using existing vectorstore '{collection_name}' with {self.vectorstore._collection.count()} documents")
                return
        
        # Load documents from configured source
        documents = self.load_data()
        
        # Use batch processing for efficient loading
        print(f"🔄 Creating Chroma vectorstore with namespace '{self.config.namespace}'...")
        if self.config.batch_mode:
            print(f"📦 Batch mode enabled (batch size: {self.config.batch_size})")
        
        self._batch_add_documents(documents, collection_name)
        print(f"✅ Vectorstore '{collection_name}' created and persisted")
    
    def setup_chains(self):
        """Setup QA and conversation chains"""
        if not self.vectorstore:
            raise RuntimeError("Setup vectorstore first")
        
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Simple QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        # Conversation chain with memory
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True
        )
        
        print("✅ Chains setup complete")
    
    def setup_system(self, force_reload: bool = False):
        """One-step setup"""
        print("🚀 Setting up Enterprise RAG system...")
        self.setup_vectorstore(force_reload)
        self.setup_chains()
        print("✅ Setup complete!")
        return self.vectorstore._collection.count() if self.vectorstore else 0
    
    def query(self, question: str, use_conversation: bool = True) -> Dict[str, Any]:
        """Query the system"""
        if not self.qa_chain or not self.conversation_chain:
            raise RuntimeError("Run setup_system() first")
        
        if use_conversation:
            result = self.conversation_chain.invoke({"question": question})
            return {
                "answer": result["answer"],
                "source_documents": result.get("source_documents", []),
                "conversation_mode": True
            }
        else:
            result = self.qa_chain.invoke({"query": question})
            return {
                "answer": result["result"],
                "source_documents": result.get("source_documents", []),
                "conversation_mode": False
            }
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
        print("🧹 Memory cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system stats"""
        return {
            "llm_model": self.config.llm_model,
            "embedding_model": self.config.embedding_model,
            "data_source": self.config.data_source,
            "namespace": self.config.namespace,
            "collection_name": f"{self.config.collection_name}_{self.config.namespace}",
            "total_documents": self.vectorstore._collection.count() if self.vectorstore else 0
        }
    
    def list_namespaces(self) -> List[str]:
        """List all available namespaces in the vector store"""
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self.config.vector_store_path)
            collections = client.list_collections()
            
            # Extract namespaces from collection names
            namespaces = set()
            prefix = f"{self.config.collection_name}_"
            
            for collection in collections:
                if collection.name.startswith(prefix):
                    namespace = collection.name[len(prefix):]
                    namespaces.add(namespace)
            
            return sorted(list(namespaces))
        except:
            return []
    
    def switch_namespace(self, namespace: str) -> bool:
        """Switch to a different namespace"""
        try:
            old_namespace = self.config.namespace
            self.config.namespace = namespace
            
            # Reinitialize vectorstore with new namespace
            collection_name = f"{self.config.collection_name}_{namespace}"
            self.vectorstore = Chroma(
                collection_name=collection_name,
                persist_directory=self.config.vector_store_path,
                embedding_function=self.embeddings
            )
            
            print(f"🔄 Switched from namespace '{old_namespace}' to '{namespace}'")
            return True
        except Exception as e:
            print(f"❌ Failed to switch namespace: {e}")
            return False 