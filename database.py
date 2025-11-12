"""
Database Module - ChromaDB Operations
Handles all vector database operations for the three-tier LLM system
Optimized for Raspberry Pi 4 (8GB RAM)
"""
from config import *
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import os
import json
import shutil
from pathlib import Path



"""
Creation of a vector database class that will handle all the operations with ChromaDB

function in the class :
- initialize_db
only used at the initialization of the database to create the client and the collections

- create_collections
Crearte the necessary collections for the system

- store_document_chunks
store the document chunks with their embeddings in the database and their metadata

- get_document_chunks
retrieve document chunks based on similarity search or optional filters

- delete_document_by_name
just delete a document using his name in the database

- store_question_answer
store a question answer pair in the question cache collection (gonna be used for the generation of the cache at the begining by the big model)

- find_similar_questions
find similar questions in the question cache collection but have to define the accuracy treshold for the 3 question max retreived


################ functionnality that are not gonna be used in the beggining but can be useful later if scaling###############

If multiple users ar eusing the RSC locally for the same course, at the end of the courses according
to respect of privacy and anonymity we can gather the most used questions to give a feedback to the teacher.

- update_question_usage
increment the usage count of a question in the question cache collection in order to track the most used questions

- get_most_used_questions
get the most used questions in the question cache collection

- store_user_interaction
store the user interactions in the user history collection, useful for later analysis of the user behavior and personal adaptation of the learning style

################################################################################################

- semantic_search
general semantic search function that can be used on any collection

- hybrid_search
hybrid search function that combine semantic search and keyword matching

- clear_collection
clear all the data in a collection





"""
class VectorDatabase:
    """
    ChromaDB wrapper for three-tier LLM system
    Manages documents, question cache, and user history
    """
    
    def __init__(self, db_path=DB_PATH):
        """
        Initialize ChromaDB client and collections
        
        Args:
            db_path: Path to store the database
        """
        self.db_path = db_path
        self.client = None
        self.collections = {}
        
        # Collection names
        self.DOCUMENTS_COLLECTION = "documents"
        self.QUESTIONS_CACHE_COLLECTION = "questions_cache"
        self.USER_HISTORY_COLLECTION = "user_history"
        
    def initialize_db(self) -> bool:
        """
        Initialize ChromaDB client and create collections
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.db_path, exist_ok=True)
            
            # Initialize persistent client
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            print(f"✅ ChromaDB client initialized at: {self.db_path}")
            
            # Create all collections
            self.create_collections()
            
            return True
            
        except Exception as e:
            print(f"❌ Error initializing database: {str(e)}")
            return False
    
    def create_collections(self) -> None:
        """
        Create all necessary collections for the system
        """
        try:
            # Documents collection - stores PDF chunks
            self.collections[self.DOCUMENTS_COLLECTION] = self.client.get_or_create_collection(
                name=self.DOCUMENTS_COLLECTION,
                metadata={
                    "description": "Document chunks from PDFs with embeddings",
                    "hnsw:space": "cosine"  # Use cosine similarity can use other but not that much difference
                }
            )
            
            # Questions cache collection - Tier 1 cache
            self.collections[self.QUESTIONS_CACHE_COLLECTION] = self.client.get_or_create_collection(
                name=self.QUESTIONS_CACHE_COLLECTION,
                metadata={
                    "description": "Cached questions and answers with high accuracy",
                    "hnsw:space": "cosine"
                }
            )
            
            # User history collection - track user interactions
            self.collections[self.USER_HISTORY_COLLECTION] = self.client.get_or_create_collection(
                name=self.USER_HISTORY_COLLECTION,
                metadata={
                    "description": "User interaction history for learning",
                    "hnsw:space": "cosine"
                }
            )
            
            print(f"✅ Created {len(self.collections)} collections:")
            for name in self.collections.keys():
                print(f"   📁 {name}")
                
        except Exception as e:
            print(f"❌ Error creating collections: {str(e)}")
            raise
    
    ###############################################################################################
    # DOCUMENT STORAGE FUNCTIONS
    ###############################################################################################
    
    def store_document_chunks(self,chunks: List[str], embeddings: List[List[float]],metadata: List[Dict],document_name: str = None) -> bool:
        """
        Store document chunks with embeddings in the database
        
        Args:
            chunks: List of text chunks
            embeddings: List of embedding vectors
            metadata: List of metadata dictionaries
            document_name: Optional document identifier
            
        Returns:
            bool: True if successful
        """
        try:
            collection = self.collections[self.DOCUMENTS_COLLECTION]
            
            # Generate unique IDs
            timestamp = datetime.now().isoformat()
            ids = [f"doc_{i}_{timestamp}" for i in range(len(chunks))]
            
            # Add document name to metadata if provided
            if document_name:
                for meta in metadata:
                    meta["document_name"] = document_name
            
            # Add timestamp to all metadata
            for meta in metadata:
                meta["stored_at"] = timestamp
            
            # Store in ChromaDB
            collection.add(
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadata,
                ids=ids
            )
            
            print(f"✅ Stored {len(chunks)} document chunks")
            return True
            
        except Exception as e:
            print(f"❌ Error storing document chunks: {str(e)}")
            return False
    
    def get_document_chunks(self,query_embedding: Optional[List[float]] = None,n_results: int = 5,where: Optional[Dict] = None,where_document: Optional[Dict] = None) -> Dict:
        """
        Retrieve document chunks with optional filtering
        
        Args:
            query_embedding: Embedding vector for similarity search
            n_results: Number of results to return
            where: Metadata filter (e.g., {"page": 5})
            where_document: Document content filter
            
        Returns:
            Dict with documents, metadatas, distances, and ids
        """
        try:
            collection = self.collections[self.DOCUMENTS_COLLECTION]
            
            if query_embedding:
                # Similarity search
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=where,
                    where_document=where_document
                )
            else:
                # Just retrieve with filters (no similarity)
                results = collection.get(
                    where=where,
                    where_document=where_document,
                    limit=n_results
                )
            
            return results
            
        except Exception as e:
            print(f"❌ Error retrieving document chunks: {str(e)}")
            return {}
    
    def delete_document_by_name(self, document_name: str) -> bool:
        """
        Delete all chunks from a specific document
        
        Args:
            document_name: Name of the document to delete
            
        Returns:
            bool: True if successful
        """
        try:
            collection = self.collections[self.DOCUMENTS_COLLECTION]
            
            # Get all IDs for this document
            results = collection.get(
                where={"document_name": document_name}
            )
            
            if results and results['ids']:
                collection.delete(ids=results['ids'])
                print(f"✅ Deleted {len(results['ids'])} chunks from {document_name}")
                return True
            else:
                print(f"⚠️  No chunks found for document: {document_name}")
                return False
                
        except Exception as e:
            print(f"❌ Error deleting document: {str(e)}")
            return False
    
    ###############################################################################################
    # QUESTION CACHE FUNCTIONS (TIER 1)
    ###############################################################################################
    
    def store_question_answer(self,question: str,answer: str,question_embedding: List[float],accuracy: float = 1.0,model_used: str = "cache",metadata: Optional[Dict] = None) -> bool:
        """
        Store a question-answer pair in the cache
        
        Args:
            question: Question text
            answer: Answer text
            question_embedding: Embedding vector of the question
            accuracy: Accuracy/confidence score (0-1)
            model_used: Model that generated the answer
            metadata: Additional metadata
            
        Returns:
            bool: True if successful
        """
        try:
            collection = self.collections[self.QUESTIONS_CACHE_COLLECTION]
            
            # Create metadata
            meta = {
                "answer": answer,
                "accuracy": accuracy,
                "model_used": model_used,
                "timestamp": datetime.now().isoformat(),
                "usage_count": 1
            }
            
            # Add additional metadata if provided
            if metadata:
                meta.update(metadata)
            
            # Generate unique ID
            question_id = f"q_{abs(hash(question))}_{datetime.now().timestamp()}"
            
            # Store in ChromaDB
            collection.add(
                documents=[question],
                embeddings=[question_embedding],
                metadatas=[meta],
                ids=[question_id]
            )
            
            print(f"✅ Cached question-answer pair (ID: {question_id})")
            return True
            
        except Exception as e:
            print(f"❌ Error storing question-answer: {str(e)}")
            return False
    
    def find_similar_questions(self,question_embedding: List[float],n_results: int = 3,min_accuracy: float = 0.0) -> Dict:
        """
        Find similar cached questions
        
        Args:
            question_embedding: Embedding of the query question
            n_results: Number of similar questions to return
            min_accuracy: Minimum accuracy threshold
            
        Returns:
            Dict with similar questions, answers, and similarity scores
        """
        try:
            collection = self.collections[self.QUESTIONS_CACHE_COLLECTION]
            
            # Search for similar questions
            results = collection.query(
                query_embeddings=[question_embedding],
                n_results=n_results,
                where={"accuracy": {"$gte": min_accuracy}} if min_accuracy > 0 else None
            )
            
            return results
            
        except Exception as e:
            print(f"❌ Error finding similar questions: {str(e)}")
            return {}
    
    def update_question_usage(self, question_id: str) -> bool:
        """
        Increment usage count for a cached question
        
        Args:
            question_id: ID of the question to update
            
        Returns:
            bool: True if successful
        """
        try:
            collection = self.collections[self.QUESTIONS_CACHE_COLLECTION]
            
            # Get current metadata
            result = collection.get(ids=[question_id])
            
            if result and result['metadatas']:
                current_meta = result['metadatas'][0]
                current_meta['usage_count'] = current_meta.get('usage_count', 0) + 1
                current_meta['last_used'] = datetime.now().isoformat()
                
                # Update metadata
                collection.update(
                    ids=[question_id],
                    metadatas=[current_meta]
                )
                
                print(f"✅ Updated usage count for question {question_id}")
                return True
            else:
                print(f"⚠️  Question not found: {question_id}")
                return False
                
        except Exception as e:
            print(f"❌ Error updating question usage: {str(e)}")
            return False
    
    def get_most_used_questions(self, n_results: int = 10) -> Dict:
        """
        Get the most frequently used cached questions
        
        Args:
            n_results: Number of questions to return
            
        Returns:
            Dict with questions sorted by usage count
        """
        try:
            collection = self.collections[self.QUESTIONS_CACHE_COLLECTION]
            
            # Get all questions
            results = collection.get()
            
            if not results or not results['metadatas']:
                return {}
            
            # Sort by usage count
            questions_with_usage = []
            for i, meta in enumerate(results['metadatas']):
                questions_with_usage.append({
                    'id': results['ids'][i],
                    'question': results['documents'][i],
                    'answer': meta.get('answer', ''),
                    'usage_count': meta.get('usage_count', 0),
                    'accuracy': meta.get('accuracy', 0)
                })
            
            # Sort by usage count
            sorted_questions = sorted(
                questions_with_usage,
                key=lambda x: x['usage_count'],
                reverse=True
            )[:n_results]
            
            return {'questions': sorted_questions}
            
        except Exception as e:
            print(f"❌ Error getting most used questions: {str(e)}")
            return {}
    
    ###############################################################################################
    # USER HISTORY FUNCTIONS
    ###############################################################################################
    
    def store_user_interaction(self,question: str,answer: str,question_embedding: List[float],model_used: str,similarity_score: float = 0.0,response_time: float = 0.0,metadata: Optional[Dict] = None) -> bool:
        """
        Store user interaction in history
        
        Args:
            question: User question
            answer: System answer
            question_embedding: Question embedding
            model_used: Model that answered (cache/1B/4B)
            similarity_score: Similarity to cached questions
            response_time: Time taken to generate answer
            metadata: Additional metadata
            
        Returns:
            bool: True if successful
        """
        try:
            collection = self.collections[self.USER_HISTORY_COLLECTION]
            
            # Create metadata
            meta = {
                "answer": answer,
                "model_used": model_used,
                "similarity_score": similarity_score,
                "response_time": response_time,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add additional metadata if provided
            if metadata:
                meta.update(metadata)
            
            # Generate unique ID
            interaction_id = f"user_{datetime.now().timestamp()}"
            
            # Store in ChromaDB
            collection.add(
                documents=[question],
                embeddings=[question_embedding],
                metadatas=[meta],
                ids=[interaction_id]
            )
            
            return True
            
        except Exception as e:
            print(f"❌ Error storing user interaction: {str(e)}")
            return False
    
    
    ###############################################################################################
    # SIMILARITY SEARCH FUNCTIONS
    ###############################################################################################
    
    def semantic_search(self,query_embedding: List[float],collection_name: str,n_results: int = 5,where: Optional[Dict] = None) -> Dict:
        """
        General semantic search across any collection
        
        Args:
            query_embedding: Query embedding vector
            collection_name: Name of collection to search
            n_results: Number of results
            where: Metadata filters
            
        Returns:
            Dict with search results
        """
        try:
            if collection_name not in self.collections:
                print(f"⚠️  Collection not found: {collection_name}")
                return {}
            
            collection = self.collections[collection_name]
            
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where
            )
            
            return results
            
        except Exception as e:
            print(f"❌ Error in semantic search: {str(e)}")
            return {}
    
    def hybrid_search(self,query_embedding: List[float],keywords: List[str],collection_name: str,n_results: int = 5) -> Dict:
        """
        Hybrid search combining semantic similarity and keyword matching
        
        Args:
            query_embedding: Query embedding vector
            keywords: Keywords to search in documents
            collection_name: Collection to search
            n_results: Number of results
            
        Returns:
            Dict with combined search results
        """
        try:
            if collection_name not in self.collections:
                print(f"⚠️  Collection not found: {collection_name}")
                return {}
            
            collection = self.collections[collection_name]
            
            # Semantic search
            semantic_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results * 2  # Get more for filtering
            )
            
            # Filter by keywords
            if keywords and semantic_results['documents']:
                filtered_results = {
                    'documents': [],
                    'metadatas': [],
                    'distances': [],
                    'ids': []
                }
                
                for i, doc in enumerate(semantic_results['documents'][0]):
                    # Check if any keyword is in document
                    doc_lower = doc.lower()
                    if any(keyword.lower() in doc_lower for keyword in keywords):
                        filtered_results['documents'].append(doc)
                        filtered_results['metadatas'].append(semantic_results['metadatas'][0][i])
                        filtered_results['distances'].append(semantic_results['distances'][0][i])
                        filtered_results['ids'].append(semantic_results['ids'][0][i])
                        
                        if len(filtered_results['documents']) >= n_results:
                            break
                
                return filtered_results
            
            return semantic_results
            
        except Exception as e:
            print(f"❌ Error in hybrid search: {str(e)}")
            return {}
    
    ###############################################################################################
    # UTILITY FUNCTIONS
    ###############################################################################################
    

    
    def clear_collection(self, collection_name: str) -> bool:
        """
        Clear all data from a collection
        
        Args:
            collection_name: Name of collection to clear
            
        Returns:
            bool: True if successful
        """
        try:
            if collection_name not in self.collections:
                print(f"⚠️  Collection not found: {collection_name}")
                return False
            
            # Delete the collection
            self.client.delete_collection(name=collection_name)
            
            # Recreate it
            self.collections[collection_name] = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            print(f"✅ Cleared collection: {collection_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error clearing collection: {str(e)}")
            return False
    



###############################################################################################
# USAGE EXAMPLE
###############################################################################################

if __name__ == "__main__":
    # Initialize database
    db = VectorDatabase(db_path="./data/vector_db")
    
    if db.initialize_db():
        print("\n✅ Database initialized successfully!")
        
        # Example: Store document chunks
        sample_chunks = [
            "This is the first chunk of text from a document.",
            "This is the second chunk with different content.",
            "This is the third chunk discussing another topic."
        ]
        
        # Mock embeddings (in real use, generate with embedding model)
        sample_embeddings = [
            [0.1, 0.2, 0.3, 0.4, 0.5] * 10,  # 50-dim vector
            [0.2, 0.3, 0.4, 0.5, 0.6] * 10,
            [0.3, 0.4, 0.5, 0.6, 0.7] * 10
        ]
        
        sample_metadata = [
            {"page": 1, "chunk_index": 0, "source_file": "test.pdf"},
            {"page": 1, "chunk_index": 1, "source_file": "test.pdf"},
            {"page": 2, "chunk_index": 2, "source_file": "test.pdf"}
        ]
        
        # Store chunks
        db.store_document_chunks(
            chunks=sample_chunks,
            embeddings=sample_embeddings,
            metadata=sample_metadata,
            document_name="test_document"
        )
        
        # Example: Store a question-answer pair
        db.store_question_answer(
            question="What is the main topic?",
            answer="The main topic is about document processing.",
            question_embedding=[0.15, 0.25, 0.35, 0.45, 0.55] * 10,
            accuracy=0.95,
            model_used="4B"
        )
        

        
        print("\n✅ Database operations completed!")
    else:
        print("\n❌ Failed to initialize database")