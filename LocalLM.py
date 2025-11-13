"""
Three-Tier LLM 

Architecture :

Three  Tier :
- first tier compose of caching that will handle response to question already in the database or almost similar
- Second tier that will handle question that are similar but not enough to be handle by the cache :
    Local LM  ( 1B model )
- Third tier that will handle question that are not similar enough to be handle by the cache or the local LM :
    Local LM ( 4B model )

So as Models we will use :
- One Local LM that is gonna be around 4B (gemma, mistral, etc)
- One Local LM that is gonna be around 1B or lower (Gemma 3 )
- One Embedding Model that is gonna be around 1B or lower (Gemma Embedding Model )

Tools used :
- memory Vector database (TBD)
- ollama  for local LLMs
- PDF loader for document ingestion
- TTS
- STT

Functionality : 
- Document Ingestion from PDF
- Question Generation from the document
- storage of the document embedding in the vector database
- storage of the generated questions and answers in the vector database
- storage of the user questions and answers in the vector database
- accuracy evalutation of the user questions compared to those in the database
- Handeling of different scenarios based on the accuracy evaluation :
    - if the accuracy is high enough, return the answer from the database
    - if the accuracy is medium, use the 1B local LM to answer the question
    - if the accuracy is low, use the 4B local LM to answer the question
- TTS of the final answer

Description of the use:

This code is designed to answers the need of a fast and accurate answering from user questions. 
This architecture will be adaptable to any subject and will be able to run on a Raspberry Pi 4 with 8GB of RAM.

Libraries used :
-ollama : to handle local LLMs
-nmupy 
-datetime

"""
from typing import List, Dict, Optional
import ollama
import numpy as np
from pathlib import Path
import os
import sys
from datetime import datetime
from config import *
from modules.pdf_processor import pdf_converter, print_pdf_summary
from modules.database import VectorDatabase
from modules.embeddings import EmbeddingGenerator
###############################################################################################
# function that will handle accuracy evaluation
###############################################################################################
###############################################################################################
# function that will handle the three tier architecture
###############################################################################################
###############################################################################################
# function that will handle the generation of questions from the document using the big model
###############################################################################################
###############################################################################################
# function that will handle the second tier LLM (1B model) that will need to answers based on the similar questions and the similar found in the documents
###############################################################################################


def process_pdf(pdf_path: str, db: VectorDatabase, embedding_gen: EmbeddingGenerator) -> bool:
    """
    Process a PDF and store it in the database
    
    Args:
        pdf_path: Path to PDF file
        db: VectorDatabase instance
        embedding_gen: EmbeddingGenerator instance
        
    Returns:
        bool: Success status
    """
    result = pdf_converter(
        file_path=pdf_path,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        min_chunk_size=MIN_CHUNK_SIZE
    )
    
    if not result["success"]:
        return False
    
    embeddings_result = embedding_gen.generate_embeddings_batch(result["chunks"])
    
    if embeddings_result["failed"] == embeddings_result["total"]:
        return False
    
    valid_data = [
        (chunk, emb, meta) 
        for chunk, emb, meta in zip(result["chunks"], embeddings_result["embeddings"], result["metadata"]) 
        if emb is not None
    ]
    
    if not valid_data:
        return False
    
    valid_chunks, valid_embeddings, valid_metadata = zip(*valid_data)
    
    return db.store_document_chunks(
        chunks=list(valid_chunks),
        embeddings=list(valid_embeddings),
        metadata=list(valid_metadata),
        document_name=os.path.basename(pdf_path)
    )


def query_documents(query: str, db: VectorDatabase, embedding_gen: EmbeddingGenerator, n_results: int = 3) -> Dict:
    """
    Query the database with a question
    
    Args:
        query: Question text
        db: VectorDatabase instance
        embedding_gen: EmbeddingGenerator instance
        n_results: Number of results to return
        
    Returns:
        Dict with search results
    """
    query_embedding = embedding_gen.generate_embedding(query)
    
    if not query_embedding:
        return {}
    
    return db.semantic_search(
        query_embedding=query_embedding,
        collection_name="documents",
        n_results=n_results
    )


def main():
    """Main execution flow"""
    print("\n" + "="*70)
    print("THREE-TIER LLM SYSTEM - PDF Processing & Storage")
    print("="*70 + "\n")
    
    # Initialize embedding generator
    print("Initializing embedding generator...")
    embedding_gen = EmbeddingGenerator(model_name=EMBEDDING_MODEL, batch_size=EMBEDDING_BATCH_SIZE)
    print(f"✓ Using model: {EMBEDDING_MODEL}\n")
    
    # Initialize database
    print("Initializing database...")
    db = VectorDatabase(db_path=VECTOR_DB_PATH)
    
    if not db.initialize_db():
        print("✗ Failed to initialize database")
        return
    
    print("✓ Database initialized\n")
    
    # Find PDFs
    print(f"Searching for PDFs in: {DOCUMENTS_DIR}")
    pdf_files = list(Path(DOCUMENTS_DIR).glob("*.pdf"))
    
    if not pdf_files:
        print("✗ No PDF files found")
        print(f"  Add PDFs to: {DOCUMENTS_DIR}")
        return
    
    print(f"✓ Found {len(pdf_files)} PDF(s)\n")
    
    # Process PDFs
    print("="*70)
    print("PROCESSING PDFs")
    print("="*70 + "\n")
    
    processed = 0
    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")
        if process_pdf(str(pdf_file), db, embedding_gen):
            print(f"✓ Success\n")
            processed += 1
        else:
            print(f"✗ Failed\n")
    
    print(f"Processed {processed}/{len(pdf_files)} PDFs\n")
    
    if processed == 0:
        return
    

    
    # Interactive query loop
    print("="*70)
    print("INTERACTIVE QUERY MODE")
    print("="*70)
    print("Commands: 'quit' to exit, 'stats' for database info\n")
    
    while True:
        try:
            user_input = input("Query: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if user_input.lower() == 'stats':
                stats = db.get_collection_stats()
                for collection, info in stats.items():
                    print(f"  {collection}: {info['count']} items")
                continue
            
            if not user_input:
                continue
            
            results = query_documents(user_input, db, embedding_gen)
            
            if results and 'documents' in results and results['documents']:
                print(f"\nFound {len(results['documents'][0])} results:\n")
                
                for i, doc in enumerate(results['documents'][0]):
                    distance = results['distances'][0][i]
                    metadata = results['metadatas'][0][i]
                    similarity = 1 - distance
                    
                    page = metadata.get('page', '?')
                    preview = doc[:250] + "..." if len(doc) > 250 else doc
                    
                    print(f"[{i+1}] Similarity: {similarity:.3f} | Page: {page}")
                    print(f"    {preview}\n")
            else:
                print("No results found\n")
            
        except KeyboardInterrupt:
            print("\n")
            break
        except Exception as e:
            print(f"Error: {str(e)}\n")
    
    print("="*70)
    print("Session ended")
    print("="*70)


if __name__ == "__main__":
    main()