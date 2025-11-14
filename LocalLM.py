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
from modules.accuracy_evaluator import AccuracyEvaluator
from modules.llm_handler import LLMHandler
from modules.question_generator import QuestionGenerator


def process_pdf(pdf_path: str, db: VectorDatabase, embedding_gen: EmbeddingGenerator) -> bool:
    """Process a PDF and store it in the database"""
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
    
    try:
        return db.store_document_chunks(
            chunks=list(valid_chunks),
            embeddings=list(valid_embeddings),
            metadata=list(valid_metadata),
            document_name=os.path.basename(pdf_path)
        )
    except Exception as e:
        if "expecting embedding with dimension" in str(e):
            print(f"  [!] Dimension mismatch - clearing documents collection...")
            db.clear_collection("documents")
            print(f"  Retrying...")
            
            return db.store_document_chunks(
                chunks=list(valid_chunks),
                embeddings=list(valid_embeddings),
                metadata=list(valid_metadata),
                document_name=os.path.basename(pdf_path)
            )
        else:
            print(f"  Error: {str(e)}")
            return False


def generate_questions_for_cache(
    db: VectorDatabase,
    embedding_gen: EmbeddingGenerator,
    question_gen: QuestionGenerator
) -> bool:
    """
    Generate questions from documents and populate cache
    
    Args:
        db: VectorDatabase instance
        embedding_gen: EmbeddingGenerator instance
        question_gen: QuestionGenerator instance
        
    Returns:
        bool: Success status
    """
    print("\n" + "="*70)
    print("GENERATING QUESTIONS FOR CACHE")
    print("="*70 + "\n")
    
    # Check if model is available
    if not question_gen.verify_model():
        print(f"[X] Model not available: {LARGE_MODEL}")
        print(f"  Install with: ollama pull {LARGE_MODEL}")
        return False
    
    print(f"Using model: {LARGE_MODEL}")
    print(f"Max chunks: {MAX_CHUNKS_FOR_QUESTIONS if MAX_CHUNKS_FOR_QUESTIONS else 'all'}")
    print(f"Questions per chunk: {QUESTIONS_PER_CHUNK}")
    
    # Check if there are documents
    doc_count = db.get_collection_stats().get('documents', {}).get('count', 0)
    if doc_count == 0:
        print("\n[!] No documents in database - skipping question generation")
        return False
    
    print(f"\nFound {doc_count} document chunks")
    
    # Check if cache already has questions
    cache_count = db.get_collection_stats().get('questions_cache', {}).get('count', 0)
    if cache_count > 0:
        print(f"\n[!] Cache already has {cache_count} questions")
        user_input = input("Generate more questions? (y/n): ").strip().lower()
        if user_input != 'y':
            print("Skipping question generation")
            return False
    
    # Generate questions
    result = question_gen.generate_and_cache_questions(
        max_chunks=MAX_CHUNKS_FOR_QUESTIONS,
        questions_per_chunk=QUESTIONS_PER_CHUNK
    )
    
    if result['success']:
        print(f"\n{'='*70}")
        print("QUESTION GENERATION COMPLETE")
        print(f"{'='*70}")
        print(f"Chunks processed: {result['total_chunks']}")
        print(f"Questions generated: {result['questions_generated']}")
        print(f"Questions cached: {result['questions_cached']}")
        print(f"{'='*70}\n")
        return True
    else:
        print(f"\n[X] Question generation failed: {result.get('error', 'Unknown error')}\n")
        return False


def main():
    """Main execution flow"""
    print("\n" + "="*70)
    print("THREE-TIER LLM SYSTEM")
    print("="*70 + "\n")
    
    # Initialize components
    print("Initializing components...")
    
    # Embedding generator
    embedding_gen = EmbeddingGenerator(model_name=EMBEDDING_MODEL, batch_size=EMBEDDING_BATCH_SIZE)
    test_emb = embedding_gen.generate_embedding("test")
    if not test_emb:
        print("[X] Failed to generate test embedding")
        print("  Make sure Ollama is running: ollama serve")
        print(f"  Make sure model is installed: ollama pull {EMBEDDING_MODEL}")
        return
    print(f"[OK] Embedding model: {EMBEDDING_MODEL} ({len(test_emb)}D)")
    
    # Database
    db = VectorDatabase(db_path=VECTOR_DB_PATH)
    if not db.initialize_db():
        print("[X] Failed to initialize database")
        return
    print("[OK] Database initialized")
    
    # Evaluator
    evaluator = AccuracyEvaluator(db, embedding_gen)
    print(f"[OK] Accuracy evaluator initialized")
    
    # LLM Handler
    llm_handler = LLMHandler(db, embedding_gen, evaluator)
    print("[OK] LLM handler initialized")
    
    # Question Generator
    question_gen = QuestionGenerator(db, embedding_gen)
    print("[OK] Question generator initialized")
    
    # Verify models
    print("\nVerifying LLM models...")
    model_status = llm_handler.verify_models()
    
    if not model_status.get("all_available", False):
        print("[!] Some models are not available:")
        if not model_status["small_model"]["available"]:
            print(f"  [X] Small model: {SMALL_MODEL}")
            print(f"    Install with: ollama pull {SMALL_MODEL}")
        else:
            print(f"  [OK] Small model: {SMALL_MODEL}")

        if not model_status["large_model"]["available"]:
            print(f"  [X] Large model: {LARGE_MODEL}")
            print(f"    Install with: ollama pull {LARGE_MODEL}")
        else:
            print(f"  [OK] Large model: {LARGE_MODEL}")
        
        print("\nYou can continue, but some features may not work.")
        user_choice = input("Continue anyway? (y/n): ").strip().lower()
        if user_choice != 'y':
            return
    else:
        print(f"[OK] Small model: {SMALL_MODEL}")
        print(f"[OK] Large model: {LARGE_MODEL}")
    
    print(f"\nTier thresholds:")
    print(f"  Tier 1 (cache): >= {HIGH_SIMILARITY_THRESHOLD}")
    print(f"  Tier 2 (small): >= {MEDIUM_SIMILARITY_THRESHOLD}")
    print(f"  Tier 3 (large): < {MEDIUM_SIMILARITY_THRESHOLD}\n")
    
    # Find and process PDFs
    print(f"Searching for PDFs in: {DOCUMENTS_DIR}")
    pdf_files = list(Path(DOCUMENTS_DIR).glob("*.pdf"))
    
    if not pdf_files:
        print("[!] No PDF files found")
        print(f"  Add PDFs to: {DOCUMENTS_DIR}")
    else:
        print(f"[OK] Found {len(pdf_files)} PDF(s)\n")
        
        print("="*70)
        print("PROCESSING PDFs")
        print("="*70 + "\n")
        
        processed = 0
        for pdf_file in pdf_files:
            print(f"Processing: {pdf_file.name}")
            if process_pdf(str(pdf_file), db, embedding_gen):
                print(f"[OK] Success\n")
                processed += 1
            else:
                print(f"[X] Failed\n")
        
        print(f"Processed {processed}/{len(pdf_files)} PDFs\n")

    
    # Generate questions if enabled
    if AUTO_GENERATE_QUESTIONS:
        generate_questions_for_cache(db, embedding_gen, question_gen)
    

    
    # Interactive query loop
    print("="*70)
    print("INTERACTIVE QUERY MODE - Three-Tier System")
    print("="*70)
    print("Commands:")
    print("  'quit' - Exit")
    print("  'stats' - Database statistics")
    print("  'tier-stats' - Tier usage statistics")
    print("  'generate' - Generate more questions")
    print("  'help' - Show this help message")
    print()
    
    while True:
        try:
            user_input = input("Query: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if user_input.lower() == 'help':
                print("\nCommands:")
                print("  'quit' - Exit the program")
                print("  'stats' - Show database statistics")
                print("  'tier-stats' - Show tier usage statistics")
                print("  'generate' - Generate more questions for cache")
                print("  'help' - Show this help message")
                print()
                continue
            
            if user_input.lower() == 'stats':
                stats = db.get_collection_stats()
                print()
                for collection, info in stats.items():
                    print(f"  {collection}: {info['count']} items")
                print()
                continue
            
            if user_input.lower() == 'tier-stats':
                tier_stats = evaluator.get_tier_statistics()
                print(f"\nTier Usage Statistics:")
                print(f"  Total queries: {tier_stats['total_queries']}")
                print(f"  Tier 1 (cache): {tier_stats['tier_1_cache']}")
                print(f"  Tier 2 (small): {tier_stats['tier_2_small']}")
                print(f"  Tier 3 (large): {tier_stats['tier_3_large']}")
                if tier_stats['total_queries'] > 0:
                    print(f"  Cache hit rate: {tier_stats['cache_hit_rate']:.1f}%")
                    print(f"  Avg similarity: {tier_stats['avg_similarity']:.3f}")
                print()
                continue
            
            if user_input.lower() == 'generate':
                generate_questions_for_cache(db, embedding_gen, question_gen)
                continue
            
            if not user_input:
                continue
            
            # Handle query with LLM
            print()
            result = llm_handler.answer_query(user_input)
            
            # Display results
            print(f"{'='*70}")
            print(f"Tier: {result['tier']} ({result['tier_name']})")
            print(f"Similarity: {result['similarity']:.3f}")
            print(f"Response time: {result['metadata'].get('response_time', 0.0):.2f}s")
            print(f"Reasoning: {result['reasoning']}")
            print(f"{'='*70}")
            print(f"\n{result['answer']}\n")
            
        except KeyboardInterrupt:
            print("\n")
            break
        except Exception as e:
            print(f"Error: {str(e)}\n")
    
    # Final statistics
    print("="*70)
    print("SESSION SUMMARY")
    print("="*70)
    
    tier_stats = evaluator.get_tier_statistics()
    if tier_stats['total_queries'] > 0:
        print(f"Total queries: {tier_stats['total_queries']}")
        print(f"Tier 1 (cache): {tier_stats['tier_1_cache']}")
        print(f"Tier 2 (small): {tier_stats['tier_2_small']}")
        print(f"Tier 3 (large): {tier_stats['tier_3_large']}")
        print(f"Cache hit rate: {tier_stats['cache_hit_rate']:.1f}%")
    else:
        print("No queries processed")
    
    print("\n" + "="*70)
    print("Session ended")
    print("="*70)


if __name__ == "__main__":
    main()