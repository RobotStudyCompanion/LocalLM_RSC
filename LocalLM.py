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
            print(f"  ⚠ Dimension mismatch detected")
            print(f"  Clearing documents collection...")
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


def handle_query(
    query: str,
    evaluator: AccuracyEvaluator,
    db: VectorDatabase
) -> Dict:
    """
    Handle a query with tier evaluation
    
    Args:
        query: User query
        evaluator: AccuracyEvaluator instance
        db: VectorDatabase instance
        
    Returns:
        Dict with answer and metadata
    """
    # Evaluate query and determine tier
    evaluation = evaluator.evaluate_query(query)
    
    tier = evaluation["tier"]
    tier_name = evaluation["tier_name"]
    max_similarity = evaluation["max_similarity"]
    
    # Get context for the query
    context = evaluator.get_context_for_tier(evaluation, max_context_items=MAX_CONTEXT_ITEMS)
    
    # For now, we'll return a placeholder answer
    # This will be replaced with actual LLM calls in llm_handler.py
    if tier == 1:
        # Tier 1: Use cached answer
        cached_item = evaluation["similar_items"][0] if evaluation["similar_items"] else None
        if cached_item and cached_item["source"] == "cache":
            answer = cached_item["metadata"].get("answer", "No cached answer found")
        else:
            answer = "[Cache tier selected but no exact match found]"
    
    elif tier == 2:
        # Tier 2: Small model with context
        answer = f"[Would use small model ({SMALL_MODEL}) with {len(context)} context items]"
    
    else:
        # Tier 3: Large model with context
        answer = f"[Would use large model ({LARGE_MODEL}) with {len(context)} context items]"
    
    return {
        "answer": answer,
        "tier": tier,
        "tier_name": tier_name,
        "similarity": max_similarity,
        "context_items": len(context),
        "reasoning": evaluation["reasoning"]
    }


def main():
    """Main execution flow"""
    print("\n" + "="*70)
    print("THREE-TIER LLM SYSTEM")
    print("="*70 + "\n")
    
    # Initialize
    print("Initializing components...")
    embedding_gen = EmbeddingGenerator(model_name=EMBEDDING_MODEL, batch_size=EMBEDDING_BATCH_SIZE)
    
    test_emb = embedding_gen.generate_embedding("test")
    if not test_emb:
        print("✗ Failed to generate test embedding")
        print("  Make sure Ollama is running: ollama serve")
        print(f"  Make sure model is installed: ollama pull {EMBEDDING_MODEL}")
        return
    
    print(f"✓ Embedding model: {EMBEDDING_MODEL} ({len(test_emb)}D)")
    
    db = VectorDatabase(db_path=VECTOR_DB_PATH)
    if not db.initialize_db():
        print("✗ Failed to initialize database")
        return
    
    print("✓ Database initialized")
    
    evaluator = AccuracyEvaluator(db, embedding_gen)
    print(f"✓ Accuracy evaluator initialized")
    print(f"  Tier 1 threshold: >= {HIGH_SIMILARITY_THRESHOLD}")
    print(f"  Tier 2 threshold: >= {MEDIUM_SIMILARITY_THRESHOLD}")
    print(f"  Tier 3 threshold: < {MEDIUM_SIMILARITY_THRESHOLD}\n")
    
    # Find and process PDFs
    print(f"Searching for PDFs in: {DOCUMENTS_DIR}")
    pdf_files = list(Path(DOCUMENTS_DIR).glob("*.pdf"))
    
    if not pdf_files:
        print("✗ No PDF files found")
        print(f"  Add PDFs to: {DOCUMENTS_DIR}")
    else:
        print(f"✓ Found {len(pdf_files)} PDF(s)\n")
        
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
    

    
    # Interactive query loop
    print("="*70)
    print("INTERACTIVE QUERY MODE (with Tier Evaluation)")
    print("="*70)
    print("Commands:")
    print("  'quit' - Exit")
    print("  'stats' - Database statistics")
    print("  'tier-stats' - Tier usage statistics")
    print()
    
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
            
            if user_input.lower() == 'tier-stats':
                tier_stats = evaluator.get_tier_statistics()
                print(f"\nTier Usage Statistics:")
                print(f"  Total queries: {tier_stats['total_queries']}")
                print(f"  Tier 1 (cache): {tier_stats['tier_1_cache']}")
                print(f"  Tier 2 (small): {tier_stats['tier_2_small']}")
                print(f"  Tier 3 (large): {tier_stats['tier_3_large']}")
                print(f"  Cache hit rate: {tier_stats['cache_hit_rate']:.1f}%")
                print(f"  Avg similarity: {tier_stats['avg_similarity']:.3f}\n")
                continue
            
            if not user_input:
                continue
            
            # Handle query with tier evaluation
            result = handle_query(user_input, evaluator, db)
            
            # Display results
            print(f"\n{'='*70}")
            print(f"Tier: {result['tier']} ({result['tier_name']})")
            print(f"Similarity: {result['similarity']:.3f}")
            print(f"Context items: {result['context_items']}")
            print(f"Reasoning: {result['reasoning']}")
            print(f"{'='*70}")
            print(f"\nAnswer: {result['answer']}\n")
            
            # Store in user history
            db.store_user_interaction(
                question=user_input,
                answer=result['answer'],
                question_embedding=embedding_gen.generate_embedding(user_input),
                model_used=result['tier_name'],
                similarity_score=result['similarity']
            )
            
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