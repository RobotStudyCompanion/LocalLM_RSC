"""
Cache Inspector
View and analyze cached questions
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import VECTOR_DB_PATH
from modules.database import VectorDatabase


def display_cached_questions(db: VectorDatabase, max_display: int = None):
    """
    Display all cached questions with details
    
    Args:
        db: VectorDatabase instance
        max_display: Maximum questions to display (None = all)
    """
    cache_collection = db.collections[db.QUESTIONS_CACHE_COLLECTION]
    results = cache_collection.get(limit=10000)
    
    if not results or not results['documents']:
        print("No cached questions found")
        return
    
    total = len(results['documents'])
    print(f"\n{'='*70}")
    print(f"CACHED QUESTIONS ({total} total)")
    print(f"{'='*70}\n")
    
    questions_to_show = max_display if max_display else total
    
    for i in range(min(questions_to_show, total)):
        question = results['documents'][i]
        metadata = results['metadatas'][i]
        
        answer = metadata.get('answer', 'No answer')
        source = metadata.get('source', 'unknown')
        model = metadata.get('model_used', 'unknown')
        accuracy = metadata.get('accuracy', 0.0)
        usage_count = metadata.get('usage_count', 1)
        source_file = metadata.get('source_file', 'unknown')
        
        print(f"{'─'*70}")
        print(f"Question {i+1}:")
        print(f"{'─'*70}")
        print(f"Q: {question}")
        print(f"\nA: {answer}")
        print(f"\nMetadata:")
        print(f"  Source: {source}")
        print(f"  Source File: {source_file}")
        print(f"  Model: {model}")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Usage Count: {usage_count}")
        print()


def search_cached_questions(db: VectorDatabase, search_term: str):
    """
    Search cached questions by keyword
    
    Args:
        db: VectorDatabase instance
        search_term: Keyword to search for
    """
    cache_collection = db.collections[db.QUESTIONS_CACHE_COLLECTION]
    results = cache_collection.get(limit=10000)
    
    if not results or not results['documents']:
        print("No cached questions found")
        return
    
    search_term_lower = search_term.lower()
    matches = []
    
    for i, question in enumerate(results['documents']):
        if search_term_lower in question.lower():
            matches.append((i, question, results['metadatas'][i]))
    
    print(f"\n{'='*70}")
    print(f"SEARCH RESULTS for '{search_term}' ({len(matches)} matches)")
    print(f"{'='*70}\n")
    
    for idx, (i, question, metadata) in enumerate(matches):
        answer = metadata.get('answer', 'No answer')
        print(f"[{idx+1}] {question}")
        print(f"    A: {answer[:100]}{'...' if len(answer) > 100 else ''}")
        print()


def show_statistics(db: VectorDatabase):
    """
    Show statistics about cached questions
    
    Args:
        db: VectorDatabase instance
    """
    cache_collection = db.collections[db.QUESTIONS_CACHE_COLLECTION]
    results = cache_collection.get(limit=10000)
    
    if not results or not results['documents']:
        print("No cached questions found")
        return
    
    total = len(results['documents'])
    
    # Analyze metadata
    sources = {}
    models = {}
    source_files = {}
    total_usage = 0
    
    for metadata in results['metadatas']:
        source = metadata.get('source', 'unknown')
        sources[source] = sources.get(source, 0) + 1
        
        model = metadata.get('model_used', 'unknown')
        models[model] = models.get(model, 0) + 1
        
        source_file = metadata.get('source_file', 'unknown')
        source_files[source_file] = source_files.get(source_file, 0) + 1
        
        total_usage += metadata.get('usage_count', 1)
    
    avg_usage = total_usage / total if total > 0 else 0
    
    print(f"\n{'='*70}")
    print("CACHE STATISTICS")
    print(f"{'='*70}\n")
    
    print(f"Total Questions: {total}")
    print(f"Total Usage Count: {total_usage}")
    print(f"Average Usage per Question: {avg_usage:.2f}\n")
    
    print("By Source:")
    for source, count in sources.items():
        percentage = (count / total * 100) if total > 0 else 0
        print(f"  {source}: {count} ({percentage:.1f}%)")
    
    print("\nBy Model:")
    for model, count in models.items():
        percentage = (count / total * 100) if total > 0 else 0
        print(f"  {model}: {count} ({percentage:.1f}%)")
    
    print("\nBy Source File:")
    for file, count in sorted(source_files.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total * 100) if total > 0 else 0
        print(f"  {file}: {count} ({percentage:.1f}%)")
    print()


def export_to_file(db: VectorDatabase, output_file: str = "cached_questions.txt"):
    """
    Export cached questions to a text file
    
    Args:
        db: VectorDatabase instance
        output_file: Output file path
    """
    cache_collection = db.collections[db.QUESTIONS_CACHE_COLLECTION]
    results = cache_collection.get(limit=10000)
    
    if not results or not results['documents']:
        print("No cached questions found")
        return
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write(f"CACHED QUESTIONS ({len(results['documents'])} total)\n")
        f.write("="*70 + "\n\n")
        
        for i, question in enumerate(results['documents']):
            metadata = results['metadatas'][i]
            answer = metadata.get('answer', 'No answer')
            source = metadata.get('source', 'unknown')
            model = metadata.get('model_used', 'unknown')
            
            f.write(f"Question {i+1}:\n")
            f.write(f"{'-'*70}\n")
            f.write(f"Q: {question}\n\n")
            f.write(f"A: {answer}\n\n")
            f.write(f"Source: {source} | Model: {model}\n")
            f.write("="*70 + "\n\n")
    
    print(f"✓ Exported {len(results['documents'])} questions to {output_file}")


def main():
    """Main menu for cache inspection"""
    print("\n" + "="*70)
    print("CACHE INSPECTOR")
    print("="*70 + "\n")
    
    # Initialize database
    db = VectorDatabase(db_path=VECTOR_DB_PATH)
    if not db.initialize_db():
        print("✗ Failed to initialize database")
        return
    
    while True:
        print("\nOptions:")
        print("  1. View all cached questions")
        print("  2. View first N questions")
        print("  3. Search questions by keyword")
        print("  4. Show statistics")
        print("  5. Export to file")
        print("  6. Exit")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == '1':
            display_cached_questions(db)
        
        elif choice == '2':
            try:
                n = int(input("How many questions to display? "))
                display_cached_questions(db, max_display=n)
            except ValueError:
                print("Invalid number")
        
        elif choice == '3':
            search_term = input("Enter search term: ").strip()
            if search_term:
                search_cached_questions(db, search_term)
        
        elif choice == '4':
            show_statistics(db)
        
        elif choice == '5':
            output_file = input("Output file name (default: cached_questions.txt): ").strip()
            if not output_file:
                output_file = "cached_questions.txt"
            export_to_file(db, output_file)
        
        elif choice == '6':
            print("\nGoodbye!")
            break
        
        else:
            print("Invalid option")


if __name__ == "__main__":
    main()