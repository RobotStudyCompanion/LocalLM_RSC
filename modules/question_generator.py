"""
Question Generator Module
Automatically generates question-answer pairs from documents
Uses large model to create high-quality Q&A for cache population
"""

import ollama
from typing import List, Dict, Optional
from modules.database import VectorDatabase
from modules.embeddings import EmbeddingGenerator
from config import LARGE_MODEL
from modules.prompts import get_question_generation_prompt


class QuestionGenerator:
    """
    Generates question-answer pairs from document chunks
    """
    
    def __init__(
        self,
        db: VectorDatabase,
        embedding_gen: EmbeddingGenerator,
        model: str = LARGE_MODEL
    ):
        """
        Initialize the question generator
        
        Args:
            db: VectorDatabase instance
            embedding_gen: EmbeddingGenerator instance
            model: Ollama model to use for generation
        """
        self.db = db
        self.embedding_gen = embedding_gen
        self.model = model
    
    def generate_questions_from_chunk(
        self,
        chunk: str,
        num_questions: int = 3
    ) -> List[Dict]:
        """
        Generate questions from a single text chunk

        Args:
            chunk: Text chunk to generate questions from
            num_questions: Number of questions to generate

        Returns:
            List of dicts with 'question' and 'answer' keys
        """
        prompt = get_question_generation_prompt(chunk, num_questions)

        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt
            )

            generated_text = response['response'].strip()
            qa_pairs = self._parse_qa_response(generated_text)

            return qa_pairs

        except Exception as e:
            print(f"Error generating questions: {str(e)}")
            return []
    
    def _parse_qa_response(self, response_text: str) -> List[Dict]:
        """
        Parse the model's response into Q&A pairs

        Args:
            response_text: Raw response from model

        Returns:
            List of dicts with 'question' and 'answer' keys
        """
        import re

        qa_pairs = []

        # Try multiple parsing strategies

        # Strategy 1: Q1:/A1: format with numbers
        pattern1 = r'Q\d+:\s*(.+?)\s*A\d+:\s*(.+?)(?=Q\d+:|$)'
        matches1 = re.findall(pattern1, response_text, re.DOTALL | re.IGNORECASE)

        if matches1:
            for q, a in matches1:
                qa_pairs.append({
                    'question': q.strip(),
                    'answer': a.strip()
                })
            return qa_pairs

        # Strategy 2: Simple Q:/A: format (original method)
        lines = response_text.split('\n')
        current_question = None
        current_answer = None

        for line in lines:
            line = line.strip()

            # Check for question (Q, Q:, Question:)
            if (line.startswith('Q') or line.lower().startswith('question')) and ':' in line:
                # Save previous Q&A if exists
                if current_question and current_answer:
                    qa_pairs.append({
                        'question': current_question,
                        'answer': current_answer
                    })

                # Extract new question
                current_question = line.split(':', 1)[1].strip()
                current_answer = None

            # Check for answer (A, A:, Answer:)
            elif (line.startswith('A') or line.lower().startswith('answer')) and ':' in line:
                current_answer = line.split(':', 1)[1].strip()

        # Add last Q&A pair
        if current_question and current_answer:
            qa_pairs.append({
                'question': current_question,
                'answer': current_answer
            })

        # Debug output if parsing failed
        if len(qa_pairs) == 0:
            print(f"\nDebug - Failed to parse Q&A. Response text:")
            print("=" * 70)
            print(response_text)
            print("=" * 70)

        return qa_pairs
    
    def generate_and_cache_questions(
        self,
        max_chunks: Optional[int] = None,
        questions_per_chunk: int = 3
    ) -> Dict:
        """
        Generate questions from all documents and cache them
        
        Args:
            max_chunks: Maximum number of chunks to process (None = all)
            questions_per_chunk: Number of questions per chunk
            
        Returns:
            Dict with generation statistics
        """
        # Get all document chunks
        doc_results = self.db.get_document_chunks(n_results=10000)

        # Debug: Print what we got
        print(f"\nDebug - doc_results keys: {doc_results.keys() if doc_results else 'None'}")
        if doc_results and 'documents' in doc_results:
            print(f"Debug - Number of documents retrieved: {len(doc_results['documents'])}")

        if not doc_results or 'documents' not in doc_results:
            return {
                "success": False,
                "error": "No documents found in database",
                "total_chunks": 0,
                "questions_generated": 0,
                "questions_cached": 0
            }

        chunks = doc_results['documents']
        metadatas = doc_results.get('metadatas', [])

        # Check if chunks is empty
        if not chunks or len(chunks) == 0:
            print("Debug - chunks list is empty!")
            return {
                "success": False,
                "error": "No document chunks available",
                "total_chunks": 0,
                "questions_generated": 0,
                "questions_cached": 0
            }
        
        # Limit chunks if specified
        if max_chunks:
            chunks = chunks[:max_chunks]
            metadatas = metadatas[:max_chunks]
        
        total_chunks = len(chunks)
        questions_generated = 0
        questions_cached = 0
        
        print(f"\nGenerating questions from {total_chunks} chunks...")
        print(f"Using model: {self.model}")
        print(f"Questions per chunk: {questions_per_chunk}\n")
        
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{total_chunks}...", end=" ")
            
            # Generate questions from chunk
            qa_pairs = self.generate_questions_from_chunk(chunk, questions_per_chunk)
            
            if not qa_pairs:
                print("[X] Failed")
                continue
            
            print(f"[OK] Generated {len(qa_pairs)} Q&A pairs")
            questions_generated += len(qa_pairs)
            
            # Cache each Q&A pair
            for qa in qa_pairs:
                question = qa['question']
                answer = qa['answer']
                
                # Generate embedding for question
                question_embedding = self.embedding_gen.generate_embedding(question)
                
                if question_embedding:
                    success = self.db.store_question_answer(
                        question=question,
                        answer=answer,
                        question_embedding=question_embedding,
                        accuracy=1.0,
                        model_used=self.model,
                        metadata={
                            "source": "auto_generated",
                            "chunk_index": i,
                            "source_file": metadatas[i].get('source_file', 'unknown') if i < len(metadatas) else 'unknown'
                        }
                    )
                    
                    if success:
                        questions_cached += 1
        
        return {
            "success": True,
            "total_chunks": total_chunks,
            "questions_generated": questions_generated,
            "questions_cached": questions_cached
        }
    
    def generate_questions_batch(
        self,
        chunks: List[str],
        questions_per_chunk: int = 3
    ) -> List[Dict]:
        """
        Generate questions from a batch of chunks
        
        Args:
            chunks: List of text chunks
            questions_per_chunk: Number of questions per chunk
            
        Returns:
            List of all Q&A pairs generated
        """
        all_qa_pairs = []
        
        for i, chunk in enumerate(chunks):
            qa_pairs = self.generate_questions_from_chunk(chunk, questions_per_chunk)
            
            for qa in qa_pairs:
                qa['source_chunk_index'] = i
                all_qa_pairs.append(qa)
        
        return all_qa_pairs
    
    def verify_model(self) -> bool:
        """
        Verify that the model is available
        
        Returns:
            bool: True if model is available
        """
        try:
            models = ollama.list()
            model_names = [model['model'] for model in models.get('models', [])]
            
            base_model_name = self.model.split(':')[0]
            available = any(base_model_name in name for name in model_names)
            
            return available
        except:
            return False

    

def generate_and_cache_all(
    db: VectorDatabase,
    embedding_gen: EmbeddingGenerator,
    max_chunks: Optional[int] = None,
    questions_per_chunk: int = 3
) -> Dict:
    """
    Convenience function to generate and cache questions
    
    Args:
        db: VectorDatabase instance
        embedding_gen: EmbeddingGenerator instance
        max_chunks: Maximum chunks to process
        questions_per_chunk: Questions per chunk
        
    Returns:
        Generation statistics
    """
    generator = QuestionGenerator(db, embedding_gen)
    return generator.generate_and_cache_questions(max_chunks, questions_per_chunk)
