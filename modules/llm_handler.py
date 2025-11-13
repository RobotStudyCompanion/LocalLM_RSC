"""
LLM Handler Module
Handles three-tier LLM routing and response generation
Tier 1: Cache lookup
Tier 2: Small model (1B-2B) with context
Tier 3: Large model (4B-7B) with context
"""

import ollama
from typing import List, Dict, Optional
from modules.database import VectorDatabase
from modules.embeddings import EmbeddingGenerator
from modules.accuracy_evaluator import AccuracyEvaluator
from config import *


class LLMHandler:
    """
    Handles LLM interactions across all three tiers
    """
    
    def __init__(
        self,
        db: VectorDatabase,
        embedding_gen: EmbeddingGenerator,
        evaluator: AccuracyEvaluator,
        small_model: str = SMALL_MODEL,
        large_model: str = LARGE_MODEL
    ):
        """
        Initialize the LLM handler
        
        Args:
            db: VectorDatabase instance
            embedding_gen: EmbeddingGenerator instance
            evaluator: AccuracyEvaluator instance
            small_model: Name of small Ollama model
            large_model: Name of large Ollama model
        """
        self.db = db
        self.embedding_gen = embedding_gen
        self.evaluator = evaluator
        self.small_model = small_model
        self.large_model = large_model
    
    def answer_query(self, query: str) -> Dict:
        """
        Answer a query using the three-tier system
        
        Args:
            query: User query text
            
        Returns:
            Dict with answer, tier used, and metadata
        """
        # Evaluate query to determine tier
        evaluation = self.evaluator.evaluate_query(query)
        
        tier = evaluation["tier"]
        tier_name = evaluation["tier_name"]
        max_similarity = evaluation["max_similarity"]
        
        # Route to appropriate tier
        if tier == 1:
            answer, metadata = self._handle_tier1_cache(query, evaluation)
        elif tier == 2:
            answer, metadata = self._handle_tier2_small(query, evaluation)
        else:
            answer, metadata = self._handle_tier3_large(query, evaluation)
        
        # Store in user history
        query_embedding = evaluation.get("query_embedding")
        if query_embedding:
            self.db.store_user_interaction(
                question=query,
                answer=answer,
                question_embedding=query_embedding,
                model_used=tier_name,
                similarity_score=max_similarity,
                response_time=metadata.get("response_time", 0.0)
            )
        
        # Cache answer if appropriate
        if self.evaluator.should_cache_answer(query, answer, tier):
            if query_embedding:
                self.db.store_question_answer(
                    question=query,
                    answer=answer,
                    question_embedding=query_embedding,
                    accuracy=max_similarity,
                    model_used=tier_name
                )
        
        return {
            "answer": answer,
            "tier": tier,
            "tier_name": tier_name,
            "similarity": max_similarity,
            "metadata": metadata,
            "reasoning": evaluation["reasoning"]
        }
    
    def _handle_tier1_cache(self, query: str, evaluation: Dict) -> tuple:
        """
        Handle tier 1: Return cached answer
        
        Args:
            query: User query
            evaluation: Evaluation result
            
        Returns:
            Tuple of (answer, metadata)
        """
        similar_items = evaluation.get("similar_items", [])
        
        # Find cached answer
        for item in similar_items:
            if item["source"] == "cache":
                cached_answer = item["metadata"].get("answer", "")
                if cached_answer:
                    metadata = {
                        "response_time": 0.0,
                        "cached_question": item["text"],
                        "cache_similarity": item["similarity"]
                    }
                    return cached_answer, metadata
        
        # Fallback if no cache found (shouldn't happen)
        return "No cached answer found", {"response_time": 0.0}
    
    def _handle_tier2_small(self, query: str, evaluation: Dict) -> tuple:
        """
        Handle tier 2: Use small model with context
        
        Args:
            query: User query
            evaluation: Evaluation result
            
        Returns:
            Tuple of (answer, metadata)
        """
        import time
        
        # Get context
        context = self.evaluator.get_context_for_tier(evaluation, max_context_items=MAX_CONTEXT_ITEMS)
        
        # Build prompt
        prompt = self._build_prompt(query, context)
        
        # Call small model
        start_time = time.time()
        try:
            response = ollama.generate(
                model=self.small_model,
                prompt=prompt
            )
            answer = response['response'].strip()
            response_time = time.time() - start_time
            
            metadata = {
                "response_time": response_time,
                "context_items": len(context),
                "model": self.small_model
            }
            
            return answer, metadata
            
        except Exception as e:
            return f"Error with small model: {str(e)}", {"response_time": 0.0, "error": str(e)}
    
    def _handle_tier3_large(self, query: str, evaluation: Dict) -> tuple:
        """
        Handle tier 3: Use large model with context
        
        Args:
            query: User query
            evaluation: Evaluation result
            
        Returns:
            Tuple of (answer, metadata)
        """
        import time
        
        # Get context
        context = self.evaluator.get_context_for_tier(evaluation, max_context_items=MAX_CONTEXT_ITEMS)
        
        # Build prompt
        prompt = self._build_prompt(query, context)
        
        # Call large model
        start_time = time.time()
        try:
            response = ollama.generate(
                model=self.large_model,
                prompt=prompt
            )
            answer = response['response'].strip()
            response_time = time.time() - start_time
            
            metadata = {
                "response_time": response_time,
                "context_items": len(context),
                "model": self.large_model
            }
            
            return answer, metadata
            
        except Exception as e:
            return f"Error with large model: {str(e)}", {"response_time": 0.0, "error": str(e)}
    
    def _build_prompt(self, query: str, context: List[str]) -> str:
        """
        Build a prompt with query and context
        
        Args:
            query: User query
            context: List of context strings
            
        Returns:
            Formatted prompt string
        """
        if not context:
            return f"Question: {query}\n\nAnswer:"
        
        context_text = "\n\n".join([f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(context)])
        
        prompt = f"""You are a helpful assistant. Answer the question based on the provided context.

{context_text}

Question: {query}

Answer:"""
        
        return prompt
    
    def verify_models(self) -> Dict:
        """
        Verify that required models are available
        
        Returns:
            Dict with model availability status
        """
        try:
            models = ollama.list()
            model_names = [model['model'] for model in models.get('models', [])]
            
            small_available = any(self.small_model.split(':')[0] in name for name in model_names)
            large_available = any(self.large_model.split(':')[0] in name for name in model_names)
            
            return {
                "small_model": {
                    "name": self.small_model,
                    "available": small_available
                },
                "large_model": {
                    "name": self.large_model,
                    "available": large_available
                },
                "all_available": small_available and large_available
            }
        except Exception as e:
            return {
                "error": str(e),
                "all_available": False
            }


def answer_query_simple(
    query: str,
    db: VectorDatabase,
    embedding_gen: EmbeddingGenerator,
    evaluator: AccuracyEvaluator
) -> Dict:
    """
    Simple convenience function to answer a query
    
    Args:
        query: User query text
        db: VectorDatabase instance
        embedding_gen: EmbeddingGenerator instance
        evaluator: AccuracyEvaluator instance
        
    Returns:
        Answer result dictionary
    """
    handler = LLMHandler(db, embedding_gen, evaluator)
    return handler.answer_query(query)

