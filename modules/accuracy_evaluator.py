"""
Accuracy Evaluator Module
Determines which tier should handle a query based on similarity scoring
Enhanced with cross-encoder re-ranking for better semantic matching
"""

from typing import List, Dict, Optional, Tuple
from modules.embeddings import EmbeddingGenerator
from modules.database import VectorDatabase
from config import (
    HIGH_SIMILARITY_THRESHOLD,
    MEDIUM_SIMILARITY_THRESHOLD,
    USE_CROSS_ENCODER,
    CROSS_ENCODER_MODEL,
    CROSS_ENCODER_TOP_K,
    EMBEDDING_WEIGHT,
    CROSS_ENCODER_WEIGHT
)

# Try to import cross-encoder (optional dependency)
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    print("[WARNING] sentence-transformers not installed. Cross-encoder re-ranking disabled.")
    print("         Install with: pip install sentence-transformers")


class AccuracyEvaluator:
    """
    Evaluates query similarity and routes to appropriate tier
    Enhanced with cross-encoder re-ranking for better semantic meaning matching
    Tier 1: Cache (high similarity >= 0.90)
    Tier 2: Small model (medium similarity >= 0.70)
    Tier 3: Large model (low similarity < 0.70)
    """

    def __init__(
        self,
        db: VectorDatabase,
        embedding_gen: EmbeddingGenerator,
        high_threshold: float = HIGH_SIMILARITY_THRESHOLD,
        medium_threshold: float = MEDIUM_SIMILARITY_THRESHOLD,
        use_cross_encoder: bool = USE_CROSS_ENCODER
    ):
        """
        Initialize the accuracy evaluator

        Args:
            db: VectorDatabase instance
            embedding_gen: EmbeddingGenerator instance
            high_threshold: Threshold for tier 1 (cache)
            medium_threshold: Threshold for tier 2 (small model)
            use_cross_encoder: Whether to use cross-encoder re-ranking
        """
        self.db = db
        self.embedding_gen = embedding_gen
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.use_cross_encoder = use_cross_encoder and CROSS_ENCODER_AVAILABLE

        # Initialize cross-encoder if enabled
        self.cross_encoder = None
        if self.use_cross_encoder:
            try:
                print(f"[INFO] Loading cross-encoder model: {CROSS_ENCODER_MODEL}")
                self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
                print("[OK] Cross-encoder loaded successfully")
            except Exception as e:
                print(f"[WARNING] Failed to load cross-encoder: {e}")
                self.use_cross_encoder = False
    
    def _rerank_with_cross_encoder(self, query: str, items: List[Dict]) -> List[Dict]:
        """
        Re-rank items using cross-encoder for better semantic matching

        Args:
            query: User query text
            items: List of similar items with embedding similarity scores

        Returns:
            List of items with updated similarity scores
        """
        if not self.use_cross_encoder or not items:
            return items

        # Only re-rank top-k candidates to save compute
        candidates_to_rerank = items[:CROSS_ENCODER_TOP_K]
        remaining_items = items[CROSS_ENCODER_TOP_K:]

        try:
            # Prepare pairs for cross-encoder
            pairs = []
            for item in candidates_to_rerank:
                text = item["text"]
                # For cache items, we want to match the question, not the answer
                pairs.append((query, text))

            # Get cross-encoder scores
            cross_scores = self.cross_encoder.predict(pairs)

            # Normalize cross-encoder scores to 0-1 range
            # MS-MARCO model outputs logits, typically in range [-10, 10]
            normalized_scores = []
            for score in cross_scores:
                # Sigmoid-like normalization
                normalized = 1 / (1 + pow(2.71828, -score))
                normalized_scores.append(normalized)

            # Update items with combined scores
            for i, item in enumerate(candidates_to_rerank):
                embedding_sim = item["similarity"]
                cross_sim = normalized_scores[i]

                # Weighted combination of embedding and cross-encoder scores
                final_similarity = (
                    EMBEDDING_WEIGHT * embedding_sim +
                    CROSS_ENCODER_WEIGHT * cross_sim
                )

                # Store both scores for debugging (convert to Python float for ChromaDB compatibility)
                item["embedding_similarity"] = float(embedding_sim)
                item["cross_encoder_score"] = float(cross_sim)
                item["similarity"] = float(final_similarity)

            # Re-sort by new similarity scores
            candidates_to_rerank.sort(key=lambda x: x["similarity"], reverse=True)

            # Combine with remaining items
            return candidates_to_rerank + remaining_items

        except Exception as e:
            print(f"[WARNING] Cross-encoder re-ranking failed: {e}")
            return items

    def evaluate_query(self, query: str, n_results: int = 5) -> Dict:
        """
        Evaluate a query and determine which tier should handle it
        Uses cross-encoder re-ranking for better semantic matching if enabled

        Args:
            query: User query text
            n_results: Number of similar items to check

        Returns:
            Dict with:
                - tier: 1, 2, or 3
                - tier_name: 'cache', 'small_model', or 'large_model'
                - max_similarity: Highest similarity score found
                - similar_items: List of similar items from database
                - query_embedding: The query embedding vector
                - reasoning: Explanation of tier selection
                - reranked: Whether cross-encoder re-ranking was applied
        """
        query_embedding = self.embedding_gen.generate_embedding(query)

        if not query_embedding:
            return {
                "tier": 3,
                "tier_name": "large_model",
                "max_similarity": 0.0,
                "similar_items": [],
                "query_embedding": None,
                "reasoning": "Failed to generate query embedding",
                "reranked": False
            }

        # Search in questions cache
        cache_results = self.db.find_similar_questions(
            question_embedding=query_embedding,
            n_results=n_results
        )

        # Search in documents
        doc_results = self.db.semantic_search(
            query_embedding=query_embedding,
            collection_name="documents",
            n_results=n_results
        )

        # Calculate max similarity from both sources
        similar_items = []

        # Check cache results
        if cache_results and 'distances' in cache_results and cache_results['distances']:
            for i, distance in enumerate(cache_results['distances'][0]):
                similarity = 1 - distance

                similar_items.append({
                    "source": "cache",
                    "similarity": similarity,
                    "text": cache_results['documents'][0][i] if cache_results['documents'] else "",
                    "metadata": cache_results['metadatas'][0][i] if cache_results['metadatas'] else {},
                    "id": cache_results['ids'][0][i] if cache_results['ids'] else ""
                })

        # Check document results
        if doc_results and 'distances' in doc_results and doc_results['distances']:
            for i, distance in enumerate(doc_results['distances'][0]):
                similarity = 1 - distance

                similar_items.append({
                    "source": "documents",
                    "similarity": similarity,
                    "text": doc_results['documents'][0][i] if doc_results['documents'] else "",
                    "metadata": doc_results['metadatas'][0][i] if doc_results['metadatas'] else {},
                    "id": doc_results['ids'][0][i] if doc_results['ids'] else ""
                })

        # Sort by initial embedding similarity
        similar_items.sort(key=lambda x: x["similarity"], reverse=True)

        # Apply cross-encoder re-ranking for better semantic matching
        reranked = False
        if self.use_cross_encoder and similar_items:
            similar_items = self._rerank_with_cross_encoder(query, similar_items)
            reranked = True

        # Keep only top n_results
        similar_items = similar_items[:n_results]

        # Get max similarity after re-ranking
        max_similarity = similar_items[0]["similarity"] if similar_items else 0.0

        # Determine tier
        tier, tier_name, reasoning = self._determine_tier(max_similarity)

        # Add re-ranking info to reasoning
        if reranked:
            reasoning += " (cross-encoder re-ranked)"

        return {
            "tier": tier,
            "tier_name": tier_name,
            "max_similarity": max_similarity,
            "similar_items": similar_items,
            "query_embedding": query_embedding,
            "reasoning": reasoning,
            "reranked": reranked
        }
    
    def _determine_tier(self, similarity: float) -> Tuple[int, str, str]:
        """
        Determine which tier based on similarity score
        
        Args:
            similarity: Similarity score (0-1)
            
        Returns:
            Tuple of (tier_number, tier_name, reasoning)
        """
        if similarity >= self.high_threshold:
            return (
                1,
                "cache",
                f"High similarity ({similarity:.3f} >= {self.high_threshold}): Using cached answer"
            )
        elif similarity >= self.medium_threshold:
            return (
                2,
                "small_model",
                f"Medium similarity ({similarity:.3f} >= {self.medium_threshold}): Using small model with context"
            )
        else:
            return (
                3,
                "large_model",
                f"Low similarity ({similarity:.3f} < {self.medium_threshold}): Using large model for complex query"
            )
    
    def get_context_for_tier(self, evaluation_result: Dict, max_context_items: int = 3) -> List[str]:
        """
        Get relevant context chunks for tier 2 and 3
        
        Args:
            evaluation_result: Result from evaluate_query
            max_context_items: Maximum number of context items to return
            
        Returns:
            List of context text chunks
        """
        similar_items = evaluation_result.get("similar_items", [])
        
        context = []
        for item in similar_items[:max_context_items]:
            if item["source"] == "documents":
                context.append(item["text"])
            elif item["source"] == "cache":
                # For cache, include both question and answer
                metadata = item.get("metadata", {})
                answer = metadata.get("answer", "")
                if answer:
                    context.append(f"Q: {item['text']}\nA: {answer}")
        
        return context
    
    def should_cache_answer(
        self,
        query: str,
        answer: str,
        tier_used: int,
        min_cache_tier: int = 2
    ) -> bool:
        """
        Determine if an answer should be cached for future use
        
        Args:
            query: Original query
            answer: Generated answer
            tier_used: Which tier was used (1, 2, or 3)
            min_cache_tier: Minimum tier that should be cached
            
        Returns:
            bool: True if answer should be cached
        """
        # Don't cache if already from cache
        if tier_used == 1:
            return False
        
        # Cache answers from tier 2 and above
        if tier_used >= min_cache_tier:
            return True
        
        return False
    


def evaluate_query_simple(
    query: str,
    db: VectorDatabase,
    embedding_gen: EmbeddingGenerator
) -> Dict:
    """
    Simple convenience function to evaluate a query
    
    Args:
        query: User query text
        db: VectorDatabase instance
        embedding_gen: EmbeddingGenerator instance
        
    Returns:
        Evaluation result dictionary
    """
    evaluator = AccuracyEvaluator(db, embedding_gen)
    return evaluator.evaluate_query(query)