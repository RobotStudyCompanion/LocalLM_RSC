import ollama
from typing import List, Dict, Optional
import numpy as np
from config import *

class EmbeddingGenerator:
    
    def __init__(self, model_name: str = EMBEDDING_MODEL, batch_size = EMBEDDING_BATCH_SIZE):
        """
        Args:
            model_name: Name of the Ollama embedding model
            batch_size: Number of texts to process in each batch
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.embedding_dim = None
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text
        Args:
            text: Text string to embed   
        Returns:
            Embedding vector or None if failed
        """
        try:
            response = ollama.embeddings(model=self.model_name, prompt=text)
            embedding = response['embedding']
            
            if self.embedding_dim is None:
                self.embedding_dim = len(embedding)
            
            return embedding
        except:
            return None
    
    def generate_embeddings_batch(self, texts: List[str]) -> Dict:
        """
        Generate embeddings for multiple texts
        Args:
            texts: List of text strings     
        Returns:
            Dict containing:
                - embeddings: List of embedding vectors (None for failed)
                - failed_indices: List of indices that failed
                - total: Total number of texts
                - successful: Number of successful embeddings
                - failed: Number of failed embeddings
                - success: True if all succeeded
        """
        embeddings = []
        failed_indices = []
        
        for i, text in enumerate(texts):
            embedding = self.generate_embedding(text)
            
            if embedding:
                embeddings.append(embedding)
            else:
                embeddings.append(None)
                failed_indices.append(i)
        
        return {
            "embeddings": embeddings,
            "failed_indices": failed_indices,
            "total": len(texts),
            "successful": len(texts) - len(failed_indices),
            "failed": len(failed_indices),
            "success": len(failed_indices) == 0
        }
    
    def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Compute cosine similarity between two embeddings
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector   
        Returns:
            Cosine similarity score (0-1, higher = more similar)
        """
        if not embedding1 or not embedding2:
            return 0.0
        
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def find_most_similar(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Find most similar embeddings to a query
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: List of candidate embeddings
            top_k: Number of top results to return
        Returns:
            List of dicts with 'index' and 'similarity' keys, sorted by similarity
        """
        similarities = []
        
        for i, candidate in enumerate(candidate_embeddings):
            if candidate is None:
                continue
            
            similarity = self.compute_similarity(query_embedding, candidate)
            similarities.append({"index": i, "similarity": similarity})
        
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_k]