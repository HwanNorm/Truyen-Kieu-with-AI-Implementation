import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Tuple

class KieuSearchEngine:
    def __init__(self, preprocessor, vectorizer, verses, tfidf_matrix):
        self.preprocessor = preprocessor
        self.vectorizer = vectorizer
        self.verses = verses  # Original verses
        self.tfidf_matrix = tfidf_matrix
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, str, float]]:
        """Search for verses matching the query"""
        # Preprocess query
        query_tokens = self.preprocessor.preprocess_verse(query)
        
        # Create query vector
        query_vector = np.zeros((1, len(self.vectorizer.vocabulary)))
        term_freq = {}
        
        # Calculate term frequencies for query
        for token in query_tokens:
            if token in self.vectorizer.vocabulary:
                term_idx = self.vectorizer.vocabulary[token]
                term_freq[term_idx] = term_freq.get(term_idx, 0) + 1
        
        # Apply TF-IDF weighting to query
        for term_idx, freq in term_freq.items():
            query_vector[0, term_idx] = freq * self.vectorizer.idf[term_idx]
        
        # Normalize query vector
        norm = np.sqrt(np.sum(query_vector ** 2))
        if norm > 0:
            query_vector = query_vector / norm
        
        # Calculate cosine similarity
        result = self.tfidf_matrix @ query_vector.T
        if hasattr(result, 'toarray'):  # Check if it's a sparse matrix
            similarity_scores = result.toarray().flatten()
        else:  # It's already a dense array
            similarity_scores = result.flatten()
        
        # Get top k results
        top_indices = np.argsort(-similarity_scores)[:top_k]
        results = []
        
        for idx in top_indices:
            if similarity_scores[idx] > 0:  # Only return relevant results
                results.append((
                    idx,  # Verse index
                    self.verses[idx],  # Original verse
                    similarity_scores[idx]  # Similarity score
                ))
        
        return results