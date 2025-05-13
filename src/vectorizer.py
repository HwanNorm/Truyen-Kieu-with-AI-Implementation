import numpy as np
from typing import List, Dict
from collections import Counter
from scipy.sparse import csr_matrix
import pickle

class TfidfVectorizer:
    def __init__(self):
        self.vocabulary = {}  # Map words to indices
        self.idf = None       # IDF values for each term
        self.doc_count = 0    # Number of documents (verses)
    
    def fit(self, tokenized_verses: List[List[str]]):
        """Build vocabulary and calculate IDF values"""
        # Create vocabulary
        unique_words = set()
        for verse_tokens in tokenized_verses:
            unique_words.update(verse_tokens)
        
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(unique_words))}
        self.doc_count = len(tokenized_verses)
        
        # Calculate IDF
        doc_freq = np.zeros(len(self.vocabulary))
        for verse_tokens in tokenized_verses:
            # Count each word only once per document
            word_presence = set(verse_tokens)
            for word in word_presence:
                if word in self.vocabulary:
                    doc_freq[self.vocabulary[word]] += 1
        
        # Calculate IDF: log(N/df)
        # Add 1 to prevent division by zero
        self.idf = np.log(self.doc_count / (1 + doc_freq))
        
        return self
    
    def transform(self, tokenized_verses: List[List[str]]) -> csr_matrix:
        """Transform documents into TF-IDF matrix"""
        rows, cols, data = [], [], []
        
        for doc_idx, verse_tokens in enumerate(tokenized_verses):
            # Calculate term frequencies
            term_freq = Counter(verse_tokens)
            
            for term, freq in term_freq.items():
                if term in self.vocabulary:
                    term_idx = self.vocabulary[term]
                    rows.append(doc_idx)
                    cols.append(term_idx)
                    # TF-IDF score
                    data.append(freq * self.idf[term_idx])
        
        # Create sparse matrix
        tfidf_matrix = csr_matrix((data, (rows, cols)), 
                                 shape=(len(tokenized_verses), len(self.vocabulary)))
        
        # Normalize vectors to unit length
        norms = np.sqrt(tfidf_matrix.multiply(tfidf_matrix).sum(axis=1))
        normalized_matrix = csr_matrix(tfidf_matrix)
        for i in range(tfidf_matrix.shape[0]):
            if norms[i, 0] > 0:
                normalized_matrix[i] = tfidf_matrix[i] / norms[i, 0]
        
        return normalized_matrix
    
    def fit_transform(self, tokenized_verses: List[List[str]]) -> csr_matrix:
        """Fit and transform in one step"""
        self.fit(tokenized_verses)
        return self.transform(tokenized_verses)
    
    def save(self, filepath: str):
        """Save the vectorizer to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath: str):
        """Load a vectorizer from disk"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
