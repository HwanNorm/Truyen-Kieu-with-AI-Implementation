import re
import unicodedata
from typing import List, Dict
from underthesea import word_tokenize  # Vietnamese tokenizer

class KieuPreprocessor:
    def __init__(self, stopwords_file=None):
        self.stopwords = set()
        if stopwords_file:
            with open(stopwords_file, 'r', encoding='utf-8') as f:
                self.stopwords = set(line.strip() for line in f)
    
    def load_poem(self, filepath: str) -> List[str]:
        """Load Truyện Kiều and split into verses"""
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
    
        verses = []
        for line in text.split('\n'):
            line = line.strip()
            if line:
                # Remove line numbers if present
                line = re.sub(r'^\d+\.\s*', '', line)
                verses.append(line)
    
        return verses
    
    def preprocess_verse(self, verse: str) -> List[str]:
        """Preprocess a single verse"""
        # Normalize unicode for Vietnamese text
        verse = unicodedata.normalize('NFC', verse)
        
        # Remove punctuation and digits
        verse = re.sub(r'[^\w\s]', '', verse)
        verse = re.sub(r'\d+', '', verse)
        
        # Tokenize (specific to Vietnamese)
        tokens = word_tokenize(verse, format="text").split()
        
        # Remove stopwords and convert to lowercase
        tokens = [token.lower() for token in tokens if token.lower() not in self.stopwords]
        
        return tokens
    
    def preprocess_all_verses(self, verses: List[str]) -> List[List[str]]:
        """Preprocess all verses in the poem"""
        return [self.preprocess_verse(verse) for verse in verses]