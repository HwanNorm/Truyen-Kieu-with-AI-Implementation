import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from typing import List, Tuple
import pickle

class AuthorshipClassifier:
    def __init__(self, classifier=None):
        self.classifier = classifier or SVC(kernel='linear', probability=True)
    
    def extract_features(self, tfidf_matrix, tokenized_verses):
        """Extract additional stylometric features"""
        # Get base TF-IDF features
        features = tfidf_matrix.toarray()
        
        # Additional features (example: verse length, special words frequency)
        verse_lengths = np.array([len(verse) for verse in tokenized_verses])
        verse_lengths = verse_lengths.reshape(-1, 1) / max(verse_lengths)  # Normalize
        
        # Combine all features
        return np.hstack((features, verse_lengths))
    
    def train(self, features, labels):
        """Train the authorship classifier"""
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42)
        
        self.classifier.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.classifier.predict(X_test)
        print(classification_report(y_test, y_pred))
        
        return self
    
    def predict(self, features):
        """Predict authorship with confidence score"""
        if not hasattr(self.classifier, 'predict_proba'):
            return self.classifier.predict(features)
        
        proba = self.classifier.predict_proba(features)
        predictions = self.classifier.predict(features)
        
        # Get confidence scores (probability of predicted class)
        confidence = np.array([proba[i, predictions[i]] for i in range(len(predictions))])
        
        return predictions, confidence
    
    def save(self, filepath):
        """Save the model to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath):
        """Load a model from disk"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)