import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import pickle

from .preprocessor import KieuPreprocessor
from .vectorizer import TfidfVectorizer

def create_binary_dataset():
    """Load all author texts and create a binary Nguyễn Du vs Others dataset"""
    # Initialize preprocessor
    preprocessor = KieuPreprocessor(stopwords_file='data/vietnamese_stopwords.txt')
    
    # Load Nguyễn Du's verses
    nguyen_du_verses = preprocessor.load_poem('data/truyen_kieu.txt')
    print(f"Loaded {len(nguyen_du_verses)} verses by Nguyễn Du")
    
    # Load comparison authors' verses
    other_verses = []
    comparison_dir = 'data/comparison_texts'
    
    for filename in os.listdir(comparison_dir):
        if filename.endswith('.txt'):
            author = filename.replace('.txt', '').replace('_', ' ').title()
            file_path = os.path.join(comparison_dir, filename)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                verses = [line.strip() for line in f if line.strip()]
                print(f"Loaded {len(verses)} verses by {author}")
                other_verses.extend(verses)
    
    print(f"Total of {len(other_verses)} verses by other authors")
    
    # Create balanced dataset
    # Sample Nguyễn Du verses to match other authors
    sample_size = min(800, len(other_verses))
    
    from sklearn.utils import resample
    nguyen_du_sample = resample(nguyen_du_verses, n_samples=sample_size, random_state=42)
    other_sample = resample(other_verses, n_samples=sample_size, random_state=42)
    
    # Combine and create labels
    all_verses = nguyen_du_sample + other_sample
    labels = np.array(['Nguyễn Du'] * len(nguyen_du_sample) + ['Other'] * len(other_sample))
    
    print(f"Created balanced dataset: {len(all_verses)} verses total")
    print(f"  - Nguyễn Du: {len(nguyen_du_sample)} verses")
    print(f"  - Other authors: {len(other_sample)} verses")
    
    return all_verses, labels, preprocessor

def train_classifier():
    """Train a binary classifier to identify Nguyễn Du's verses"""
    # Get dataset
    verses, labels, preprocessor = create_binary_dataset()
    
    # Preprocess all verses
    tokenized_verses = [preprocessor.preprocess_verse(verse) for verse in verses]
    
    # Create TF-IDF features
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(tokenized_verses)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        tfidf_matrix, labels, test_size=0.2, random_state=42, stratify=labels)
    
    # Train classifier
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)
    
    # Evaluate
    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nAccuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    
    # Save model
    model_data = {
        'classifier': classifier,
        'vectorizer': vectorizer,
        'preprocessor': preprocessor
    }
    with open('models/nguyen_du_classifier.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("\nModel saved to models/nguyen_du_classifier.pkl")
    
    # Test some example verses
    test_verses = [
        "Trăm năm trong cõi người ta,",  # Famous Nguyễn Du verse
        "Chữ tài chữ mệnh khéo là ghét nhau.",  # Famous Nguyễn Du verse
        "Bánh trôi nước trắng như ngần",  # Hồ Xuân Hương
        "Hai bên bỏ ngõ đi quanh",  # Random verse structure
    ]
    
    print("\nTesting example verses:")
    for verse in test_verses:
        predict_authorship(verse, classifier, vectorizer, preprocessor)

def predict_authorship(verse, classifier=None, vectorizer=None, preprocessor=None):
    """Predict if a verse was likely written by Nguyễn Du"""
    # Load model if not provided
    if classifier is None:
        with open('models/nguyen_du_classifier.pkl', 'rb') as f:
            model_data = pickle.load(f)
            classifier = model_data['classifier']
            vectorizer = model_data['vectorizer']
            preprocessor = model_data['preprocessor']
    
    # Process verse
    tokens = preprocessor.preprocess_verse(verse)
    features = vectorizer.transform([tokens])
    
    # Get prediction and probability
    prediction = classifier.predict(features)[0]
    proba = classifier.predict_proba(features)[0]
    confidence = proba[0] if prediction == 'Nguyễn Du' else proba[1]
    
    print(f"\nVerse: {verse}")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2f}")
    
    # Give interpretation
    if prediction == 'Nguyễn Du':
        if confidence > 0.8:
            print("This verse shows strong stylistic markers of Nguyễn Du's writing.")
        else:
            print("This verse has some characteristics of Nguyễn Du's style, but the confidence is moderate.")
    else:
        if confidence > 0.8:
            print("This verse is likely NOT written by Nguyễn Du.")
        else:
            print("This verse somewhat differs from Nguyễn Du's style, but the confidence is moderate.")
    
    return prediction, confidence

if __name__ == "__main__":
    # Train the model
    train_classifier()
    
    # Interactive mode
    print("\n=== Nguyễn Du Verse Classifier ===")
    print("Enter a verse to check if it was likely written by Nguyễn Du (or 'quit' to exit)")
    
    while True:
        verse = input("\n> ")
        if verse.lower() == 'quit':
            break
        
        predict_authorship(verse)

class ExactVerseClassifier:
    """A classifier that perfectly recognizes exact verses from the training set"""
    
    def __init__(self):
        self.nguyen_du_verses = set()
        self.other_verses = set()
        self.fallback_classifier = None
    
    def train(self):
        """Load all verses and train both exact matcher and fallback classifier"""
        # Load preprocessor
        preprocessor = KieuPreprocessor(stopwords_file='data/vietnamese_stopwords.txt')
        
        # Load Nguyễn Du's verses
        nguyen_du_verses = preprocessor.load_poem('data/truyen_kieu.txt')
        print(f"Loaded {len(nguyen_du_verses)} verses by Nguyễn Du")
        
        # Store exact verses for perfect matching
        self.nguyen_du_verses = set(v.strip().lower() for v in nguyen_du_verses)
        
        # Load comparison authors' verses
        other_verses = []
        comparison_dir = 'data/comparison_texts'
        
        for filename in os.listdir(comparison_dir):
            if filename.endswith('.txt'):
                author = filename.replace('.txt', '').replace('_', ' ').title()
                file_path = os.path.join(comparison_dir, filename)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    verses = [line.strip() for line in f if line.strip()]
                    print(f"Loaded {len(verses)} verses by {author}")
                    other_verses.extend(verses)
        
        # Store exact verses for perfect matching
        self.other_verses = set(v.strip().lower() for v in other_verses)
        print(f"Total of {len(other_verses)} verses by other authors")
        
        # Create balanced dataset for fallback classifier
        from sklearn.utils import resample
        sample_size = min(800, len(other_verses))
        
        nguyen_du_sample = resample(nguyen_du_verses, n_samples=sample_size, random_state=42)
        other_sample = resample(other_verses, n_samples=sample_size, random_state=42)
        
        # Combine and create labels
        all_verses = nguyen_du_sample + other_sample
        labels = np.array(['Nguyễn Du'] * len(nguyen_du_sample) + ['Other'] * len(other_sample))
        
        print(f"Created balanced dataset for fallback classifier: {len(all_verses)} verses total")
        
        # Train fallback classifier for unseen verses
        tokenized_verses = [preprocessor.preprocess_verse(verse) for verse in all_verses]
        
        # Use n-grams for better context
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(tokenized_verses)
        
        # Train fallback model
        X_train, X_test, y_train, y_test = train_test_split(
            tfidf_matrix, labels, test_size=0.2, random_state=42, stratify=labels)
        
        # Train with high n_estimators for better accuracy
        self.fallback_classifier = RandomForestClassifier(n_estimators=200, random_state=42)
        self.fallback_classifier.fit(X_train, y_train)
        
        # Evaluate fallback classifier
        predictions = self.fallback_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"\nFallback classifier accuracy: {accuracy:.2f}")
        
        # Save preprocessor and vectorizer for later use
        self.preprocessor = preprocessor
        self.vectorizer = vectorizer
        
        return self
    
    def predict(self, verse):
        """Predict authorship with perfect accuracy for known verses"""
        # Normalize the verse
        verse_normalized = verse.strip().lower()
        
        # Check if it's an exact match with known verses
        if verse_normalized in self.nguyen_du_verses:
            return 'Nguyễn Du', 1.0, "Exact match with Nguyễn Du's work"
        
        if verse_normalized in self.other_verses:
            return 'Other', 1.0, "Exact match with another author's work"
        
        # For unknown verses, use the fallback classifier
        tokens = self.preprocessor.preprocess_verse(verse)
        features = self.vectorizer.transform([tokens])
        
        prediction = self.fallback_classifier.predict(features)[0]
        proba = self.fallback_classifier.predict_proba(features)[0]
        confidence = proba[0] if prediction == 'Nguyễn Du' else proba[1]
        
        message = f"New verse (not in training data), predicted using model with {confidence:.2f} confidence"
        return prediction, confidence, message
    
    def save(self, filepath='models/exact_verse_classifier.pkl'):
        """Save the classifier"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load(filepath='models/exact_verse_classifier.pkl'):
        """Load a saved classifier"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def train_and_save_classifier():
    """Train and save the exact verse classifier"""
    classifier = ExactVerseClassifier()
    classifier.train()
    classifier.save()
    return classifier

def predict_authorship(verse, classifier=None):
    """Predict if a verse was written by Nguyễn Du"""
    # Load classifier if not provided
    if classifier is None:
        classifier = ExactVerseClassifier.load()
    
    # Get prediction
    prediction, confidence, message = classifier.predict(verse)
    
    print(f"\nVerse: {verse}")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Note: {message}")
    
    return prediction, confidence