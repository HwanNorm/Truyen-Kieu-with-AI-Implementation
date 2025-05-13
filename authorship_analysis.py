import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight

# Fix the imports to work from the main directory
from src.preprocessor import KieuPreprocessor
from src.vectorizer import TfidfVectorizer
from src.authorship import AuthorshipClassifier

def load_author_texts(base_dir='data'):
    """Load all author texts and return labeled dataset"""
    authors_data = {}
    
    # Load Nguyễn Du's Truyện Kiều
    preprocessor = KieuPreprocessor(stopwords_file=f'{base_dir}/vietnamese_stopwords.txt')
    nguyen_du_verses = preprocessor.load_poem(f'{base_dir}/truyen_kieu.txt')
    authors_data['Nguyễn Du'] = nguyen_du_verses
    
    # Load comparison authors
    comparison_dir = f'{base_dir}/comparison_texts'
    for filename in os.listdir(comparison_dir):
        if filename.endswith('.txt'):
            author_name = filename.replace('.txt', '').replace('_', ' ').title()
            file_path = os.path.join(comparison_dir, filename)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                verses = [line.strip() for line in f if line.strip()]
                
            authors_data[author_name] = verses
    
    return authors_data

def prepare_dataset(authors_data, preprocessor):
    """Prepare labeled dataset from author texts"""
    all_verses = []
    all_labels = []
    
    for author, verses in authors_data.items():
        for verse in verses:
            all_verses.append(verse)
            all_labels.append(author)
    
    # Convert to arrays
    X = np.array(all_verses)
    y = np.array(all_labels)
    
    return X, y

def main():
    # Load data
    authors_data = load_author_texts()
    
    # Print dataset stats
    print("Dataset Statistics:")
    for author, verses in authors_data.items():
        print(f"  {author}: {len(verses)} verses")
    
    # Prepare preprocessor
    preprocessor = KieuPreprocessor(stopwords_file='data/vietnamese_stopwords.txt')
    
    # Create labeled dataset
    X, y = prepare_dataset(authors_data, preprocessor)
    
    # Preprocess all verses
    tokenized_verses = [preprocessor.preprocess_verse(verse) for verse in X]
    
    # Create TF-IDF features
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(tokenized_verses)
    
    # Initialize classifier
    classifier = AuthorshipClassifier()
    
    # Extract features (TF-IDF + stylometric)
    features = classifier.extract_features(tfidf_matrix, tokenized_verses)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train model
    classifier.train(X_train, y_train)
    
    # Evaluate
    predictions = classifier.classifier.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(y),
                yticklabels=np.unique(y))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('authorship_confusion_matrix.png')
    
    # Example prediction function
    def predict_authorship(verse):
        tokens = preprocessor.preprocess_verse(verse)
        # Create dummy matrix to extract features
        dummy_matrix = vectorizer.transform([tokens])
        features = classifier.extract_features(dummy_matrix, [tokens])
        
        # Get prediction and confidence
        pred = classifier.classifier.predict(features)[0]
        proba = classifier.classifier.predict_proba(features)[0]
        confidence = proba.max()
        
        return pred, confidence
    
    # Example usage
    print("\nExample Predictions:")
    test_verses = [
        "Trăm năm trong cõi người ta,",
        "Chớ nên giận thiếp thiếp đà biết điều.",
        "Người về chiếc bóng năm canh,"
    ]
    
    for verse in test_verses:
        author, confidence = predict_authorship(verse)
        print(f"Verse: {verse}")
        print(f"Predicted author: {author} (confidence: {confidence:.4f})")
        print()
    
    # Save model
    classifier.save('models/authorship_classifier.pkl')
    print("Model saved to models/authorship_classifier.pkl")

if __name__ == "__main__":
    main()