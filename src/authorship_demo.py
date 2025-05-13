import pickle
from src.preprocessor import KieuPreprocessor
from src.vectorizer import TfidfVectorizer

def load_model(model_path='models/authorship_classifier.pkl'):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def main():
    # Load model
    classifier = load_model()
    
    # Initialize preprocessor
    preprocessor = KieuPreprocessor(stopwords_file='data/vietnamese_stopwords.txt')
    
    # Load vectorizer
    vectorizer = TfidfVectorizer.load('models/tfidf_model.pkl')
    
    print("=== Truyện Kiều Authorship Attribution ===")
    print("Enter a verse to check its likely author (or 'quit' to exit)")
    
    while True:
        verse = input("\n> ")
        if verse.lower() == 'quit':
            break
        
        # Preprocess
        tokens = preprocessor.preprocess_verse(verse)
        
        # Create dummy matrix for feature extraction
        dummy_matrix = vectorizer.transform([tokens])
        
        # Extract features
        features = classifier.extract_features(dummy_matrix, [tokens])
        
        # Predict
        author = classifier.classifier.predict(features)[0]
        proba = classifier.classifier.predict_proba(features)[0]
        confidence = proba.max()
        
        print(f"This verse was most likely written by: {author}")
        print(f"Confidence: {confidence:.2%}")
        
        # If it's a high-confidence prediction for Nguyễn Du, add extra detail
        if author == "Nguyễn Du" and confidence > 0.8:
            print("This shows strong stylistic markers of Nguyễn Du's writing in Truyện Kiều.")
        elif confidence < 0.6:
            print("Note: This prediction has relatively low confidence.")

if __name__ == "__main__":
    main()