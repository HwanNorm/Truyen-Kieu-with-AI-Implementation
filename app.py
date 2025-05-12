import argparse
from src.preprocessor import KieuPreprocessor
from src.vectorizer import TfidfVectorizer
from src.search_engine import KieuSearchEngine

def main():
    parser = argparse.ArgumentParser(description='Truyện Kiều Search Engine')
    parser.add_argument('--data', type=str, default='data/truyen_kieu.txt',
                        help='Path to Truyện Kiều text file')
    parser.add_argument('--stopwords', type=str, default=None,
                        help='Path to stopwords file')
    parser.add_argument('--model', type=str, default='models/tfidf_model.pkl',
                        help='Path to save/load the TF-IDF model')
    parser.add_argument('--mode', type=str, choices=['train', 'search'], default='search',
                        help='Mode: train the model or search')
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = KieuPreprocessor(stopwords_file=args.stopwords)
    
    if args.mode == 'train':
        print("Loading and preprocessing Truyện Kiều...")
        verses = preprocessor.load_poem(args.data)
        tokenized_verses = preprocessor.preprocess_all_verses(verses)
        
        print("Building TF-IDF model...")
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(tokenized_verses)
        
        print(f"Saving model to {args.model}")
        vectorizer.save(args.model)
        
        print("Model trained successfully!")
        print(f"Vocabulary size: {len(vectorizer.vocabulary)}")
        print(f"Number of verses: {len(verses)}")
    
    elif args.mode == 'search':
        print("Loading Truyện Kiều...")
        verses = preprocessor.load_poem(args.data)
        
        print(f"Loading TF-IDF model from {args.model}")
        vectorizer = TfidfVectorizer.load(args.model)
        
        print("Preprocessing verses...")
        tokenized_verses = preprocessor.preprocess_all_verses(verses)
        tfidf_matrix = vectorizer.transform(tokenized_verses)
        
        search_engine = KieuSearchEngine(preprocessor, vectorizer, verses, tfidf_matrix)
        
        print("\n=== Truyện Kiều Search Engine ===")
        print("Enter your query (or 'quit' to exit):")
        
        while True:
            query = input("> ")
            if query.lower() == 'quit':
                break
            
            # Add debug functionality
            if query.lower().startswith("debug:"):
                text = query[6:].strip()
                search_engine.debug_text_presence(text)
                continue
            
            # Regular search
            results = search_engine.hybrid_search(query, top_k=16)
            
            if not results:
                print("No matching verses found.")
            else:
                print("\nTop matching verses:")
                for i, (idx, verse, score) in enumerate(results, 1):
                    print(f"{i}. [Line {idx+1}] {verse} (score: {score:.4f})")
            print()

if __name__ == "__main__":
    main()
