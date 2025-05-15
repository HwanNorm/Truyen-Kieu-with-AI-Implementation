import argparse
import os
from PIL import Image

from src.preprocessor import KieuPreprocessor
from src.vectorizer import TfidfVectorizer
from src.search_engine import KieuSearchEngine
from src.language_model import KieuLanguageModelTrainer
from src.verse_generator import KieuVerseGenerator
from src.image_generator import VerseToImageGenerator
from src.multimodal_retriever import ImageToVerseRetriever
from src.cultural_context import CulturalContextEnhancer

def main():
    parser = argparse.ArgumentParser(description='Truyện Kiều Analysis and Generation Toolkit')
    
    # Basic parameters
    parser.add_argument('--data', type=str, default='data/truyen_kieu.txt',
                        help='Path to Truyện Kiều text file')
    parser.add_argument('--stopwords', type=str, default=None,
                        help='Path to stopwords file')
    parser.add_argument('--output-dir', type=str, default='output/',
                        help='Directory for output files')
    
    # Mode selection
    parser.add_argument('--mode', type=str, 
                        choices=['search', 'train-search', 'train-authorship', 'train-language-model',
                                 'generate-verse', 'generate-image', 'image-to-verse'],
                        default='search',
                        help='Mode to operate in')
    
    # Search mode parameters
    parser.add_argument('--tfidf-model', type=str, default='models/tfidf_model.pkl',
                        help='Path to load/save TF-IDF model')
    parser.add_argument('--query', type=str, default=None,
                        help='Search query (for search mode)')
    
    # Language model parameters
    parser.add_argument('--language-model', type=str, default='models/kieu_ngram.pkl',
                        help='Path to load/save language model')
    parser.add_argument('--model-type', type=str, choices=['ngram', 'lstm', 'transformer'],
                        default='ngram', help='Type of language model')
    parser.add_argument('--prompt', type=str, default='Trăm năm',
                        help='Prompt for verse generation')
    parser.add_argument('--num-samples', type=int, default=3,
                        help='Number of samples to generate')
    parser.add_argument('--max-length', type=int, default=50,
                        help='Maximum length for generation')
    
    # Image generation parameters
    parser.add_argument('--verse', type=str, default=None,
                        help='Verse for image generation')
    parser.add_argument('--style', type=str, 
                        choices=['traditional', 'modern', 'fantasy', 'ukiyo-e', 'impressionist'],
                        default='traditional', help='Artistic style for image generation')
    parser.add_argument('--image-output', type=str, default='output/generated_image.jpg',
                        help='Path to save generated image')
    
    # Image to verse parameters
    parser.add_argument('--image', type=str, default=None,
                        help='Path to input image for verse retrieval')
    parser.add_argument('--clip-index', type=str, default='models/clip_index/kieu_verses.pt',
                        help='Path to save/load CLIP index')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Number of top results to return')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = KieuPreprocessor(stopwords_file=args.stopwords)
    
    # Load verses
    verses = preprocessor.load_poem(args.data)
    print(f"Loaded {len(verses)} verses from {args.data}")
    
    # Execute the requested mode
    if args.mode == 'search' or args.mode == 'train-search':
        run_search_mode(args, preprocessor, verses)
    elif args.mode == 'train-language-model':
        run_train_language_model(args, verses)
    elif args.mode == 'generate-verse':
        run_generate_verse(args)
    elif args.mode == 'generate-image':
        run_generate_image(args)
    elif args.mode == 'image-to-verse':
        run_image_to_verse(args, verses)
    else:
        print(f"Mode {args.mode} not implemented yet.")

def run_search_mode(args, preprocessor, verses):
    """Run the search engine mode"""
    # Check if we need to train a new model
    if args.mode == 'train-search' or not os.path.exists(args.tfidf_model):
        print("Training TF-IDF model...")
        tokenized_verses = preprocessor.preprocess_all_verses(verses)
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(tokenized_verses)
        vectorizer.save(args.tfidf_model)
        print(f"Model trained and saved to {args.tfidf_model}")
    else:
        # Load existing model
        print(f"Loading TF-IDF model from {args.tfidf_model}")
        vectorizer = TfidfVectorizer.load(args.tfidf_model)
        tokenized_verses = preprocessor.preprocess_all_verses(verses)
        tfidf_matrix = vectorizer.transform(tokenized_verses)
    
    # Initialize search engine
    search_engine = KieuSearchEngine(preprocessor, vectorizer, verses, tfidf_matrix)
    
    # If query provided, run search
    if args.query:
        results = search_engine.hybrid_search(args.query, top_k=args.top_k)
        
        if not results:
            print("No matching verses found.")
        else:
            print("\nTop matching verses:")
            for i, (idx, verse, score) in enumerate(results, 1):
                print(f"{i}. [Line {idx+1}] {verse} (score: {score:.4f})")
    else:
        # Interactive mode
        print("\n=== Truyện Kiều Search Engine ===")
        print("Enter your query (or 'quit' to exit):")
        
        while True:
            query = input("> ")
            if query.lower() == 'quit':
                break
            
            # Regular search
            results = search_engine.hybrid_search(query, top_k=args.top_k)
            
            if not results:
                print("No matching verses found.")
            else:
                print("\nTop matching verses:")
                for i, (idx, verse, score) in enumerate(results, 1):
                    print(f"{i}. [Line {idx+1}] {verse} (score: {score:.4f})")
            print()

def run_train_language_model(args, verses):
    """Train a language model for verse generation"""
    print(f"Training {args.model_type.upper()} language model...")
    
    # Initialize trainer
    trainer = KieuLanguageModelTrainer(model_type=args.model_type)
    
    # Train the model
    if args.model_type == 'ngram':
        trainer.train(verses)
    elif args.model_type == 'lstm':
        # LSTM training takes longer and has more parameters
        print("Training LSTM model (this may take a while)...")
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Use smaller parameters for CPU training
        if device == 'cpu':
            print("Warning: Training on CPU may be slow. Consider using smaller model parameters.")
            trainer.train(verses, epochs=3, batch_size=16, learning_rate=0.001)
        else:
            trainer.train(verses, epochs=5, batch_size=32, learning_rate=0.001)
    else:  # transformer
        print("Transformer model training not implemented in CLI mode. Please use the notebook.")
        return
    
    # Save the model
    os.makedirs(os.path.dirname(args.language_model) or '.', exist_ok=True)
    trainer.save(args.language_model)
    print(f"Model saved to {args.language_model}")
    
    # Test the model
    print("\nTesting model with sample generation:")
    test_prompt = "Trăm năm"
    generated = trainer.generate(test_prompt, max_length=args.max_length)
    print(f"Prompt: '{test_prompt}'")
    print(f"Generated: '{generated}'")

def run_generate_verse(args):
    """Generate verses using a trained language model"""
    # Check if model exists
    if not os.path.exists(args.language_model):
        print(f"Error: Model file {args.language_model} not found.")
        print("Please train a model first using --mode train-language-model")
        return
    
    try:
        # Load the verse generator
        print(f"Loading {args.model_type} model from {args.language_model}")
        generator = KieuVerseGenerator(args.language_model, args.model_type)
        
        # Generate verses
        if args.prompt:
            print(f"Generating verses from: '{args.prompt}'")
            verses = generator.generate_verse(args.prompt, 
                                             max_length=args.max_length, 
                                             num_samples=args.num_samples)
            
            if verses:
                print("\nGenerated verses:")
                for i, verse in enumerate(verses, 1):
                    print(f"{i}. {verse}")
                    
                # Evaluate the first verse
                if verses:
                    quality = generator.evaluate_verse_quality(verses[0])
                    print("\nQuality metrics for first verse:")
                    for metric, score in quality.items():
                        print(f"- {metric}: {score:.2f}")
            else:
                print("Failed to generate verses. Try a different prompt or model.")
        else:
            # Interactive mode
            print("\n=== Truyện Kiều Verse Generator ===")
            print("Enter a starting phrase (or 'quit' to exit):")
            
            while True:
                prompt = input("> ")
                if prompt.lower() == 'quit':
                    break
                
                verses = generator.generate_verse(prompt, 
                                                max_length=args.max_length, 
                                                num_samples=args.num_samples)
                
                if verses:
                    print("\nGenerated verses:")
                    for i, verse in enumerate(verses, 1):
                        print(f"{i}. {verse}")
                else:
                    print("Failed to generate verses. Try a different prompt.")
                print()
                
    except Exception as e:
        print(f"Error: {e}")

def run_generate_image(args):
    """Generate an image from a verse"""
    try:
        # Initialize the image generator
        generator = VerseToImageGenerator()
        
        # Get verse to visualize
        verse = args.verse
        if not verse:
            # Interactive mode
            print("\n=== Truyện Kiều Image Generator ===")
            verse = input("Enter a verse to visualize: ")
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(args.image_output) or '.', exist_ok=True)
        
        # Generate and save the image
        print(f"Generating image for verse: {verse}")
        print(f"Using style: {args.style}")
        generator.generate_and_save(verse, args.image_output, args.style)
        
        print(f"Image saved to {args.image_output}")
        
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install the required packages: pip install diffusers transformers torch")

def run_image_to_verse(args, verses):
    """Find matching verses for an input image"""
    try:
        # Initialize the verse retriever
        retriever = ImageToVerseRetriever()
        
        # Check if we have a saved index
        index_dir = os.path.dirname(args.clip_index)
        if index_dir:
            os.makedirs(index_dir, exist_ok=True)
            
        if os.path.exists(args.clip_index):
            # Load existing index
            print(f"Loading verse index from {args.clip_index}")
            retriever.load_index(args.clip_index)
        else:
            # Create new index
            print("Creating new verse index (this may take a while)...")
            retriever.index_verses(verses)
            
            # Save index for future use
            print(f"Saving verse index to {args.clip_index}")
            retriever.save_index(args.clip_index)
        
        # Process the input image
        if args.image:
            if not os.path.exists(args.image):
                print(f"Error: Image file {args.image} not found.")
                return
                
            print(f"Finding verses that match image: {args.image}")
            image = Image.open(args.image).convert('RGB')
            
            # Find matching verses
            results = retriever.find_matching_verses(image, top_k=args.top_k)
            
            # Display results
            print("\nTop matching verses:")
            for i, result in enumerate(results, 1):
                print(f"{i}. [{result['score']:.4f}] {result['verse']}")
                
            # Get explanation for top match
            if results:
                explanation = retriever.explain_match(image, results[0]['index'])
                print("\nMatch explanation:")
                print(f"Key imagery: {', '.join(explanation['key_imagery'])}")
        else:
            # Interactive mode
            print("\n=== Truyện Kiều Image-to-Verse Retriever ===")
            print("Enter the path to an image file (or 'quit' to exit):")
            
            while True:
                image_path = input("> ")
                if image_path.lower() == 'quit':
                    break
                
                if not os.path.exists(image_path):
                    print(f"Error: Image file {image_path} not found.")
                    continue
                
                try:
                    image = Image.open(image_path).convert('RGB')
                    results = retriever.find_matching_verses(image, top_k=args.top_k)
                    
                    print("\nTop matching verses:")
                    for i, result in enumerate(results, 1):
                        print(f"{i}. [{result['score']:.4f}] {result['verse']}")
                        
                    # Get explanation for top match
                    if results:
                        explanation = retriever.explain_match(image, results[0]['index'])
                        print("\nMatch explanation:")
                        print(f"Key imagery: {', '.join(explanation['key_imagery'])}")
                    
                except Exception as e:
                    print(f"Error processing image: {e}")
                
                print()
                
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install the required packages: pip install transformers torch")

if __name__ == "__main__":
    main()