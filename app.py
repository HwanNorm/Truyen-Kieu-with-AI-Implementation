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
from src.ollama_kieu_generator import OllamaKieuGenerator

def main():
    # Check dependencies first
    dependencies_ready = check_and_install_dependencies()
    if not dependencies_ready:
        print("Warning: Running with missing dependencies. Some features may not work.")
    
    # Load configuration
    config = load_config()
        
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
                                 'generate-verse', 'generate-image', 'image-to-verse','ollama-verse'],
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
    parser.add_argument('--enhanced', action='store_true',
                       help='Use enhanced image-to-verse matching with cultural context')
    parser.add_argument('--clip-model', type=str, default=None,
                       help='CLIP model to use (defaults to config.json setting)')
    parser.add_argument('--no-color-analysis', action='store_true',
                       help='Disable color-based image analysis')
    parser.add_argument('--feedback', action='store_true',
                       help='Enable user feedback collection for image-to-verse')
    
    # LSTM parameters
    parser.add_argument('--cpu-lstm', action='store_true',
                       help='Train an LSTM model optimized for CPU usage')
    parser.add_argument('--early-stopping', action='store_true',
                       help='Train LSTM with early stopping to prevent overfitting')
    parser.add_argument('--post-process', action='store_true',
                       help='Apply Vietnamese poetic post-processing to generated verses')
    
    # Additional LSTM hyperparameters
    parser.add_argument('--embedding-dim', type=int, default=256,
                       help='Embedding dimension for LSTM model')
    parser.add_argument('--hidden-dim', type=int, default=512,
                       help='Hidden dimension for LSTM model')
    parser.add_argument('--num-layers', type=int, default=2,
                       help='Number of LSTM layers')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Maximum number of training epochs')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate for training')
    
    #Ollama parameters
    parser.add_argument('--ollama-model', type=str, default='llama3',
                        help='Ollama model name to use (default: llama3)')
    parser.add_argument('--save', action='store_true',
                   help='Save generated verses to file')
    
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
        run_image_to_verse(args, verses, config)
    elif args.mode == 'ollama-verse':
        print(f"Generating verses using Ollama {args.ollama_model}")
        
        try:
            # Initialize generator
            model_name = args.ollama_model
            if ":" not in model_name:
                model_name = f"{model_name}:latest"

            generator = OllamaKieuGenerator(
                model_name=model_name,
                truyen_kieu_path=args.data
            )
            
            # Generate verses
            if args.prompt:
                print(f"Generating {args.num_samples} verses starting with '{args.prompt}'...")
                
                verses = generator.generate_verse(
                    initial_phrase=args.prompt,
                    num_samples=args.num_samples,
                    max_length=args.max_length
                )
                
                for i, verse in enumerate(verses, 1):
                    print(f"\n=== Generated Verse {i} ===")
                    print(verse)
                    
                    # Save if requested
                    if args.save:
                        os.makedirs(args.output_dir, exist_ok=True)
                        output_file = os.path.join(args.output_dir, f"ollama_verse_{i}.txt")
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(verse)
                        print(f"Verse saved to {output_file}")
            else:
                # Interactive mode
                print("\n=== Ollama Truyện Kiều Verse Generator ===")
                print("Enter a starting phrase (or 'quit' to exit):")
                
                while True:
                    prompt = input("> ")
                    if prompt.lower() == 'quit':
                        break
                    
                    verses = generator.generate_verse(
                        initial_phrase=prompt,
                        num_samples=1
                    )
                    
                    if verses:
                        print("\nGenerated verse:")
                        print(verses[0])
                    else:
                        print("Failed to generate verse. Try a different prompt.")
                    print()
                    
        except Exception as e:
            print(f"Error with Ollama generator: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Mode {args.mode} not implemented yet.")

def load_config():
    """Load custom configuration settings"""
    import json
    import os
    
    config_file = "config.json"
    default_config = {
        "image_to_verse": {
            "clip_model": "openai/clip-vit-base-patch32",
            "top_k": 5,
            "boost_factors": {
                "hoa": 1.5,
                "trăng": 1.3,
                "mây": 1.2
            },
            "feedback_file": "feedback_data.json"
        }
    }
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config file: {e}")
            print("Using default configuration.")
            return default_config
    else:
        # Create default config file
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, ensure_ascii=False, indent=2)
        return default_config
    
def record_feedback(image_path, verse, rating, notes=None):
    """Record user feedback for image-to-verse matching"""
    import datetime
    import json
    import os
    
    # Load config to get feedback file path
    config = load_config()
    feedback_file = config["image_to_verse"]["feedback_file"]
    
    feedback_entry = {
        "image": image_path,
        "verse": verse,
        "rating": rating,
        "notes": notes,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # Load existing feedback or create new file
    if os.path.exists(feedback_file):
        try:
            with open(feedback_file, 'r', encoding='utf-8') as f:
                feedback_data = json.load(f)
        except:
            feedback_data = []
    else:
        feedback_data = []
    
    # Add new entry
    feedback_data.append(feedback_entry)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(feedback_file) or '.', exist_ok=True)
    
    # Save updated data
    with open(feedback_file, 'w', encoding='utf-8') as f:
        json.dump(feedback_data, f, ensure_ascii=False, indent=2)
    
    return "Feedback recorded. Thank you for helping improve the system!"

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
    # If CPU-optimized LSTM is requested
    if args.cpu_lstm:
        args.model_type = 'lstm'  # Ensure model type is lstm
        if args.early_stopping:
            return run_cpu_optimized_lstm_with_early_stopping(args, verses)
        else:
            return run_cpu_optimized_lstm(args, verses)
    
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
    model_path = args.language_model
    
    # If no model path is specified, use the default matching the model type
    if model_path is None or "ngram" in model_path and args.model_type == "lstm":
        if args.model_type == "lstm":
            model_path = "models/kieu_lstm.pkl"
        else:
            model_path = "models/kieu_ngram.pkl"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        print("Please train a model first using --mode train-language-model")
        return
    
    try:
        # Load the verse generator
        print(f"Loading {args.model_type} model from {model_path}")
        generator = KieuVerseGenerator(model_path, args.model_type)
        
        # Generate verses
        if args.prompt:
            print(f"Generating verses from: '{args.prompt}'")
            
            # Apply post-processing if requested
            if args.post_process:
                # Use language model directly for post-processing
                from src.language_model import KieuLanguageModelTrainer
                
                trainer = KieuLanguageModelTrainer.load(model_path, args.model_type)
                verses = []
                
                for _ in range(args.num_samples):
                    processed_verse = trainer.generate_with_postprocessing(args.prompt, max_length=args.max_length)
                    verses.append(processed_verse)
            else:
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
                
                if args.post_process:
                    # Use language model directly for post-processing
                    from src.language_model import KieuLanguageModelTrainer
                    
                    trainer = KieuLanguageModelTrainer.load(model_path, args.model_type)
                    verses = []
                    
                    for _ in range(args.num_samples):
                        processed_verse = trainer.generate_with_postprocessing(prompt, max_length=args.max_length)
                        verses.append(processed_verse)
                else:
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

def run_image_to_verse(args, verses, config):
    """Find matching verses for an input image using enhanced matching"""
    try:
        # Get CLIP model from args or config
        clip_model = args.clip_model or config["image_to_verse"]["clip_model"]
        
        # Initialize the verse retriever with the specified model
        print(f"Initializing image-to-verse with CLIP model: {clip_model}")
        retriever = ImageToVerseRetriever(model_name=clip_model)
        
        # Check if we have a saved index
        index_dir = os.path.dirname(args.clip_index)
        if index_dir:
            os.makedirs(index_dir, exist_ok=True)
            
        # Check if the index exists
        index_exists = os.path.exists(args.clip_index)
        
        # If the index exists, check if we should regenerate it
        if index_exists:
            try:
                # Try to load the index to check if it's compatible
                retriever.load_index(args.clip_index)
                print(f"Successfully loaded verse index from {args.clip_index}")
            except RuntimeError as e:
                # If there's a shape mismatch error, the index was created with a different model
                print(f"Index was created with a different model. Regenerating...")
                index_exists = False
                # Remove the old index
                os.remove(args.clip_index)
        
        # Create a new index if needed
        if not index_exists:
            print("Creating new verse index (this may take a while)...")
            retriever.index_verses(verses)
            
            # Save index for future use
            print(f"Saving verse index to {args.clip_index}")
            retriever.save_index(args.clip_index)
        
        # Import the cultural context enhancer
        context_enhancer = CulturalContextEnhancer()
        
        # Process the input image
        if args.image:
            if not os.path.exists(args.image):
                print(f"Error: Image file {args.image} not found.")
                return
                
            print(f"Finding verses that match image: {args.image}")
            image = Image.open(args.image).convert('RGB')
            
            # Use standard or enhanced matching based on args
            if args.enhanced:
                print("Using enhanced matching with cultural context...")
                # Get image content analysis if not disabled
                if not args.no_color_analysis:
                    print("Analyzing image content...")
                    image_elements = retriever.analyze_image_content(image)
                    print(f"Detected elements: {', '.join(image_elements.keys())}")
                
                # Find matching verses with enhanced approach
                results = retriever.find_matching_verses_enhanced(image, top_k=args.top_k)
            else:
                # Use standard matching
                results = retriever.find_matching_verses(image, top_k=args.top_k)
            
            # Display results
            print("\nTop matching verses:")
            for i, result in enumerate(results, 1):
                boost = result.get('boost_factor', 1.0)
                boost_str = f" (boost: {boost:.2f})" if boost > 1.0 else ""
                print(f"{i}. [{result['score']:.4f}]{boost_str} {result['verse']}")
                
            # Get explanation for top match
            if results:
                if args.enhanced:
                    explanation = retriever.explain_match_enhanced(image, results[0]['index'])
                    print("\nMatch explanation:")
                    print(f"Key imagery: {', '.join(explanation['key_imagery'])}")
                    
                    if 'themes' in explanation:
                        print(f"Themes: {', '.join(explanation['themes'])}")
                    
                    if 'detected_elements' in explanation and not args.no_color_analysis:
                        print(f"Detected in image: {', '.join(explanation['detected_elements'])}")
                else:
                    explanation = retriever.explain_match(image, results[0]['index'])
                    print("\nMatch explanation:")
                    print(f"Key imagery: {', '.join(explanation['key_imagery'])}")
                
                # Request feedback if enabled
                if args.feedback:
                    print("\nWas this a good match? (1-5, where 5 is excellent):")
                    rating = input("> ")
                    try:
                        rating = int(rating)
                        if 1 <= rating <= 5:
                            print("Any additional notes or suggestions?")
                            notes = input("> ")
                            record_feedback(args.image, results[0]['verse'], rating, notes)
                            print("Feedback recorded. Thank you!")
                    except ValueError:
                        print("Invalid rating. Feedback not recorded.")
        else:
            # Interactive mode
            if args.enhanced:
                print("\n=== Truyện Kiều Image-to-Verse Retriever (Enhanced) ===")
            else:
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
                    
                    if args.enhanced:
                        # Enhanced matching
                        if not args.no_color_analysis:
                            print("Analyzing image content...")
                            image_elements = retriever.analyze_image_content(image)
                            print(f"Detected elements: {', '.join(image_elements.keys())}")
                        
                        results = retriever.find_matching_verses_enhanced(image, top_k=args.top_k)
                    else:
                        # Standard matching
                        results = retriever.find_matching_verses(image, top_k=args.top_k)
                    
                    print("\nTop matching verses:")
                    for i, result in enumerate(results, 1):
                        boost = result.get('boost_factor', 1.0)
                        boost_str = f" (boost: {boost:.2f})" if boost > 1.0 else ""
                        print(f"{i}. [{result['score']:.4f}]{boost_str} {result['verse']}")
                        
                    # Get explanation for top match
                    if results:
                        if args.enhanced:
                            explanation = retriever.explain_match_enhanced(image, results[0]['index'])
                            print("\nMatch explanation:")
                            print(f"Key imagery: {', '.join(explanation['key_imagery'])}")
                            
                            if 'themes' in explanation:
                                print(f"Themes: {', '.join(explanation['themes'])}")
                            
                            if 'detected_elements' in explanation and not args.no_color_analysis:
                                print(f"Detected in image: {', '.join(explanation['detected_elements'])}")
                        else:
                            explanation = retriever.explain_match(image, results[0]['index'])
                            print("\nMatch explanation:")
                            print(f"Key imagery: {', '.join(explanation['key_imagery'])}")
                        
                        # Request feedback if enabled
                        if args.feedback:
                            print("\nWas this a good match? (1-5, where 5 is excellent):")
                            rating = input("> ")
                            try:
                                rating = int(rating)
                                if 1 <= rating <= 5:
                                    print("Any additional notes or suggestions?")
                                    notes = input("> ")
                                    record_feedback(image_path, results[0]['verse'], rating, notes)
                                    print("Feedback recorded. Thank you!")
                            except ValueError:
                                print("Invalid rating. Feedback not recorded.")
                except Exception as e:
                    print(f"Error processing image: {e}")
                
                print()
    
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install the required packages: pip install transformers torch scikit-learn")

def run_cpu_optimized_lstm(args, verses):
    """Train an LSTM model optimized for CPU usage"""
    from src.language_model import KieuLanguageModelTrainer
    import torch
    
    print("Training CPU-optimized LSTM model...")
    print("Note: This may take 30-60 minutes on a laptop without GPU")
    
    # Get hyperparameters from args
    embedding_dim = args.embedding_dim
    hidden_dim = args.hidden_dim
    num_layers = args.num_layers
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    
    # CPU-optimized hyperparameters
    trainer = KieuLanguageModelTrainer(
        model_type='lstm',
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.2
    )
    
    print(f"Using CPU-optimized parameters:")
    print(f"  - Embedding dimension: {embedding_dim}")
    print(f"  - Hidden dimension: {hidden_dim}")
    print(f"  - LSTM layers: {num_layers}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Epochs: {epochs}")
    
    # Train with progress reporting
    try:
        from tqdm import tqdm
        
        # Prepare data first to show progress
        print("Preparing training data...")
        trainer.prepare_data(verses)
        
        # Then start training with epochs progress
        print("Starting training...")
        for epoch in tqdm(range(epochs), desc="Epochs"):
            trainer.train_epoch(verses, batch_size=batch_size, learning_rate=learning_rate)
            print(f"  Completed epoch {epoch+1}/{epochs}")
    except ImportError:
        # If tqdm not available, train normally
        trainer.train(verses, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
    
    # Save model
    model_path = args.language_model
    trainer.save(model_path)
    print(f"LSTM model saved to {model_path}")
    
    # Test generation
    print("\nTesting LSTM model with sample generation:")
    prompts = ["Trăm năm", "Tình duyên", "Hồng nhan"]
    for prompt in prompts:
        generated = trainer.generate(prompt, max_length=40)
        print(f"Prompt: '{prompt}'")
        print(f"Generated: '{generated}'\n")
    
    return model_path

def run_cpu_optimized_lstm_with_early_stopping(args, verses):
    """Train an LSTM model with early stopping for better results"""
    from src.language_model import KieuLanguageModelTrainer
    import torch
    
    print("Training CPU-optimized LSTM model with early stopping...")
    print("Note: This may take several hours on a laptop without GPU")
    
    # Get hyperparameters from args
    embedding_dim = args.embedding_dim
    hidden_dim = args.hidden_dim
    num_layers = args.num_layers
    batch_size = args.batch_size
    epochs = args.epochs
    patience = args.patience
    learning_rate = args.learning_rate
    
    # CPU-optimized hyperparameters
    trainer = KieuLanguageModelTrainer(
        model_type='lstm',
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.2
    )
    
    print(f"Using CPU-optimized parameters with early stopping:")
    print(f"  - Embedding dimension: {embedding_dim}")
    print(f"  - Hidden dimension: {hidden_dim}")
    print(f"  - LSTM layers: {num_layers}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Max epochs: {epochs}")
    print(f"  - Early stopping patience: {patience}")
    
    # Train with progress reporting
    try:
        from tqdm import tqdm
        
        # Prepare data first to show progress
        print("Preparing training data...")
        trainer.prepare_data(verses)
        
        # Train with early stopping
        print("Starting training with early stopping...")
        best_model = trainer.train_with_early_stopping(
            verses, 
            epochs=epochs, 
            batch_size=batch_size, 
            learning_rate=learning_rate,
            patience=patience
        )
        
        if best_model:
            # Save the best model
            torch.save(best_model, args.language_model)
            print(f"Best model saved to {args.language_model}")
        
    except ImportError:
        print("tqdm package not available, training without progress bar")
        best_model = trainer.train_with_early_stopping(
            verses, 
            epochs=epochs, 
            batch_size=batch_size, 
            learning_rate=learning_rate,
            patience=patience
        )
        if best_model:
            torch.save(best_model, args.language_model)
    
    # Test generation with post-processing
    print("\nTesting LSTM model with sample generation and post-processing:")
    prompts = ["Trăm năm", "Tình duyên", "Hồng nhan"]
    for prompt in prompts:
        generated = trainer.generate_with_postprocessing(prompt, max_length=50)
        print(f"Prompt: '{prompt}'")
        print(f"Generated: '{generated}'\n")
    
    return args.language_model


def check_and_install_dependencies():
    """Check for required packages and install them if needed"""
    # Map between pip package names and their import names
    packages = {
        "transformers": "transformers",
        "torch": "torch",
        "scikit-learn": "sklearn",  # Import name differs from pip name
        "numpy": "numpy",
        "pillow": "PIL"  # Import name differs from pip name
    }
    
    missing_packages = []
    
    for package_name, import_name in packages.items():
        try:
            __import__(import_name)
            print(f"✓ {package_name} is installed")
        except ImportError:
            missing_packages.append(package_name)
            print(f"✗ {package_name} is missing")
    
    if missing_packages:
        print("\nMissing packages detected. Do you want to install them? (y/n)")
        answer = input("> ").lower()
        if answer == 'y':
            import subprocess
            import sys
            
            print("Installing missing packages...")
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            print("Packages installed successfully!")
            return True
        else:
            print("Warning: Some features might not work without the required packages.")
            return False
    
    return True

if __name__ == "__main__":
    main()