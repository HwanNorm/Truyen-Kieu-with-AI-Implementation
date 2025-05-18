import os
import subprocess
import sys
from PIL import Image

# Add current directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessor import KieuPreprocessor
from src.vectorizer import TfidfVectorizer
from src.search_engine import KieuSearchEngine
from src.image_generator import VerseToImageGenerator
from src.multimodal_retriever import ImageToVerseRetriever
from src.ollama_kieu_generator import OllamaKieuGenerator

def clear_screen():
    """Clear the console screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print the application header"""
    print("\n" + "=" * 80)
    print(" " * 25 + "TRUYỆN KIỀU ANALYSIS TOOLKIT")
    print("=" * 80)
    
    print("\nThis toolkit provides various functions for analyzing Truyện Kiều:")
    print("1. Vector Space Model and Search Engine")
    print("2. Authorship Attribution")
    print("3. Language Modeling and Verse Generation")
    print("4. Image Generation from Verses")
    print("5. Image-to-Verse Retrieval")
    print("0. Exit")
    print("=" * 80)

def search_engine():
    """Run the search engine for Truyện Kiều"""
    clear_screen()
    print("\n" + "=" * 80)
    print(" " * 25 + "SEARCH ENGINE")
    print("=" * 80)
    
    # Paths
    data_path = 'data/truyen_kieu.txt'
    stopwords_path = 'data/vietnamese_stopwords.txt'
    tfidf_model_path = 'models/tfidf_model.pkl'
    
    # Check if model exists
    model_exists = os.path.exists(tfidf_model_path)
    
    # Ask if user wants to train a new model
    if not model_exists:
        print("\nTF-IDF model not found. A new model will be trained.")
        train_model = True
    else:
        print(f"\nTF-IDF model found at {tfidf_model_path}.")
        train_choice = input("Do you want to use the existing model (E) or train a new one (T)? [E/T]: ").strip().upper()
        train_model = train_choice == 'T'
    
    # Initialize preprocessor
    print("\nInitializing preprocessor...")
    preprocessor = KieuPreprocessor(stopwords_file=stopwords_path)
    
    # Load verses
    print(f"Loading verses from {data_path}...")
    verses = preprocessor.load_poem(data_path)
    print(f"Loaded {len(verses)} verses.")
    
    # Train or load model
    if train_model:
        print("\nTraining TF-IDF model...")
        tokenized_verses = preprocessor.preprocess_all_verses(verses)
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(tokenized_verses)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(tfidf_model_path), exist_ok=True)
        vectorizer.save(tfidf_model_path)
        print(f"Model trained and saved to {tfidf_model_path}")
    else:
        print(f"\nLoading existing TF-IDF model from {tfidf_model_path}...")
        vectorizer = TfidfVectorizer.load(tfidf_model_path)
        tokenized_verses = preprocessor.preprocess_all_verses(verses)
        tfidf_matrix = vectorizer.transform(tokenized_verses)
    
    # Initialize search engine
    search_engine = KieuSearchEngine(preprocessor, vectorizer, verses, tfidf_matrix)
    
    # Interactive search
    print("\n" + "=" * 80)
    print(" " * 25 + "INTERACTIVE SEARCH")
    print("=" * 80)
    print("Enter your search query to find relevant verses from Truyện Kiều.")
    print("Type 'quit' to return to the main menu.")
    print("=" * 80)
    
    while True:
        query = input("\nSearch query: ").strip()
        if query.lower() == 'quit':
            break
        
        results = search_engine.hybrid_search(query, top_k=5)
        
        if not results:
            print("No matching verses found.")
        else:
            print("\nTop matching verses:")
            for i, (idx, verse, score) in enumerate(results, 1):
                print(f"{i}. [Line {idx+1}] {verse} (score: {score:.4f})")

def authorship_attribution():
    """Run the authorship attribution module"""
    clear_screen()
    print("\n" + "=" * 80)
    print(" " * 25 + "AUTHORSHIP ATTRIBUTION")
    print("=" * 80)
    print("This module analyzes whether a verse was likely written by Nguyễn Du.")
    print("=" * 80)
    
    # Check if we need to train the model first
    model_path = 'models/exact_verse_classifier.pkl'
    
    if not os.path.exists(model_path):
        print("\nAuthorship classifier model not found. Training a new model...")
        print("This may take a few minutes...")
        
        # Run the training script
        subprocess.run([sys.executable, "-m", "src.run_exact_classifier"])
    else:
        print(f"\nFound existing classifier model at {model_path}")
        
        # Ask if user wants to run in interactive mode
        choice = input("Enter 'I' for interactive mode or any key to run the classifier script: ").strip().upper()
        
        if choice == 'I':
            # Import the needed function
            print("\nStarting interactive mode for authorship attribution...")
            print("Enter verses to check if they were likely written by Nguyễn Du.")
            print("Type 'quit' to return to the main menu.")
            print("=" * 80)
            
            # Import the classifier
            from src.nguyen_du_classifier import ExactVerseClassifier, predict_authorship
            
            # Load the classifier
            classifier = ExactVerseClassifier.load(model_path)
            
            while True:
                verse = input("\nEnter verse: ").strip()
                if verse.lower() == 'quit':
                    break
                
                predict_authorship(verse, classifier)
        else:
            # Run the full classifier script
            subprocess.run([sys.executable, "-m", "src.run_exact_classifier"])
    
    input("\nPress Enter to return to the main menu...")

def verse_generation():
    """Generate verses using Ollama"""
    clear_screen()
    print("\n" + "=" * 80)
    print(" " * 25 + "VERSE GENERATION")
    print("=" * 80)
    
    # Check for Ollama availability
    print("Checking for Ollama availability...")
    ollama_available = False
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            ollama_available = True
            models = response.json().get("models", [])
            available_models = [model.get("name") for model in models]
            print(f"Ollama is available with {len(available_models)} models.")
        else:
            print("Ollama API is not responding correctly.")
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
    
    if not ollama_available:
        print("\nOllama seems to be unavailable. Please make sure Ollama is installed and running.")
        print("You can install Ollama from: https://ollama.ai/")
        print("After installation, run 'ollama pull llama3' to download the model.")
        input("\nPress Enter to return to the main menu...")
        return
    
    # Ask for model choice
    print("\nAvailable models:")
    for i, model in enumerate(available_models, 1):
        print(f"{i}. {model}")
        
    default_model = "llama3:latest" if "llama3:latest" in available_models else available_models[0] if available_models else "llama3"
    model_choice = input(f"\nSelect model (default: {default_model}): ").strip()
    
    if not model_choice:
        model_name = default_model
    elif model_choice.isdigit() and 1 <= int(model_choice) <= len(available_models):
        model_name = available_models[int(model_choice) - 1]
    else:
        model_name = model_choice
    
    # Initialize generator
    print(f"\nInitializing verse generator with model: {model_name}")
    try:
        generator = OllamaKieuGenerator(
            model_name=model_name,
            truyen_kieu_path='data/truyen_kieu.txt'
        )
        
        # Interactive verse generation
        print("\n" + "=" * 80)
        print(" " * 25 + "INTERACTIVE VERSE GENERATION")
        print("=" * 80)
        print("Enter a starting phrase to generate verses in the style of Truyện Kiều.")
        print("Type 'quit' to return to the main menu.")
        print("=" * 80)
        
        while True:
            prompt = input("\nStarting phrase: ").strip()
            if prompt.lower() == 'quit':
                break
            
            print("\nGenerating verse... (this may take a moment)")
            verses = generator.generate_verse(
                initial_phrase=prompt,
                num_samples=1
            )
            
            if verses:
                print("\nGenerated verse:")
                print(verses[0])
            else:
                print("Failed to generate verse. Try a different prompt.")
        
    except Exception as e:
        print(f"Error with verse generator: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to return to the main menu...")

def image_generation():
    """Generate images from verses"""
    clear_screen()
    print("\n" + "=" * 80)
    print(" " * 25 + "IMAGE GENERATION")
    print("=" * 80)
    
    # Check dependencies
    try:
        from diffusers import StableDiffusionPipeline
        import torch
        dependencies_available = True
    except ImportError:
        dependencies_available = False
    
    if not dependencies_available:
        print("\nRequired dependencies for image generation are not installed.")
        print("Please install the required packages:")
        print("pip install diffusers transformers torch")
        input("\nPress Enter to return to the main menu...")
        return
    
    # Initialize the generator
    print("Initializing image generator...")
    try:
        generator = VerseToImageGenerator()
        
        # Create output directory
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Ask for artistic style
        print("\nAvailable artistic styles:")
        styles = ['traditional', 'modern', 'fantasy', 'ukiyo-e', 'impressionist']
        for i, style in enumerate(styles, 1):
            print(f"{i}. {style}")
        
        style_choice = input("\nSelect style (default: traditional): ").strip()
        if not style_choice:
            style = "traditional"
        elif style_choice.isdigit() and 1 <= int(style_choice) <= len(styles):
            style = styles[int(style_choice) - 1]
        else:
            style = "traditional"
        
        # Interactive image generation
        print("\n" + "=" * 80)
        print(" " * 25 + "INTERACTIVE IMAGE GENERATION")
        print("=" * 80)
        print("Enter a verse to generate a corresponding image.")
        print("Type 'quit' to return to the main menu.")
        print("=" * 80)
        
        verse_count = 1
        while True:
            verse = input("\nEnter verse: ").strip()
            if verse.lower() == 'quit':
                break
            
            output_path = os.path.join(output_dir, f"generated_image_{verse_count}.jpg")
            
            print(f"\nGenerating image in {style} style...")
            print("This may take a while depending on your hardware...")
            
            generator.generate_and_save(verse, output_path, style)
            print(f"Image saved to {output_path}")
            
            verse_count += 1
    
    except Exception as e:
        print(f"Error with image generator: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to return to the main menu...")

def image_to_verse():
    """Find verses that match an input image with enhanced functionality"""
    clear_screen()
    print("\n" + "=" * 80)
    print(" " * 25 + "IMAGE-TO-VERSE RETRIEVAL")
    print("=" * 80)
    
    # Check dependencies
    try:
        from transformers import CLIPProcessor, CLIPModel
        import torch
        from sklearn.cluster import KMeans
        dependencies_available = True
    except ImportError:
        dependencies_available = False
    
    if not dependencies_available:
        print("\nRequired dependencies for image-to-verse retrieval are not installed.")
        print("Please install the required packages:")
        print("pip install transformers torch scikit-learn")
        input("\nPress Enter to return to the main menu...")
        return
    
    # Load configuration
    config_file = "config.json"
    default_config = {
        "image_to_verse": {
            "clip_model": "openai/clip-vit-base-patch32",
            "top_k": 5,
            "boost_factors": {
                "hoa": 1.5,
                "trăng": 1.3,
                "mây": 1.2,
                "nước": 1.2,
                "gió": 1.1
            },
            "feedback_file": "feedback_data.json"
        }
    }
    
    import json
    config = default_config
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Error loading config file: {e}")
            print("Using default configuration.")
    else:
        # Create default config file
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, ensure_ascii=False, indent=2)
        print(f"Created default config file: {config_file}")
    
    # Initialize the retriever
    print("Initializing image-to-verse retriever...")
    try:
        # Get model name from config
        clip_model = config["image_to_verse"]["clip_model"]
        print(f"Using CLIP model: {clip_model}")
        
        retriever = ImageToVerseRetriever(model_name=clip_model)
        
        # Load or create verse index
        data_path = 'data/truyen_kieu.txt'
        clip_index_path = 'models/clip_index/kieu_verses.pt'
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(clip_index_path), exist_ok=True)
        
        # Check if the index exists
        index_exists = os.path.exists(clip_index_path)
        
        # Try to load existing index
        if index_exists:
            try:
                print(f"Loading existing verse index from {clip_index_path}...")
                retriever.load_index(clip_index_path)
                print("Index loaded successfully.")
            except RuntimeError as e:
                print(f"Error loading index: {e}")
                print("This usually means the index was created with a different model.")
                print("Regenerating index...")
                index_exists = False
                
                # Remove old index
                try:
                    os.remove(clip_index_path)
                except:
                    pass
        
        # Create new index if needed
        if not index_exists:
            print("Creating new verse index (this may take a while)...")
            # Load verses
            print(f"Loading verses from {data_path}...")
            verses = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        verses.append(line)
            
            print(f"Indexing {len(verses)} verses...")
            retriever.index_verses(verses)
            
            print(f"Saving verse index to {clip_index_path}...")
            retriever.save_index(clip_index_path)
        
        # Import cultural context enhancer
        from src.cultural_context import CulturalContextEnhancer
        context_enhancer = CulturalContextEnhancer()
        
        # Add analyze_image_content method if not present
        if not hasattr(retriever, 'analyze_image_content'):
            def analyze_image_content(image):
                """Analyze image content to detect Vietnamese poetic elements"""
                # Convert to numpy array for color analysis
                if isinstance(image, str):
                    image = Image.open(image).convert('RGB')
                
                img_array = np.array(image)
                
                # Simple color analysis
                elements = {}
                
                # Extract dominant colors
                h, w, _ = img_array.shape
                pixels = img_array.reshape(-1, 3)
                
                # Check for blue (sky/water)
                blue_pixels = np.sum((pixels[:, 2] > 150) & (pixels[:, 0] < 150) & (pixels[:, 1] < 150))
                # Check for pink/red (flowers)
                pink_pixels = np.sum((pixels[:, 0] > 200) & (pixels[:, 1] < 180) & (pixels[:, 2] > 180))
                # Check for green (foliage)
                green_pixels = np.sum((pixels[:, 1] > 150) & (pixels[:, 0] < 150) & (pixels[:, 2] < 150))
                
                total_pixels = h * w
                
                # Add elements based on color detection
                if blue_pixels / total_pixels > 0.1:
                    elements["trời"] = 0.8  # sky
                    elements["nước"] = 0.7  # water
                
                if pink_pixels / total_pixels > 0.05:
                    elements["hoa"] = 0.9   # flowers
                
                if green_pixels / total_pixels > 0.1:
                    elements["cỏ"] = 0.7    # grass
                
                return elements
            
            retriever.analyze_image_content = analyze_image_content
        
        # Add enhanced matching method if not present
        if not hasattr(retriever, 'find_matching_verses_enhanced'):
            def find_matching_verses_enhanced(image, top_k=5):
                """Enhanced matching with cultural context"""
                # Get basic embedding matching
                image_embedding = retriever.encode_image(image)
                similarity = torch.matmul(image_embedding, retriever.verse_embeddings.T)[0]
                
                # Get content analysis
                image_elements = retriever.analyze_image_content(image)
                
                # Get candidates
                candidates_k = min(top_k * 3, len(retriever.verses))
                top_scores, top_indices = similarity.topk(candidates_k)
                
                # Enhance with boosts
                results = []
                
                for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
                    verse = retriever.verses[idx]
                    base_score = score.item()
                    
                    # Apply boosting based on elements
                    boost_factor = 1.0
                    verse_lower = verse.lower()
                    
                    # Apply boosts from config
                    for element, confidence in image_elements.items():
                        if element in verse_lower:
                            boost = config["image_to_verse"]["boost_factors"].get(element, 1.0)
                            boost_factor += confidence * (boost - 1.0)
                    
                    # Calculate enhanced score
                    enhanced_score = base_score * boost_factor
                    
                    # Add to results
                    results.append({
                        'verse': verse,
                        'score': enhanced_score,
                        'base_score': base_score,
                        'boost_factor': boost_factor,
                        'index': idx.item()
                    })
                
                # Sort by enhanced score
                results.sort(key=lambda x: x['score'], reverse=True)
                return results[:top_k]
            
            retriever.find_matching_verses_enhanced = find_matching_verses_enhanced
        
        # Add enhanced explanation method if not present
        if not hasattr(retriever, 'explain_match_enhanced'):
            def explain_match_enhanced(image, verse_idx):
                """Enhanced explanation for matches"""
                # Get the verse
                verse = retriever.verses[verse_idx]
                
                # Get image content
                image_elements = retriever.analyze_image_content(image)
                
                # Get explanations from cultural context
                suggestions = context_enhancer.suggest_verse_pairings(verse)
                
                # Extract symbols and themes
                symbols = []
                themes = []
                
                for suggestion in suggestions:
                    if suggestion.startswith("Symbols:"):
                        symbols_text = suggestion[len("Symbols: "):]
                        symbols = [s.strip() for s in symbols_text.split(',')]
                    elif suggestion.startswith("Theme:"):
                        themes.append(suggestion[len("Theme: "):])
                
                # Match image elements with verse content
                matching_elements = []
                for element, confidence in image_elements.items():
                    if element in verse.lower():
                        matching_elements.append(f"{element}")
                
                # Default elements if none found
                if not matching_elements and not symbols:
                    if any(k in ["trời", "mây"] for k in image_elements):
                        matching_elements.append("sky imagery")
                    if any(k in ["hoa", "đào"] for k in image_elements):
                        matching_elements.append("floral elements")
                    
                    if not matching_elements:
                        matching_elements = ["aesthetic elements"]
                
                # Return explanation
                return {
                    'verse': verse,
                    'key_imagery': symbols if symbols else matching_elements,
                    'detected_elements': list(image_elements.keys()),
                    'themes': themes if themes else ["General Vietnamese poetic themes"]
                }
            
            retriever.explain_match_enhanced = explain_match_enhanced
        
        # Function to record feedback
        def record_feedback(image_path, verse, rating, notes=None):
            """Record user feedback for image-to-verse matching"""
            import datetime
            
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
            
            # Create directory if needed
            os.makedirs(os.path.dirname(feedback_file) or '.', exist_ok=True)
            
            # Save updated data
            with open(feedback_file, 'w', encoding='utf-8') as f:
                json.dump(feedback_data, f, ensure_ascii=False, indent=2)
            
            return "Feedback recorded. Thank you!"
        
        # Ask if user wants to use enhanced matching
        print("\nDo you want to use enhanced matching with cultural context? (y/n)")
        enhanced = input("> ").lower() == 'y'
        
        if enhanced:
            print("Enhanced matching enabled. This will use cultural context and image analysis.")
        else:
            print("Using standard matching.")
        
        # Ask if user wants to provide feedback
        print("\nDo you want to provide feedback on matches? (y/n)")
        feedback_enabled = input("> ").lower() == 'y'
        
        if feedback_enabled:
            print("Feedback collection enabled.")
        
        # Interactive image-to-verse retrieval
        if enhanced:
            print("\n" + "=" * 80)
            print(" " * 25 + "ENHANCED IMAGE-TO-VERSE RETRIEVAL")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print(" " * 25 + "IMAGE-TO-VERSE RETRIEVAL")
            print("=" * 80)
            
        print("Enter the path to an image file to find matching verses from Truyện Kiều.")
        print("Type 'quit' to return to the main menu.")
        print("=" * 80)
        
        while True:
            image_path = input("\nEnter image path: ").strip()
            if image_path.lower() == 'quit':
                break
            
            if not os.path.exists(image_path):
                print(f"Error: Image file {image_path} not found.")
                continue
            
            try:
                image = Image.open(image_path).convert('RGB')
                
                if enhanced:
                    # Analyze image content
                    print("\nAnalyzing image content...")
                    image_elements = retriever.analyze_image_content(image)
                    if image_elements:
                        print(f"Detected elements: {', '.join(image_elements.keys())}")
                    else:
                        print("No specific elements detected.")
                    
                    # Enhanced matching
                    print("Finding matching verses with cultural context...")
                    results = retriever.find_matching_verses_enhanced(image, top_k=config["image_to_verse"]["top_k"])
                else:
                    # Standard matching
                    print("\nFinding matching verses...")
                    results = retriever.find_matching_verses(image, top_k=config["image_to_verse"]["top_k"])
                
                # Display results
                print("\nTop matching verses:")
                for i, result in enumerate(results, 1):
                    boost = result.get('boost_factor', 1.0)
                    boost_str = f" (boost: {boost:.2f})" if boost > 1.0 else ""
                    print(f"{i}. [{result['score']:.4f}]{boost_str} {result['verse']}")
                    
                # Get explanation for top match
                if results:
                    if enhanced:
                        explanation = retriever.explain_match_enhanced(image, results[0]['index'])
                        print("\nMatch explanation:")
                        print(f"Key imagery: {', '.join(explanation['key_imagery'])}")
                        
                        if 'themes' in explanation:
                            print(f"Themes: {', '.join(explanation['themes'])}")
                        
                        if 'detected_elements' in explanation:
                            print(f"Detected in image: {', '.join(explanation['detected_elements'])}")
                    else:
                        explanation = retriever.explain_match(image, results[0]['index'])
                        print("\nMatch explanation:")
                        print(f"Key imagery: {', '.join(explanation['key_imagery'])}")
                    
                    # Collect feedback if enabled
                    if feedback_enabled and results:
                        print("\nHow would you rate this match? (1-5, where 5 is excellent)")
                        try:
                            rating = int(input("> "))
                            if 1 <= rating <= 5:
                                print("Any additional notes or suggestions?")
                                notes = input("> ")
                                
                                # Record feedback
                                record_feedback(image_path, results[0]['verse'], rating, notes)
                                print("Thank you for your feedback!")
                            else:
                                print("Invalid rating. Feedback not recorded.")
                        except ValueError:
                            print("Invalid input. Feedback not recorded.")
                
            except Exception as e:
                print(f"Error processing image: {e}")
                import traceback
                traceback.print_exc()
    
    except Exception as e:
        print(f"Error with image-to-verse retriever: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to return to the main menu...")

def main_menu():
    """Display the main menu and handle user input"""
    while True:
        clear_screen()
        print_header()
        
        choice = input("\nEnter your choice (0-5): ").strip()
        
        if choice == '0':
            clear_screen()
            print("\nThank you for using the Truyện Kiều Analysis Toolkit!")
            break
        elif choice == '1':
            search_engine()
        elif choice == '2':
            authorship_attribution()
        elif choice == '3':
            verse_generation()
        elif choice == '4':
            image_generation()
        elif choice == '5':
            image_to_verse()
        else:
            print("\nInvalid choice. Please try again.")
            input("Press Enter to continue...")

if __name__ == "__main__":
    # Check if any command line arguments are provided
    if len(sys.argv) > 1:
        # If arguments are provided, use the original app.py behavior
        # Import and call the original main function
        from app import main as original_main
        original_main()
    else:
        # If no arguments are provided, show the interactive menu
        main_menu()