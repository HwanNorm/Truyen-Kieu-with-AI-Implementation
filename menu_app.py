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
    """Find verses that match an input image"""
    clear_screen()
    print("\n" + "=" * 80)
    print(" " * 25 + "IMAGE-TO-VERSE RETRIEVAL")
    print("=" * 80)
    
    # Check dependencies
    try:
        from transformers import CLIPProcessor, CLIPModel
        import torch
        dependencies_available = True
    except ImportError:
        dependencies_available = False
    
    if not dependencies_available:
        print("\nRequired dependencies for image-to-verse retrieval are not installed.")
        print("Please install the required packages:")
        print("pip install transformers torch")
        input("\nPress Enter to return to the main menu...")
        return
    
    # Initialize the retriever
    print("Initializing image-to-verse retriever...")
    try:
        retriever = ImageToVerseRetriever()
        
        # Load or create verse index
        data_path = 'data/truyen_kieu.txt'
        clip_index_path = 'models/clip_index/kieu_verses.pt'
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(clip_index_path), exist_ok=True)
        
        if os.path.exists(clip_index_path):
            print(f"Loading existing verse index from {clip_index_path}...")
            retriever.load_index(clip_index_path)
        else:
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
        
        # Interactive image-to-verse retrieval
        print("\n" + "=" * 80)
        print(" " * 25 + "INTERACTIVE IMAGE-TO-VERSE RETRIEVAL")
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
                print("\nFinding matching verses...")
                results = retriever.find_matching_verses(image, top_k=5)
                
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