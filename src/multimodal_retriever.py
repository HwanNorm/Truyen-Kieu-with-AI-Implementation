import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional, Union
import os
import re

class ImageToVerseRetriever:
    """Find verses from Truyện Kiều that match an input image"""
    
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):  
        """
        Initialize the retriever with a vision-language model
        
        Args:
            model_name: Name of the CLIP model to use
        """
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            # Load model and processor
            print(f"Loading CLIP model: {model_name}")
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            
            # Move to GPU if available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {self.device}")
            self.model = self.model.to(self.device)
            
            # Storage for indexed verses
            self.verses = None
            self.verse_embeddings = None
            
        except ImportError:
            print("Warning: transformers package not found. Install with 'pip install transformers'")
            self.model = None
            self.processor = None

    def analyze_image_content(self, image: Union[str, Image.Image]) -> Dict[str, float]:
        """
        Analyze the content of an image to identify Vietnamese poetic elements
        
        Args:
            image: Path to image file or PIL Image
                
        Returns:
            Dictionary of detected elements and their confidence scores
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
            
        # Convert to numpy for color analysis
        import numpy as np
        img_array = np.array(image)
        
        # Detect dominant colors (simplified HSV analysis)
        # This helps identify elements like blue sky, green fields, pink flowers, etc.
        def get_dominant_colors(img):
            from sklearn.cluster import KMeans
            # Reshape the image for KMeans
            pixels = img.reshape(-1, 3)
            # Limit samples for speed
            samples = pixels[np.random.choice(pixels.shape[0], min(10000, pixels.shape[0]), replace=False)]
            # Apply KMeans to find dominant colors
            kmeans = KMeans(n_clusters=5, n_init=10)
            kmeans.fit(samples)
            # Get the colors
            colors = kmeans.cluster_centers_
            # Count pixels in each cluster
            labels = kmeans.predict(pixels)
            counts = np.bincount(labels)
            # Return colors and their percentages
            percentages = counts / counts.sum()
            return [(color, percent) for color, percent in zip(colors, percentages)]
        
        try:
            dominant_colors = get_dominant_colors(img_array)
            
            # Map colors to Vietnamese poetic elements
            elements = {}
            
            # Check for blue (sky, water)
            for color, percent in dominant_colors:
                r, g, b = color
                # Blue detection
                if b > max(r, g) + 20:
                    elements["trời"] = elements.get("trời", 0) + percent * 0.7  # sky
                    elements["nước"] = elements.get("nước", 0) + percent * 0.5  # water
                
                # Pink/red detection (flowers)
                if r > max(g, b) + 20 and r > 180:
                    elements["hoa"] = elements.get("hoa", 0) + percent * 0.8  # flowers
                    elements["đào"] = elements.get("đào", 0) + percent * 0.6  # peach blossoms
                
                # Green detection (leaves, plants)
                if g > max(r, b) + 20:
                    elements["lá"] = elements.get("lá", 0) + percent * 0.7  # leaves
                    elements["cỏ"] = elements.get("cỏ", 0) + percent * 0.6  # grass
            
            # Use PIL to get image brightness - helps detect day vs night imagery
            brightness = np.mean(img_array)
            if brightness > 170:
                elements["sáng"] = 0.8  # bright day
            elif brightness < 100:
                elements["đêm"] = 0.7  # night
                elements["trăng"] = 0.5  # moon
            
            return elements
        
        except Exception as e:
            print(f"Error in image content analysis: {e}")
            return {}
    
    def load_verses(self, verses_file: str) -> List[str]:
        """
        Load verses from a file
        
        Args:
            verses_file: Path to the file containing verses
            
        Returns:
            List of verses
        """
        with open(verses_file, 'r', encoding='utf-8') as f:
            # Remove line numbers and clean up
            verses = []
            for line in f:
                line = line.strip()
                if line:
                    # Remove line numbers if present (e.g., "123. ")
                    line = line.split('. ', 1)[-1] if '. ' in line and line.split('. ')[0].isdigit() else line
                    verses.append(line)
            
        return verses
    
    def index_verses(self, verses: List[str], batch_size: int = 32) -> None:
        """
        Compute and store embeddings for all verses
        
        Args:
            verses: List of verses to index
            batch_size: Number of verses to process in each batch
        """
        if self.model is None:
            raise ImportError("Verse retrieval requires the transformers package")
            
        self.verses = verses
        
        # Process in batches to avoid memory issues
        all_embeddings = []
        
        for i in range(0, len(verses), batch_size):
            batch = verses[i:i+batch_size]
            
            # Encode the batch
            inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get text embeddings
            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)
                all_embeddings.append(outputs.cpu())
                
        # Concatenate all embeddings
        self.verse_embeddings = torch.cat(all_embeddings)
        
        # Normalize for cosine similarity
        self.verse_embeddings = torch.nn.functional.normalize(self.verse_embeddings, p=2, dim=1)
        
        print(f"Indexed {len(verses)} verses")
    
    def save_index(self, filepath: str) -> None:
        """
        Save the verse index to disk
        
        Args:
            filepath: Path to save the index
        """
        if self.verses is None or self.verse_embeddings is None:
            raise ValueError("No verses indexed yet")
            
        torch.save({
            'verses': self.verses,
            'embeddings': self.verse_embeddings
        }, filepath)
        
        print(f"Index saved to {filepath}")
    
    def load_index(self, filepath: str) -> None:
        """
        Load a verse index from disk
        
        Args:
            filepath: Path to the saved index
        """
        data = torch.load(filepath, map_location=torch.device('cpu'))
        self.verses = data['verses']
        self.verse_embeddings = data['embeddings']
        
        print(f"Loaded index with {len(self.verses)} verses")
    
    def encode_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """
        Encode an image into a feature vector
        
        Args:
            image: Path to image file or PIL Image
            
        Returns:
            Tensor containing image embedding
        """
        if self.model is None:
            raise ImportError("Image encoding requires the transformers package")
            
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
            
        # Process the image
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get image embedding
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            
        # Normalize for cosine similarity
        image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)
        
        return image_features.cpu()
    
    def find_matching_verses(self, image: Union[str, Image.Image], 
                           top_k: int = 5) -> List[Dict[str, Union[str, float, int]]]:
        """
        Find verses that best match an input image
        
        Args:
            image: Path to image file or PIL Image
            top_k: Number of top matches to return
            
        Returns:
            List of dictionaries containing matching verses and scores
        """
        if self.verses is None or self.verse_embeddings is None:
            raise ValueError("No verses indexed yet. Call index_verses() first")
            
        # Encode the image
        image_embedding = self.encode_image(image)
        
        # Calculate similarity with all verses
        similarity = torch.matmul(image_embedding, self.verse_embeddings.T)[0]
        
        # Get top matches
        top_scores, top_indices = similarity.topk(min(top_k, len(self.verses)))
        
        # Create result list
        results = []
        for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
            results.append({
                'verse': self.verses[idx],
                'score': score.item(),
                'rank': i + 1,
                'index': idx.item()
            })
            
        return results
    
    def explain_match(self, image: Union[str, Image.Image], 
                    verse_idx: int) -> Dict[str, Union[str, float, List[str]]]:
        """
        Explain why an image matches a specific verse
        
        Args:
            image: Path to image file or PIL Image
            verse_idx: Index of the verse to explain
            
        Returns:
            Dictionary with explanation
        """
        if self.verses is None or self.verse_embeddings is None:
            raise ValueError("No verses indexed yet")
            
        # Encode the image
        image_embedding = self.encode_image(image)
        
        # Get the verse embedding
        verse_embedding = self.verse_embeddings[verse_idx].unsqueeze(0)
        
        # Calculate similarity
        similarity = torch.matmul(image_embedding, verse_embedding.T)[0, 0].item()
        
        # Extract key words from the verse
        verse = self.verses[verse_idx]
        words = verse.lower().split()
        
        # Simple keyword extraction
        # Vietnamese keywords typically related to imagery
        imagery_keywords = [
            "trăng", "hoa", "liễu", "mây", "núi", "sông", "gió", "mưa", 
            "trời", "biển", "tuyết", "sao", "nắng", "đêm", "đồi", "rừng"
        ]
        
        # Filter words based on imagery keywords
        important_words = []
        for word in words:
            for keyword in imagery_keywords:
                if keyword in word:
                    important_words.append(word)
                    break
        
        # Return explanation
        return {
            'verse': verse,
            'similarity_score': similarity,
            'key_imagery': important_words if important_words else ["No specific imagery identified"]
        }
    
    def find_complementary_verses(self, image: Union[str, Image.Image], 
                               top_k: int = 3) -> List[Dict[str, Union[str, float, int]]]:
        """
        Find verses that complement (rather than directly match) an image
        
        Args:
            image: Path to image file or PIL Image
            top_k: Number of top complementary verses to return
            
        Returns:
            List of dictionaries containing complementary verses and scores
        """
        if self.verses is None or self.verse_embeddings is None:
            raise ValueError("No verses indexed yet")
            
        # Encode the image
        image_embedding = self.encode_image(image)
        
        # Calculate similarity with all verses
        similarity = torch.matmul(image_embedding, self.verse_embeddings.T)[0]
        
        # Find verses with medium similarity (complementary rather than identical)
        # Get verses from the middle of the similarity distribution
        similarity_values, similarity_indices = similarity.sort(descending=True)
        
        # Take verses from the top quarter but not the very top
        start_idx = len(similarity_indices) // 10  # Skip the most similar 10%
        end_idx = start_idx + top_k
        
        # Create result list
        results = []
        for i, idx in enumerate(similarity_indices[start_idx:end_idx]):
            results.append({
                'verse': self.verses[idx],
                'score': similarity[idx].item(),
                'rank': i + 1,
                'index': idx.item()
            })
            
        return results
    
    def batch_process_images(self, image_dir: str, output_file: str, 
                           top_k: int = 3) -> None:
        """
        Process multiple images and save their matching verses
        
        Args:
            image_dir: Directory containing images
            output_file: File to save results
            top_k: Number of top matches per image
        """
        if self.verses is None or self.verse_embeddings is None:
            raise ValueError("No verses indexed yet")
            
        # Get all image files
        image_files = []
        for file in os.listdir(image_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_files.append(os.path.join(image_dir, file))
                
        # Process each image
        results = {}
        for img_file in image_files:
            try:
                matches = self.find_matching_verses(img_file, top_k)
                results[os.path.basename(img_file)] = matches
                print(f"Processed {img_file}")
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                
        # Save results
        import json
        
        # Convert results to JSON-serializable format
        serializable_results = {}
        for img_file, matches in results.items():
            serializable_results[img_file] = [
                {k: v for k, v in match.items() if k != 'tensor'} 
                for match in matches
            ]
            
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
            
        print(f"Results saved to {output_file}")


    def find_matching_verses_enhanced(self, image: Union[str, Image.Image], 
                               top_k: int = 5) -> List[Dict[str, Union[str, float, int]]]:
        """
        Find verses that best match an input image with enhanced cultural context
        
        Args:
            image: Path to image file or PIL Image
            top_k: Number of top matches to return
                
        Returns:
            List of dictionaries containing matching verses and scores
        """
        if self.verses is None or self.verse_embeddings is None:
            raise ValueError("No verses indexed yet. Call index_verses() first")
            
        # Import cultural context enhancer
        from src.cultural_context import CulturalContextEnhancer
        context_enhancer = CulturalContextEnhancer()
        
        # Basic embedding matching using CLIP
        image_embedding = self.encode_image(image)
        similarity = torch.matmul(image_embedding, self.verse_embeddings.T)[0]
        
        # Get image content analysis
        image_elements = self.analyze_image_content(image)
        
        # Get candidate verses (get more than needed for reranking)
        candidates_k = min(top_k * 3, len(self.verses))
        top_scores, top_indices = similarity.topk(candidates_k)
        
        # Calculate enhanced scores with cultural context
        enhanced_results = []
        for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
            verse = self.verses[idx]
            base_score = score.item()
            
            # Apply content-based boosting
            verse_lower = verse.lower()
            boost_factor = 1.0
            
            # Boost score based on detected elements
            for element, confidence in image_elements.items():
                if element in verse_lower:
                    # The boost depends on confidence and exact term match
                    boost_factor += confidence * 0.3
            
            # Get thematic suggestions from cultural context
            theme_suggestions = context_enhancer.suggest_verse_pairings(verse)
            
            # Check for symbolic matches
            for suggestion in theme_suggestions:
                if suggestion.startswith("Symbols:"):
                    # If any symbols mentioned, boost the score
                    boost_factor += 0.2
            
            # Calculate enhanced score
            enhanced_score = base_score * boost_factor
            
            enhanced_results.append({
                'verse': verse,
                'score': enhanced_score,
                'base_score': base_score,
                'boost_factor': boost_factor,
                'rank': i + 1,
                'index': idx.item()
            })
                
        # Sort by enhanced score
        enhanced_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top-k results
        return enhanced_results[:top_k]
    
    def explain_match_enhanced(self, image: Union[str, Image.Image], 
                       verse_idx: int) -> Dict[str, Union[str, float, List[str]]]:
        """
        Provide a detailed explanation of why an image matches a specific verse
        
        Args:
            image: Path to image file or PIL Image
            verse_idx: Index of the verse to explain
                
        Returns:
            Dictionary with detailed explanation
        """
        if self.verses is None or self.verse_embeddings is None:
            raise ValueError("No verses indexed yet")
            
        from src.cultural_context import CulturalContextEnhancer
        context = CulturalContextEnhancer()
        
        # Get the verse
        verse = self.verses[verse_idx]
        
        # Encode the image
        image_embedding = self.encode_image(image)
        
        # Get the verse embedding
        verse_embedding = self.verse_embeddings[verse_idx].unsqueeze(0)
        
        # Calculate similarity
        similarity = torch.matmul(image_embedding, verse_embedding.T)[0, 0].item()
        
        # Analyze image content
        image_elements = self.analyze_image_content(image)
        
        # Get cultural context suggestions
        suggestions = context.suggest_verse_pairings(verse)
        
        # Extract key symbols and themes from suggestions
        symbols = []
        themes = []
        
        for suggestion in suggestions:
            if suggestion.startswith("Symbols:"):
                symbols_text = suggestion[len("Symbols: "):]
                for symbol_entry in symbols_text.split(', '):
                    if '(' in symbol_entry:
                        symbol, meaning = symbol_entry.split('(', 1)
                        symbols.append(f"{symbol.strip()} - {meaning.rstrip(')')}")
                    else:
                        symbols.append(symbol_entry)
            elif suggestion.startswith("Theme:"):
                themes.append(suggestion[len("Theme: "):])
        
        # Match image elements with verse content
        matching_elements = []
        for element, confidence in image_elements.items():
            if element in verse.lower():
                matching_elements.append(f"{element} ({confidence:.2f})")
        
        if not matching_elements and not symbols:
            # If no specific elements matched, identify general imagery
            general_imagery = []
            if any(k in ["trời", "mây"] for k in image_elements):
                general_imagery.append("sky imagery")
            if any(k in ["hoa", "đào", "mai"] for k in image_elements):
                general_imagery.append("floral elements")
            if any(k in ["nước", "sông"] for k in image_elements):
                general_imagery.append("water imagery")
            
            if not general_imagery:
                general_imagery = ["aesthetic elements typical in Vietnamese poetry"]
                
            matching_elements = general_imagery
        
        # Return enhanced explanation
        return {
            'verse': verse,
            'similarity_score': similarity,
            'key_imagery': symbols if symbols else matching_elements,
            'detected_elements': list(image_elements.keys()),
            'themes': themes if themes else ["General Vietnamese poetic themes"]
        }


# Command line interface if run directly
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Find verses from Truyện Kiều that match images')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to the input image')
    parser.add_argument('--verses', type=str, default='data/truyen_kieu.txt',
                       help='Path to the file containing verses')
    parser.add_argument('--index', type=str, default=None,
                       help='Path to saved index (if already created)')
    parser.add_argument('--save-index', type=str, default=None,
                       help='Save index to this path')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of top matches to return')
    
    args = parser.parse_args()
    
    try:
        retriever = ImageToVerseRetriever()
        
        # Load or create index
        if args.index and os.path.exists(args.index):
            retriever.load_index(args.index)
        else:
            verses = retriever.load_verses(args.verses)
            retriever.index_verses(verses)
            
            if args.save_index:
                retriever.save_index(args.save_index)
                
        # Find matching verses
        results = retriever.find_matching_verses(args.image, args.top_k)
        
        print(f"\nTop {len(results)} verses matching the image:")
        for result in results:
            print(f"{result['rank']}. [{result['score']:.4f}] {result['verse']}")
            
        # Get explanation for top match
        explanation = retriever.explain_match(args.image, results[0]['index'])
        print(f"\nMatch explanation for top verse:")
        print(f"Key imagery: {', '.join(explanation['key_imagery'])}")
        
    except Exception as e:
        print(f"Error: {e}")

    