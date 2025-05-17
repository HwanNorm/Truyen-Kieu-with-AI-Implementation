import torch
from PIL import Image
from typing import Union, List, Dict, Tuple, Optional
import os
import io
import base64
import numpy as np

class VerseToImageGenerator:
    """Generate images based on Truyện Kiều verses"""
    
    def __init__(self, model_id: str = "stabilityai/stable-diffusion-2"):
        """
        Initialize the image generator
        
        Args:
            model_id: Identifier for the image generation model
        """
        try:
            from diffusers import StableDiffusionPipeline
            import torch
            
            # Initialize the pipeline
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id, 
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            # Move to GPU if available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.pipe = self.pipe.to(self.device)
            
            # Memory optimization
            if self.device == "cuda":
                self.pipe.enable_attention_slicing()
                
            self.translation_dict = self._initialize_translation_dict()
            
        except ImportError:
            print("Warning: diffusers package not found. Install with 'pip install diffusers transformers'")
            self.pipe = None
    
    def _initialize_translation_dict(self) -> Dict[str, str]:
        """
        Initialize dictionary for translating Vietnamese terms to English
        
        Returns:
            Dictionary mapping Vietnamese terms to English equivalents
        """
        # Common terms and imagery in Truyện Kiều
        return {
            # Nature
            "trăng": "moon",
            "sao": "stars",
            "trời": "sky",
            "mây": "clouds",
            "gió": "wind",
            "mưa": "rain",
            "sông": "river",
            "suối": "stream",
            "biển": "sea",
            "núi": "mountains",
            "đồi": "hills",
            "rừng": "forest",
            
            # Flora
            "hoa": "flowers",
            "cỏ": "grass",
            "cây": "trees",
            "lá": "leaves",
            "liễu": "willow tree",
            "đào": "peach blossoms",
            "mai": "plum blossoms",
            "sen": "lotus",
            "trúc": "bamboo",
            
            # People and emotions
            "người": "person",
            "nàng": "young woman",
            "chàng": "young man",
            "tình": "love",
            "buồn": "sadness",
            "sầu": "melancholy",
            "đau": "pain",
            "vui": "happiness",
            "mộng": "dream",
            
            # Culture-specific
            "kiều": "beautiful Vietnamese woman",
            "áo dài": "traditional Vietnamese dress",
            "đàn": "Vietnamese musical instrument",
            "chùa": "Buddhist temple",
            "lầu": "traditional pavilion",
            
            # Abstract concepts
            "duyên": "fate",
            "nghĩa": "righteousness",
            "tài": "talent",
            "phận": "destiny"
        }
        
    def extract_imagery(self, verse: str) -> List[str]:
        """
        Extract key visual elements from a verse
        
        Args:
            verse: Vietnamese verse text
            
        Returns:
            List of English terms for visual elements
        """
        imagery = []
        
        # Convert to lowercase for matching
        verse_lower = verse.lower()
        
        # Extract imagery terms
        for vn_term, en_term in self.translation_dict.items():
            if vn_term in verse_lower:
                imagery.append(en_term)
                
        return imagery
    
    def create_prompt(self, verse: str) -> str:
        """
        Create an image generation prompt from a verse
        
        Args:
            verse: Vietnamese verse text
            
        Returns:
            English prompt for image generation
        """
        # Extract imagery from the verse
        imagery_terms = self.extract_imagery(verse)
        
        # Fallback if no specific imagery found
        if not imagery_terms:
            imagery_terms = ["Vietnamese landscape", "traditional scene"]
            
        # Build the prompt
        imagery_text = ", ".join(imagery_terms)
        
        # Create a detailed prompt for stable diffusion
        prompt = (
            f"Traditional Vietnamese art depicting {imagery_text}. "
            f"Ink painting with soft watercolors in the style of ancient Vietnamese scrolls. "
            f"Ethereal, atmospheric, delicate brushwork, subtle details. "
            f"Scene from Truyện Kiều epic poem. "
        )
        
        return prompt
    
    def enhance_prompt_with_style(self, prompt: str, style: str = "traditional") -> str:
        """
        Enhance the prompt with a specific artistic style
        
        Args:
            prompt: Base image generation prompt
            style: Artistic style to apply
            
        Returns:
            Enhanced prompt
        """
        style_enhancements = {
            "traditional": "in the style of traditional Vietnamese ink painting, soft muted colors",
            "modern": "in contemporary Vietnamese art style, vibrant colors, clean lines",
            "fantasy": "ethereal, dreamlike, mystical, glowing elements, fantasy illustration",
            "ukiyo-e": "in the style of ukiyo-e, woodblock print techniques, flat perspective",
            "impressionist": "impressionist style, light brushstrokes, emphasis on light and movement"
        }
        
        # Get enhancement for the requested style
        enhancement = style_enhancements.get(style, style_enhancements["traditional"])
        
        # Add to the prompt
        enhanced_prompt = f"{prompt} {enhancement}"
        
        return enhanced_prompt
    
    def generate_image(self, verse: str, style: str = "traditional", 
                      guidance_scale: float = 7.5, num_inference_steps: int = 50) -> Image.Image:
        """
        Generate an image based on a verse
        
        Args:
            verse: Vietnamese verse text
            style: Artistic style to apply
            guidance_scale: Controls how closely the image follows the prompt
            num_inference_steps: Number of denoising steps (higher = more detail)
            
        Returns:
            Generated image
        """
        if self.pipe is None:
            raise ImportError("Image generation requires the diffusers and transformers packages")
            
        # Create prompt
        base_prompt = self.create_prompt(verse)
        prompt = self.enhance_prompt_with_style(base_prompt, style)
        
        # Generate the image
        with torch.inference_mode():
            output = self.pipe(
                prompt=prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            )
            
        # Extract image
        image = output.images[0]
        
        return image
    
    def generate_variations(self, verse: str, num_variations: int = 3, 
                          style: str = "traditional") -> List[Image.Image]:
        """
        Generate multiple variations of an image for the same verse
        
        Args:
            verse: Vietnamese verse text
            num_variations: Number of different images to generate
            style: Artistic style to apply
            
        Returns:
            List of generated images
        """
        images = []
        
        for i in range(num_variations):
            # Vary the seed for each generation
            seed = i * 1000 + np.random.randint(1000)
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # Create prompt with slight variations
            base_prompt = self.create_prompt(verse)
            prompt = self.enhance_prompt_with_style(base_prompt, style)
            
            # Generate image
            with torch.inference_mode():
                output = self.pipe(
                    prompt=prompt,
                    generator=generator,
                    guidance_scale=7.5,
                    num_inference_steps=50
                )
                
            images.append(output.images[0])
            
        return images
    
    def image_to_base64(self, image: Image.Image, format: str = "JPEG") -> str:
        """
        Convert PIL Image to base64 string
        
        Args:
            image: PIL Image
            format: Image format (JPEG, PNG)
            
        Returns:
            Base64 encoded string
        """
        buffered = io.BytesIO()
        image.save(buffered, format=format)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    
    def save_image(self, image: Image.Image, output_path: str) -> None:
        """
        Save generated image to disk
        
        Args:
            image: PIL Image
            output_path: Path where the image should be saved
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        # Save the image
        image.save(output_path)
        
        print(f"Image saved to {output_path}")
    
    def generate_and_save(self, verse: str, output_path: str, 
                         style: str = "traditional") -> None:
        """
        Generate image from verse and save it to disk
        
        Args:
            verse: Vietnamese verse text
            output_path: Path where the image should be saved
            style: Artistic style to apply
        """
        image = self.generate_image(verse, style)
        self.save_image(image, output_path)


# Command line interface if run directly
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate images from Truyện Kiều verses')
    parser.add_argument('--verse', type=str, required=True,
                       help='Verse to visualize')
    parser.add_argument('--output', type=str, default='output.jpg',
                       help='Output image path')
    parser.add_argument('--style', type=str, 
                       choices=['traditional', 'modern', 'fantasy', 'ukiyo-e', 'impressionist'],
                       default='traditional', 
                       help='Artistic style')
    parser.add_argument('--variations', type=int, default=1,
                       help='Number of variations to generate')
    
    args = parser.parse_args()
    
    try:
        generator = VerseToImageGenerator()
        
        if args.variations > 1:
            # Generate multiple variations
            base_name, ext = os.path.splitext(args.output)
            
            images = generator.generate_variations(args.verse, args.variations, args.style)
            
            for i, image in enumerate(images):
                output_path = f"{base_name}_{i+1}{ext}"
                generator.save_image(image, output_path)
                print(f"Generated variation {i+1}")
        else:
            # Generate a single image
            generator.generate_and_save(args.verse, args.output, args.style)
            
    except Exception as e:
        print(f"Error: {e}")