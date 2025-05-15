import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import re

class KieuVerseGenerator:
    """Generator for new verses in the style of Truyện Kiều"""
    
    def __init__(self, model_path: str, model_type: str = 'lstm'):
        """
        Initialize the verse generator with a trained language model
        
        Args:
            model_path: Path to the trained language model
            model_type: Type of language model ('ngram', 'lstm', or 'transformer')
        """
        from .language_model import KieuLanguageModelTrainer, TransformerKieuModel
        
        self.model_type = model_type
        
        if model_type in ['ngram', 'lstm']:
            self.model = KieuLanguageModelTrainer.load(model_path, model_type)
        elif model_type == 'transformer':
            self.model = TransformerKieuModel(model_path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def generate_verse(self, initial_phrase: str, max_length: int = 50, 
                      num_samples: int = 5) -> List[str]:
        """
        Generate verses starting with the provided phrase
        
        Args:
            initial_phrase: The starting phrase for generation
            max_length: Maximum length of the generated verse
            num_samples: Number of different verses to generate
            
        Returns:
            A list of generated verses
        """
        generated_verses = []
        
        if self.model_type == 'transformer':
            # Transformer models can generate multiple samples at once
            generated_verses = self.model.generate(
                initial_phrase, 
                max_length=max_length, 
                num_return_sequences=num_samples
            )
        else:
            # For ngram and lstm, generate samples sequentially
            for _ in range(num_samples):
                verse = self.model.generate(initial_phrase, max_length)
                generated_verses.append(verse)
                
        # Post-process and filter generated verses
        return self.refine_verses(generated_verses)
    
    def refine_verses(self, verses: List[str]) -> List[str]:
        """
        Apply post-processing to improve the quality of generated verses
        
        Args:
            verses: List of raw generated verses
            
        Returns:
            List of refined verses
        """
        refined_verses = []
        
        for verse in verses:
            # Apply Vietnamese poetic structure refinements
            refined = self.apply_poetic_constraints(verse)
            
            if refined and len(refined) >= 10:  # Minimum length check
                refined_verses.append(refined)
                
        return refined_verses
    
    def apply_poetic_constraints(self, verse: str) -> str:
        """
        Apply Vietnamese poetic structure constraints to the verse
        
        Args:
            verse: Raw generated verse
            
        Returns:
            Refined verse that better follows the Truyện Kiều structure
        """
        # 1. Clean up the verse
        verse = verse.strip()
        
        # 2. Ensure the verse ends with a comma or period
        if not verse.endswith((',', '.', '?', '!')):
            if len(verse) > 3 and verse[-1].isalpha():
                verse += ','
                
        # 3. Apply 6-8 syllable constraint (typical for Truyện Kiều)
        # Count Vietnamese syllables (approximately - each word is roughly a syllable)
        words = verse.split()
        
        if len(words) < 4:
            # Too short, not a valid verse
            return ""
            
        if len(words) > 9:
            # Too long, truncate to 8 syllables + punctuation
            verse = ' '.join(words[:8])
            
            # Add ending punctuation
            if not verse.endswith((',', '.', '?', '!')):
                verse += ','
                
        # 4. Check for basic Vietnamese tonal patterns
        # (simplified - a full implementation would be more complex)
        
        return verse
    
    def evaluate_verse_quality(self, verse: str) -> Dict[str, float]:
        """
        Evaluate the quality of a generated verse
        
        Args:
            verse: The verse to evaluate
            
        Returns:
            Dictionary with quality metrics
        """
        # Simplified evaluation metrics
        scores = {
            'length': 0.0,  # Length appropriateness
            'structure': 0.0,  # Vietnamese poetic structure
            'fluency': 0.0  # Language fluency
        }
        
        # 1. Length score
        words = verse.split()
        if 6 <= len(words) <= 8:  # Ideal length for Truyện Kiều
            scores['length'] = 1.0
        elif 4 <= len(words) <= 10:
            scores['length'] = 0.7
        else:
            scores['length'] = 0.3
            
        # 2. Structure score - check for ending punctuation
        if verse.endswith((',', '.', '?', '!')):
            scores['structure'] = 0.8
        else:
            scores['structure'] = 0.4
            
        # 3. Fluency score - simplified
        # Count vowels as proxy for Vietnamese syllable flow
        vowel_pattern = re.compile(r'[aàáảãạăằắẳẵặâầấẩẫậeèéẻẽẹêềếểễệiìíỉĩịoòóỏõọôồốổỗộơờớởỡợuùúủũụưừứửữựyỳýỷỹỵ]', re.IGNORECASE)
        vowel_count = len(vowel_pattern.findall(verse))
        
        if vowel_count >= len(words):
            scores['fluency'] = 0.8
        else:
            scores['fluency'] = 0.5
            
        # Overall quality score
        scores['overall'] = (scores['length'] + scores['structure'] + scores['fluency']) / 3
        
        return scores
        
    def generate_with_theme(self, theme: str, max_length: int = 50, 
                           num_samples: int = 5) -> List[str]:
        """
        Generate verses related to a specific theme
        
        Args:
            theme: Theme or topic for the verse
            max_length: Maximum length of the generated verse
            num_samples: Number of different verses to generate
            
        Returns:
            List of generated verses related to the theme
        """
        # Map themes to starting phrases typical in Truyện Kiều
        theme_starters = {
            'love': ['Tình yêu', 'Lòng thương', 'Duyên tình'],
            'nature': ['Trăng sao', 'Cỏ cây', 'Sông núi'],
            'fate': ['Số phận', 'Duyên trời', 'Kiếp người'],
            'beauty': ['Hồng nhan', 'Sắc đẹp', 'Má hồng'],
            'sadness': ['Sầu đau', 'Lệ rơi', 'Buồn thương'],
            'happiness': ['Vui mừng', 'Hạnh phúc', 'Niềm vui'],
            'default': ['Trăm năm', 'Cuộc đời', 'Thời gian']
        }
        
        # Get appropriate starters for the theme
        starters = theme_starters.get(theme.lower(), theme_starters['default'])
        
        # Generate verses with each starter
        verses = []
        for starter in starters:
            theme_verses = self.generate_verse(starter, max_length, 
                                             num_samples=num_samples//len(starters) + 1)
            verses.extend(theme_verses)
            
        # Return the best verses up to num_samples
        return verses[:num_samples]
    
    def generate_verse_pair(self, initial_phrase: str = None, 
                           max_length: int = 50) -> Tuple[str, str]:
        """
        Generate a pair of verses that work together poetically
        
        Args:
            initial_phrase: Optional starting phrase
            max_length: Maximum length for each verse
            
        Returns:
            A tuple of two verses
        """
        if initial_phrase is None:
            # Start with common Truyện Kiều beginnings if no phrase provided
            starters = ['Trăm năm', 'Một ngày', 'Tình duyên', 'Kiếp người']
            initial_phrase = np.random.choice(starters)
            
        # Generate first verse
        first_verses = self.generate_verse(initial_phrase, max_length, num_samples=3)
        
        if not first_verses:
            # Fallback if generation failed
            return (initial_phrase + "...", "...")
            
        first_verse = first_verses[0]
        
        # Extract the last few words to use as context for the second verse
        words = first_verse.split()
        if len(words) > 2:
            second_starter = ' '.join(words[-2:])
        else:
            second_starter = first_verse
            
        # Generate second verse that follows from the first
        second_verses = self.generate_verse(second_starter, max_length, num_samples=3)
        
        if not second_verses:
            # Fallback if generation failed
            return (first_verse, second_starter + "...")
            
        second_verse = second_verses[0]
        
        return (first_verse, second_verse)


# Command line interface if run directly
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate verses in the style of Truyện Kiều')
    parser.add_argument('--model', type=str, default='models/kieu_ngram.pkl',
                       help='Path to the trained model')
    parser.add_argument('--model-type', type=str, choices=['ngram', 'lstm', 'transformer'],
                       default='ngram', help='Type of language model')
    parser.add_argument('--prompt', type=str, default='Trăm năm',
                       help='Initial phrase for generation')
    parser.add_argument('--samples', type=int, default=3,
                       help='Number of verses to generate')
    parser.add_argument('--theme', type=str, default=None,
                       help='Theme for verse generation')
    parser.add_argument('--pair', action='store_true',
                       help='Generate a pair of verses')
    
    args = parser.parse_args()
    
    try:
        # Initialize generator
        generator = KieuVerseGenerator(args.model, args.model_type)
        
        print(f"\nGenerating verses in the style of Truyện Kiều...")
        
        if args.pair:
            # Generate a verse pair
            first, second = generator.generate_verse_pair(args.prompt)
            print("\nGenerated verse pair:")
            print(f"1: {first}")
            print(f"2: {second}")
        elif args.theme:
            # Generate verses with a theme
            verses = generator.generate_with_theme(args.theme, num_samples=args.samples)
            print(f"\nGenerated verses on theme '{args.theme}':")
            for i, verse in enumerate(verses, 1):
                print(f"{i}: {verse}")
        else:
            # Generate from prompt
            verses = generator.generate_verse(args.prompt, num_samples=args.samples)
            print(f"\nGenerated verses from '{args.prompt}':")
            for i, verse in enumerate(verses, 1):
                print(f"{i}: {verse}")
                
    except Exception as e:
        print(f"Error: {e}")