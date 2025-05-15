import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import re

class KieuVerseGenerator:
    """Generator for new verses in the style of Truyện Kiều with enhanced poetic structure"""
    
    def __init__(self, model_path: str, model_type: str = 'ngram'):
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
            
        # Initialize Vietnamese linguistic resources
        self._init_vietnamese_resources()
    
    def _init_vietnamese_resources(self):
        """Initialize resources for Vietnamese poetry analysis"""
        # Vietnamese tone marks for analysis
        self.tone_marks = {
            'level': ['a', 'ă', 'â', 'e', 'ê', 'i', 'o', 'ô', 'ơ', 'u', 'ư', 'y'],
            'falling': ['à', 'ằ', 'ầ', 'è', 'ề', 'ì', 'ò', 'ồ', 'ờ', 'ù', 'ừ', 'ỳ'],
            'rising': ['á', 'ắ', 'ấ', 'é', 'ế', 'í', 'ó', 'ố', 'ớ', 'ú', 'ứ', 'ý'],
            'question': ['ả', 'ẳ', 'ẩ', 'ẻ', 'ể', 'ỉ', 'ỏ', 'ổ', 'ở', 'ủ', 'ử', 'ỷ'],
            'tumbling': ['ã', 'ẵ', 'ẫ', 'ẽ', 'ễ', 'ĩ', 'õ', 'ỗ', 'ỡ', 'ũ', 'ữ', 'ỹ'],
            'heavy': ['ạ', 'ặ', 'ậ', 'ẹ', 'ệ', 'ị', 'ọ', 'ộ', 'ợ', 'ụ', 'ự', 'ỵ']
        }
        
        # Group tones for poetry analysis
        self.tone_groups = {
            'flat': ['level', 'falling'],  # bằng
            'sharp': ['rising', 'question', 'tumbling', 'heavy']  # trắc
        }
        
        # Common ending words for Truyện Kiều verses
        self.common_endings = ['ta', 'này', 'kia', 'đây', 'chi', 'sao', 'nào', 'thay', 
                              'chăng', 'vay', 'thôi', 'rồi', 'ra', 'vào', 
                              'ai', 'người', 'rằng', 'đà']
        
        # Patterns for lục bát structure
        self.luc_bat_patterns = {
            'luc': [None, 'flat', None, 'sharp', None, 'flat'],  # 6-syllable line
            'bat': [None, 'flat', None, 'flat', None, 'sharp', None, 'flat']  # 8-syllable line
        }
    
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
    
    def get_syllable_tone(self, syllable: str) -> str:
        """
        Determine the tone group (flat or sharp) of a Vietnamese syllable
        
        Args:
            syllable: A Vietnamese syllable
            
        Returns:
            Tone group ('flat' or 'sharp')
        """
        syllable = syllable.lower()
        
        # Check for tone marks
        for tone_name, tone_chars in self.tone_marks.items():
            if any(char in syllable for char in tone_chars):
                # Find which tone group this belongs to
                for group, tones in self.tone_groups.items():
                    if tone_name in tones:
                        return group
        
        # Default to flat tone if no tone mark found
        return 'flat'
    
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
        if not verse:
            return ""
            
        # 2. Split into words/syllables
        words = verse.split()
        if len(words) < 4:
            return ""  # Too short to be valid
        
        # 3. Determine if this should be lục (6) or bát (8) based on content
        # Use the first few words to decide
        starting_words = ' '.join(words[:2]).lower()
        if any(word in starting_words for word in ["trăm", "năm", "chữ", "một", "người", "tài"]):
            target_length = 6  # Lục (six)
            pattern = self.luc_bat_patterns['luc']
        else:
            target_length = 8  # Bát (eight)
            pattern = self.luc_bat_patterns['bat']
        
        # 4. Adjust to target length
        if len(words) > target_length:
            # Keep the beginning and select appropriate words for the ending
            # to better follow Vietnamese verse structure
            if len(words) > target_length + 2:
                # Take beginning and end, joining them coherently
                words = words[:target_length-1] + [self._select_ending_word(words)]
            else:
                words = words[:target_length]
                
        # 5. Ensure proper ending punctuation
        verse = ' '.join(words)
        if not verse.endswith((',', '.', '?', '!')):
            verse += ','
            
        # 6. Apply tone pattern adjustments if needed
        # This is a simplification - a full implementation would require
        # deeper linguistic processing
        verse = self._adjust_tones(verse, pattern)
            
        return verse
    
    def _select_ending_word(self, words: List[str]) -> str:
        """Select an appropriate ending word for a verse"""
        # Try to use an existing ending word from the original
        for word in reversed(words):
            if word.lower() in self.common_endings:
                return word
                
        # Otherwise use a common ending
        return np.random.choice(self.common_endings)
    
    def _adjust_tones(self, verse: str, pattern: List[Optional[str]]) -> str:
        """
        Attempt to adjust the verse to follow Vietnamese tonal patterns
        This is a simplified approximation
        """
        words = verse.split()
        
        # If we don't have enough words to match the pattern, return as is
        if len(words) < len(pattern):
            return verse
            
        # Check if the verse already follows the pattern
        matches_pattern = True
        for i, tone in enumerate(pattern):
            if i >= len(words) or tone is None:
                continue
                
            if self.get_syllable_tone(words[i]) != tone:
                matches_pattern = False
                break
                
        # If it already matches, return as is
        if matches_pattern:
            return verse
            
        # Otherwise make minor adjustments - this is where more sophisticated
        # linguistic processing would be needed for a full implementation
        
        # For simplicity, we'll just ensure the last syllable has the right tone
        last_idx = len(pattern) - 1
        if last_idx < len(words) and pattern[last_idx] == 'flat':
            # For demonstration - a real implementation would need a dictionary
            # of Vietnamese word alternatives with appropriate tones
            if self.get_syllable_tone(words[last_idx]) != 'flat':
                # Try to replace with a common flat-toned ending
                flat_endings = [e for e in self.common_endings 
                               if self.get_syllable_tone(e) == 'flat']
                if flat_endings:
                    words[last_idx] = np.random.choice(flat_endings)
        
        return ' '.join(words)
    
    def evaluate_verse_quality(self, verse: str) -> Dict[str, float]:
        """
        Evaluate the quality of a generated verse using enhanced metrics
        
        Args:
            verse: The verse to evaluate
            
        Returns:
            Dictionary with quality metrics
        """
        # Simplified evaluation metrics
        scores = {
            'length': 0.0,  # Length appropriateness
            'structure': 0.0,  # Vietnamese poetic structure
            'fluency': 0.0,  # Language fluency
            'tone_pattern': 0.0  # Vietnamese tone pattern adherence
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
            
        # 4. Tone pattern score - NEW!
        # Check if the verse follows Vietnamese tonal patterns
        if len(words) == 6:  # lục
            pattern = self.luc_bat_patterns['luc']
        else:  # bát or other
            pattern = self.luc_bat_patterns['bat']
            
        matches = 0
        required_positions = [i for i, tone in enumerate(pattern) if tone is not None]
        
        for pos in required_positions:
            if pos < len(words) and self.get_syllable_tone(words[pos]) == pattern[pos]:
                matches += 1
                
        if required_positions:
            scores['tone_pattern'] = matches / len(required_positions)
        else:
            scores['tone_pattern'] = 0.5
            
        # Overall quality score
        scores['overall'] = (scores['length'] + scores['structure'] + 
                           scores['fluency'] + scores['tone_pattern']) / 4
        
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
        
        # Ensure poetic connection between verses
        first_words = first_verse.split()
        second_words = second_verse.split()
        
        # For lục bát form, ensure proper structure:
        # - If first verse has 6 syllables, second should have 8
        # - If first verse has 8 syllables, second should have 6
        if len(first_words) <= 6 and len(second_words) > 8:
            second_verse = ' '.join(second_words[:8])
        elif len(first_words) > 6 and len(second_words) > 6:
            second_verse = ' '.join(second_words[:6])
            
        # Ensure ending punctuation
        if not second_verse.endswith((',', '.', '?', '!')):
            second_verse += '.'
            
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
