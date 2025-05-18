import os
import re
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import random

class LlamaKieuGenerator:
    """
    Vietnamese verse generator that uses LLaMA 3.2 to generate poetry in the style of Truyện Kiều.
    Specifically designed to maintain the lục bát (6-8 syllable) pattern and Vietnamese poetic style.
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-8B-Instruct", 
                 device: str = None, 
                 truyen_kieu_path: str = "data/truyen_kieu.txt",
                 use_4bit: bool = False):
        """
        Initialize the LLaMA-based verse generator
        
        Args:
            model_name: Name or path of the LLaMA model to use
            device: Device to run the model on (None for auto-detection)
            truyen_kieu_path: Path to the Truyện Kiều text file
            use_4bit: Whether to use 4-bit quantization (for memory efficiency)
        """
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading LLaMA model on {self.device}...")
        
        # Load model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Configure quantization if needed
            if use_4bit and self.device == "cuda":
                print("Using 4-bit quantization for memory efficiency")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please make sure you have the required libraries installed and permissions to download the model")
            raise
        
        # Load Truyện Kiều data
        self.truyen_kieu_verses = self.load_truyen_kieu(truyen_kieu_path)
        print(f"Loaded {len(self.truyen_kieu_verses)} verses from Truyện Kiều")
        
        # Initialize Vietnamese language resources
        self._init_vietnamese_resources()
    
    def load_truyen_kieu(self, filepath: str) -> List[str]:
        """
        Load Truyện Kiều verses from file
        
        Args:
            filepath: Path to the Truyện Kiều text file
            
        Returns:
            List of verses
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            
            verses = []
            lines = text.strip().split('\n')
            
            # Process lines
            current_verse = []
            
            for line in lines:
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Remove line numbers if present (e.g., "123.")
                line = re.sub(r'^\d+\.\s*', '', line)
                
                # Add to current verse
                current_verse.append(line)
                
                # If we have a complete lục-bát pair (or more), save it
                if len(current_verse) >= 2:
                    if len(current_verse) % 2 == 0:  # Even number of lines
                        # Process pairs (lục-bát)
                        for i in range(0, len(current_verse), 2):
                            if i+1 < len(current_verse):
                                verse_pair = '\n'.join(current_verse[i:i+2])
                                verses.append(verse_pair)
                        current_verse = []
            
            # Add any remaining incomplete verse
            if current_verse:
                verses.append('\n'.join(current_verse))
                
            return verses
            
        except Exception as e:
            print(f"Error loading Truyện Kiều data: {e}")
            return []
    
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
        
        # Extract common ending words from the actual Truyện Kiều text
        self.common_endings = self._extract_common_endings()
        
        # Patterns for lục bát structure (None means any tone is acceptable)
        self.luc_bat_patterns = {
            'luc': [None, 'flat', None, 'sharp', None, 'flat'],  # 6-syllable line
            'bat': [None, 'flat', None, 'flat', None, 'sharp', None, 'flat']  # 8-syllable line
        }
        
        # Extract common themes and vocabulary from Truyện Kiều
        self.common_themes, self.common_vocab = self._extract_themes_and_vocabulary()
    
    def _extract_common_endings(self) -> List[str]:
        """Extract common ending words from Truyện Kiều verses"""
        if not self.truyen_kieu_verses:
            # Fallback to predefined list if no verses are loaded
            return ['ta', 'này', 'kia', 'đây', 'chi', 'sao', 'nào', 'thay', 
                    'chăng', 'vay', 'thôi', 'rồi', 'ra', 'vào', 
                    'ai', 'người', 'rằng', 'đà']
        
        # Count ending words
        ending_words = []
        for verse in self.truyen_kieu_verses:
            lines = verse.strip().split('\n')
            for line in lines:
                words = line.strip().split()
                if words:
                    ending_words.append(words[-1])
        
        # Count frequencies
        from collections import Counter
        counter = Counter(ending_words)
        
        # Get the most common endings (with flat tones)
        common_endings = []
        for word, count in counter.most_common(30):
            if self.get_syllable_tone(word) == 'flat':
                common_endings.append(word)
                if len(common_endings) >= 20:
                    break
        
        return common_endings if common_endings else ['ta', 'này', 'kia', 'đây', 'thay']
    
    def _extract_themes_and_vocabulary(self) -> Tuple[Dict[str, List[str]], List[str]]:
        """Extract common themes and vocabulary from Truyện Kiều verses"""
        if not self.truyen_kieu_verses:
            # Fallback to predefined themes
            return {
                'love': ['tình', 'yêu', 'duyên', 'nhớ', 'thương'],
                'nature': ['trăng', 'mây', 'gió', 'hoa', 'cây'],
                'fate': ['số phận', 'duyên', 'kiếp', 'nghiệp', 'trời'],
                'beauty': ['đẹp', 'xinh', 'mỹ miều', 'lộng lẫy', 'hồng nhan'],
                'sadness': ['buồn', 'sầu', 'khổ', 'đau', 'thương']
            }, []
        
        # Create theme categories and collect vocabulary
        themes = {
            'love': [],
            'nature': [],
            'fate': [],
            'beauty': [],
            'sadness': [],
            'time': [],
            'virtue': []
        }
        
        # Map Vietnamese terms to themes
        theme_keywords = {
            'love': ['tình', 'yêu', 'duyên', 'nhớ', 'thương', 'ái', 'luyến'],
            'nature': ['trăng', 'mây', 'gió', 'hoa', 'cây', 'nước', 'sông', 'núi'],
            'fate': ['số', 'phận', 'duyên', 'kiếp', 'nghiệp', 'trời', 'mệnh'],
            'beauty': ['sắc', 'đẹp', 'xinh', 'hồng', 'nhan', 'mỹ', 'lệ'],
            'sadness': ['buồn', 'sầu', 'khổ', 'đau', 'thương', 'lệ', 'khóc'],
            'time': ['năm', 'tháng', 'ngày', 'xuân', 'thu', 'đông', 'hạ'],
            'virtue': ['hiếu', 'nghĩa', 'trung', 'trinh', 'lễ', 'tiết', 'đức']
        }
        
        # Collect all words
        all_words = []
        
        # Process each verse
        for verse in self.truyen_kieu_verses:
            words = re.findall(r'\b\w+\b', verse.lower())
            all_words.extend(words)
            
            # Assign words to themes
            for theme, keywords in theme_keywords.items():
                for keyword in keywords:
                    if keyword in verse.lower():
                        # Find surrounding context (nearby words)
                        for word in words:
                            if len(word) >= 2 and word not in themes[theme]:
                                themes[theme].append(word)
        
        # Get unique vocabulary
        vocabulary = list(set(all_words))
        vocabulary = [word for word in vocabulary if len(word) >= 2]
        
        return themes, vocabulary
    
    def count_vietnamese_syllables(self, text: str) -> int:
        """
        Count the number of Vietnamese syllables in a text string
        
        Args:
            text: Vietnamese text
            
        Returns:
            Number of syllables
        """
        # In Vietnamese, syllables are typically separated by spaces
        # This is a simplified approach - actual Vietnamese syllable counting may be more complex
        words = text.strip().split()
        return len(words)
    
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
    
    def check_luc_bat_pattern(self, verse: str) -> bool:
        """
        Check if a verse follows the lục bát pattern
        
        Args:
            verse: A Vietnamese verse
            
        Returns:
            True if the verse follows the lục bát pattern, False otherwise
        """
        lines = verse.strip().split('\n')
        
        # If there's only one line, check if it's a lục (6) or bát (8)
        if len(lines) == 1:
            words = lines[0].split()
            if len(words) == 6:
                return self._check_line_pattern(words, self.luc_bat_patterns['luc'])
            elif len(words) == 8:
                return self._check_line_pattern(words, self.luc_bat_patterns['bat'])
            return False
        
        # For multiple lines, check if they alternate between lục and bát
        if len(lines) < 2:
            return False
            
        for i, line in enumerate(lines):
            words = line.split()
            if i % 2 == 0:  # Even index (0, 2, 4...) should be lục (6 syllables)
                if len(words) != 6 or not self._check_line_pattern(words, self.luc_bat_patterns['luc']):
                    return False
            else:  # Odd index (1, 3, 5...) should be bát (8 syllables)
                if len(words) != 8 or not self._check_line_pattern(words, self.luc_bat_patterns['bat']):
                    return False
        
        return True
    
    def _check_line_pattern(self, words: List[str], pattern: List[Optional[str]]) -> bool:
        """
        Check if a line follows a specific tone pattern
        
        Args:
            words: List of words in the line
            pattern: The tone pattern to check against
            
        Returns:
            True if the line follows the pattern, False otherwise
        """
        if len(words) != len(pattern):
            return False
            
        for i, (word, expected_tone) in enumerate(zip(words, pattern)):
            if expected_tone is not None:  # Only check positions where tone matters
                actual_tone = self.get_syllable_tone(word)
                if actual_tone != expected_tone:
                    return False
        
        return True
    
    def generate_verse(self, initial_phrase: str = None, num_samples: int = 1, max_length: int = 100) -> List[str]:
        """
        Generate verses in the style of Truyện Kiều, following the lục bát pattern
        
        Args:
            initial_phrase: Optional starting phrase
            num_samples: Number of verses to generate
            max_length: Maximum length of each verse
            
        Returns:
            List of generated verses
        """
        # Include reference verses from the actual Truyện Kiều to guide the model
        reference_verses = self._select_reference_verses(initial_phrase)
        reference_text = "\n\n".join(reference_verses)
        
        prompt_template = """
        Hãy sáng tác thơ lục bát theo phong cách Truyện Kiều của Nguyễn Du. Thơ lục bát có cấu trúc câu 6 chữ xen kẽ với câu 8 chữ. Các quy tắc về vần, nhịp và cách gieo vần của thơ lục bát cần phải tuân thủ chặt chẽ.

        Quy tắc thơ lục bát:
        1. Câu lục: 6 chữ, chữ thứ 6 thanh bằng
        2. Câu bát: 8 chữ, chữ thứ 8 thanh bằng
        3. Vần: chữ cuối câu lục vần với chữ thứ 6 của câu bát kế tiếp
        4. Âm hưởng: giống phong cách Truyện Kiều, đảm bảo sự uyển chuyển, tinh tế

        Dưới đây là một số đoạn từ Truyện Kiều để tham khảo phong cách:

        {reference_text}

        {instruction}

        Bài thơ:
        """
        
        # Prepare the specific instruction
        if initial_phrase:
            instruction = f"Hãy sáng tác một bài thơ lục bát với khổ đầu bắt đầu bằng '{initial_phrase}'. Đảm bảo phong cách, cách gieo vần và âm hưởng giống Truyện Kiều của Nguyễn Du."
        else:
            instruction = "Hãy sáng tác một bài thơ lục bát với phong cách, cách gieo vần và âm hưởng giống Truyện Kiều của Nguyễn Du."
        
        # Format the prompt
        prompt = prompt_template.format(reference_text=reference_text, instruction=instruction)
        
        # Generate verses
        generated_verses = []
        
        for _ in range(num_samples):
            # Prepare inputs for the model
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate text
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode the generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the verse part (after "Bài thơ:")
            verse_match = re.search(r'Bài thơ:(.*?)(?:$|\.)', generated_text, re.DOTALL)
            if verse_match:
                verse = verse_match.group(1).strip()
            else:
                # Try to extract any lines that look like verses
                lines = generated_text.split('\n')
                verse_lines = []
                for line in lines:
                    # Look for lines that might be verses (after any prompt text)
                    if "Bài thơ:" in line:
                        verse_lines.append(line.split("Bài thơ:")[1].strip())
                    elif len(line.split()) in [6, 8] and not line.startswith("Hãy") and not line.startswith("Quy"):
                        verse_lines.append(line.strip())
                
                verse = '\n'.join(verse_lines)
            
            # Post-process to ensure proper structure
            verse = self.apply_poetic_constraints(verse)
            
            # Only add if we have something valid
            if verse:
                generated_verses.append(verse)
        
        return generated_verses
    
    def _select_reference_verses(self, initial_phrase: str = None) -> List[str]:
        """
        Select reference verses from Truyện Kiều that are relevant to the initial phrase
        
        Args:
            initial_phrase: Optional starting phrase to guide the selection
            
        Returns:
            List of reference verses
        """
        if not self.truyen_kieu_verses:
            return []  # No verses to select from
        
        # Number of reference verses to include
        num_references = min(5, len(self.truyen_kieu_verses))
        
        # If no initial phrase, select random high-quality verses
        if not initial_phrase:
            return random.sample(self.truyen_kieu_verses, num_references)
        
        # If there's an initial phrase, find verses that contain similar words
        initial_words = set(initial_phrase.lower().split())
        
        # Score verses based on similarity to initial phrase
        scored_verses = []
        for verse in self.truyen_kieu_verses:
            verse_words = set(verse.lower().split())
            score = len(verse_words.intersection(initial_words))
            scored_verses.append((verse, score))
        
        # Sort by score (descending) and get top matches
        scored_verses.sort(key=lambda x: x[1], reverse=True)
        
        # Get top matching verses, then add some random ones for diversity
        top_verses = [v[0] for v in scored_verses[:2] if v[1] > 0]
        
        # If we don't have enough matching verses, add some random ones
        remaining_needed = num_references - len(top_verses)
        if remaining_needed > 0:
            remaining_verses = [v for v in self.truyen_kieu_verses if v not in top_verses]
            random_verses = random.sample(remaining_verses, min(remaining_needed, len(remaining_verses)))
            top_verses.extend(random_verses)
        
        return top_verses
        
    def generate_verse_pair(self, initial_phrase: str = None) -> Tuple[str, str]:
        """
        Generate a pair of verses (lục-bát) that work together poetically
        
        Args:
            initial_phrase: Optional starting phrase for the first verse
            
        Returns:
            A tuple of two verses (lục and bát)
        """
        # Find similar verse pairs in Truyện Kiều for reference
        reference_pairs = self._select_verse_pairs(initial_phrase)
        reference_text = "\n\n".join(reference_pairs)
        
        prompt_template = """
        Hãy sáng tác một cặp câu thơ lục bát (một câu 6 chữ và một câu 8 chữ liên tiếp) theo phong cách Truyện Kiều của Nguyễn Du. 

        Quy tắc:
        1. Câu lục: 6 chữ, chữ thứ 6 thanh bằng
        2. Câu bát: 8 chữ, chữ thứ 8 thanh bằng
        3. Vần: chữ cuối câu lục vần với chữ thứ 6 của câu bát
        
        Dưới đây là một số cặp câu lục bát từ Truyện Kiều để tham khảo phong cách:
        
        {reference_text}
        
        {instruction}

        Cặp câu lục bát:
        """
        
        # Prepare the specific instruction
        if initial_phrase:
            instruction = f"Hãy sáng tác một cặp câu lục bát với câu đầu tiên bắt đầu bằng '{initial_phrase}'. Hai câu phải liên kết chặt chẽ về nghĩa và vần điệu, đúng phong cách Truyện Kiều."
        else:
            instruction = "Hãy sáng tác một cặp câu lục bát có nội dung liên kết chặt chẽ và vần điệu đúng chuẩn thơ lục bát, theo đúng phong cách Truyện Kiều."
        
        # Format the prompt
        prompt = prompt_template.format(reference_text=reference_text, instruction=instruction)
        
        # Generate verses
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the verse pair
        verse_match = re.search(r'Cặp câu lục bát:(.*?)(?:$|\.)', generated_text, re.DOTALL)
        if verse_match:
            verse_pair = verse_match.group(1).strip()
        else:
            lines = generated_text.split('\n')
            verse_lines = []
            for line in lines:
                if len(line.split()) in [6, 8] and not line.startswith("Hãy") and not line.startswith("Quy"):
                    verse_lines.append(line.strip())
            
            verse_pair = '\n'.join(verse_lines[:2])  # Just the first two lines
        
        # Split into two verses
        verses = verse_pair.split('\n')
        if len(verses) >= 2:
            luc_verse = verses[0].strip()
            bat_verse = verses[1].strip()
        else:
            # If we don't have clear line breaks, try to split based on syllable count
            words = verse_pair.split()
            if len(words) >= 14:  # Enough for both a lục and bát
                luc_verse = ' '.join(words[:6])
                bat_verse = ' '.join(words[6:14])
            else:
                # Fallback to finding a verse pair from the original text
                if self.truyen_kieu_verses and len(self.truyen_kieu_verses) > 0:
                    random_verse = random.choice(self.truyen_kieu_verses)
                    lines = random_verse.strip().split('\n')
                    if len(lines) >= 2:
                        luc_verse = lines[0]
                        bat_verse = lines[1]
                    else:
                        # Ultimate fallback
                        luc_verse = initial_phrase or "Trăm năm trong cõi người ta"
                        bat_verse = "Chữ tài chữ mệnh khéo là ghét nhau"
                else:
                    # Fallback if no verses loaded
                    luc_verse = initial_phrase or "Trăm năm trong cõi người ta"
                    bat_verse = "Chữ tài chữ mệnh khéo là ghét nhau"
        
        # Post-process to ensure proper structure
        luc_verse = self.apply_poetic_constraints(luc_verse, force_luc=True)
        bat_verse = self.apply_poetic_constraints(bat_verse, force_bat=True)
        
        return (luc_verse, bat_verse)
    
    def _select_verse_pairs(self, initial_phrase: str = None) -> List[str]:
        """
        Select verse pairs from Truyện Kiều that can serve as examples
        
        Args:
            initial_phrase: Optional starting phrase to find relevant pairs
            
        Returns:
            List of verse pairs (lục-bát)
        """
        if not self.truyen_kieu_verses:
            # No verses available, return empty list
            return []
        
        # Number of pairs to select
        num_pairs = min(3, len(self.truyen_kieu_verses))
        
        # If no initial phrase, select random pairs
        if not initial_phrase:
            return random.sample(self.truyen_kieu_verses, num_pairs)
        
        # If there's an initial phrase, find pairs with similar words
        initial_words = set(initial_phrase.lower().split())
        
        # Score pairs based on similarity to initial phrase
        scored_pairs = []
        for verse_pair in self.truyen_kieu_verses:
            pair_words = set(verse_pair.lower().split())
            score = len(pair_words.intersection(initial_words))
            scored_pairs.append((verse_pair, score))
        
        # Sort by score (descending)
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Get top matching pairs
        top_pairs = [p[0] for p in scored_pairs[:1] if p[1] > 0]
        
        # Add some random pairs for diversity
        remaining_needed = num_pairs - len(top_pairs)
        if remaining_needed > 0:
            remaining_pairs = [p for p in self.truyen_kieu_verses if p not in top_pairs]
            random_pairs = random.sample(remaining_pairs, min(remaining_needed, len(remaining_pairs)))
            top_pairs.extend(random_pairs)
        
        return top_pairs
    
    def generate_with_theme(self, theme: str, num_samples: int = 1) -> List[str]:
        """
        Generate verses related to a specific theme
        
        Args:
            theme: Theme for the verse
            num_samples: Number of verses to generate
            
        Returns:
            List of generated verses
        """
        # Dictionary mapping themes to Vietnamese translations/descriptions
        theme_mapping = {
            'love': 'tình yêu, tình cảm, duyên phận',
            'nature': 'thiên nhiên, phong cảnh, sông núi',
            'fate': 'số phận, duyên phận, kiếp người',
            'beauty': 'sắc đẹp, nhan sắc, hồng nhan',
            'sadness': 'buồn bã, sầu thảm, đau khổ',
            'happiness': 'hạnh phúc, niềm vui, an lạc',
            'loyalty': 'trung thành, chung thủy, đạo nghĩa',
            'regret': 'hối tiếc, nuối tiếc, ân hận'
        }
        
        # Get the Vietnamese version of the theme
        vietnamese_theme = theme_mapping.get(theme.lower(), theme)
        
        # Select themed examples from Truyện Kiều
        themed_examples = self._select_themed_verses(theme)
        examples_text = "\n\n".join(themed_examples)
        
        prompt_template = """
        Hãy sáng tác thơ lục bát theo phong cách Truyện Kiều của Nguyễn Du với chủ đề {theme}. Thơ lục bát có cấu trúc câu 6 chữ xen kẽ với câu 8 chữ.

        Quy tắc thơ lục bát:
        1. Câu lục: 6 chữ, chữ thứ 6 thanh bằng
        2. Câu bát: 8 chữ, chữ thứ 8 thanh bằng
        3. Vần: chữ cuối câu lục vần với chữ thứ 6 của câu bát kế tiếp
        4. Âm hưởng: phải giống phong cách Truyện Kiều, thể hiện sâu sắc về chủ đề {theme}
        
        Dưới đây là một số đoạn từ Truyện Kiều có chủ đề {theme} để tham khảo:
        
        {examples}

        Bài thơ:
        """
        
        # Format the prompt
        prompt = prompt_template.format(theme=vietnamese_theme, examples=examples_text)
        
        # Generate verses
        generated_verses = []
        
        for _ in range(num_samples):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the verse part
            verse_match = re.search(r'Bài thơ:(.*?)(?:$|\.)', generated_text, re.DOTALL)
            if verse_match:
                verse = verse_match.group(1).strip()
            else:
                # Try to extract verse lines
                lines = generated_text.split('\n')
                verse_lines = []
                for line in lines:
                    if len(line.split()) in [6, 8] and not line.startswith("Hãy") and not line.startswith("Quy"):
                        verse_lines.append(line.strip())
                
                verse = '\n'.join(verse_lines)
            
            # Post-process to ensure proper structure
            verse = self.apply_poetic_constraints(verse)
            
            if verse:
                generated_verses.append(verse)
        
        return generated_verses
    
    def _select_themed_verses(self, theme: str) -> List[str]:
        """
        Select verses from Truyện Kiều that match a given theme
        
        Args:
            theme: Theme to match
            
        Returns:
            List of thematically relevant verses
        """
        if not self.truyen_kieu_verses:
            return []
        
        # Convert theme to lower case for matching
        theme_lower = theme.lower()
        
        # Try to find theme in our extracted themes
        theme_words = []
        for t, words in self.common_themes.items():
            if theme_lower in t or t in theme_lower:
                theme_words.extend(words)
        
        # If we found theme words, use them to score verses
        if theme_words:
            # Score verses based on theme relevance
            scored_verses = []
            for verse in self.truyen_kieu_verses:
                verse_lower = verse.lower()
                # Count number of theme words in the verse
                score = sum(1 for word in theme_words if word in verse_lower)
                scored_verses.append((verse, score))
            
            # Sort by score and get top matches
            scored_verses.sort(key=lambda x: x[1], reverse=True)
            
            # Get top matching verses
            top_verses = [v[0] for v in scored_verses[:3] if v[1] > 0]
            
            # If we found enough theme-related verses, return them
            if len(top_verses) >= 2:
                return top_verses
        
        # If we didn't find enough themed verses, search for keyword matches
        keyword_matches = []
        
        # Dictionary mapping themes to keywords
        theme_keywords = {
            'love': ['tình', 'yêu', 'thương', 'nhớ', 'duyên'],
            'nature': ['trăng', 'mây', 'gió', 'hoa', 'cây'],
            'fate': ['số', 'phận', 'trời', 'định', 'kiếp'],
            'beauty': ['sắc', 'đẹp', 'xinh', 'hồng', 'nhan'],
            'sadness': ['buồn', 'sầu', 'đau', 'khổ', 'lệ'],
            'happiness': ['vui', 'mừng', 'hạnh', 'phúc', 'sướng'],
            'loyalty': ['trung', 'thành', 'nghĩa', 'đạo', 'thủy'],
            'regret': ['tiếc', 'hối', 'ân', 'hận', 'lỗi']
        }
        
        # Get keywords for the theme
        keywords = []
        for t, words in theme_keywords.items():
            if theme_lower in t or t in theme_lower:
                keywords.extend(words)
        
        # If no explicit mapping, use the theme itself as keyword
        if not keywords:
            keywords = [theme_lower]
        
        # Find verses containing the keywords
        for verse in self.truyen_kieu_verses:
            verse_lower = verse.lower()
            if any(keyword in verse_lower for keyword in keywords):
                keyword_matches.append(verse)
        
        # Limit to 3 examples
        keyword_matches = keyword_matches[:3]
        
        # If we still don't have enough, add some random verses
        if len(keyword_matches) < 3 and self.truyen_kieu_verses:
            remaining_needed = 3 - len(keyword_matches)
            random_verses = random.sample(self.truyen_kieu_verses, min(remaining_needed, len(self.truyen_kieu_verses)))
            keyword_matches.extend(random_verses)
        
        return keyword_matches
    
    def apply_poetic_constraints(self, verse: str, force_luc: bool = False, force_bat: bool = False) -> str:
        """
        Apply Vietnamese poetic structure constraints to the verse
        
        Args:
            verse: Raw generated verse
            force_luc: Force the verse to be a lục (6 syllables)
            force_bat: Force the verse to be a bát (8 syllables)
            
        Returns:
            Refined verse that better follows the Truyện Kiều structure
        """
        # 1. Clean up the verse
        verse = verse.strip()
        if not verse:
            return ""
        
        # 2. Split into lines
        lines = verse.split('\n')
        refined_lines = []
        
        # Process each line
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Remove line numbers if present
            line = re.sub(r'^\d+\.\s*', '', line)
            
            # Split into words/syllables
            words = line.split()
            
            if force_luc or (not force_bat and (i % 2 == 0 or len(words) <= 6)):
                # This should be a lục (6-syllable) line
                target_length = 6
                pattern = self.luc_bat_patterns['luc']
            else:
                # This should be a bát (8-syllable) line
                target_length = 8
                pattern = self.luc_bat_patterns['bat']
            
            # Adjust to target length
            if len(words) > target_length:
                words = words[:target_length]
            elif len(words) < target_length:
                # For short lines, try to pad with common words to reach target length
                padding_needed = target_length - len(words)
                if i % 2 == 0:  # lục line
                    padding = ["cho", "này", "mà", "thì"][:padding_needed]
                else:  # bát line
                    padding = ["còn", "mà", "chăng", "thay"][:padding_needed]
                
                words.extend(padding)
            
            # Ensure the last syllable has a flat tone as per lục bát rules
            if words:
                last_word = words[-1]
                if self.get_syllable_tone(last_word) != 'flat':
                    # Replace with a common ending with flat tone
                    words[-1] = np.random.choice(self.common_endings)
            
            # Add to refined lines
            refined_lines.append(' '.join(words))
        
        # If we have no lines after processing, return a placeholder verse
        if not refined_lines:
            if force_luc:
                return "Trăm năm trong cõi người ta"
            elif force_bat:
                return "Chữ tài chữ mệnh khéo là ghét nhau"
            else:
                return "Trăm năm trong cõi người ta,\nChữ tài chữ mệnh khéo là ghét nhau."
        
        # 3. Ensure proper ending punctuation for each line
        for i in range(len(refined_lines)):
            if i < len(refined_lines) - 1:
                # All lines except the last should end with a comma
                if not refined_lines[i].endswith((',', '.', '?', '!')):
                    refined_lines[i] += ','
            else:
                # Last line should end with a period
                if not refined_lines[i].endswith(('.', '?', '!')):
                    refined_lines[i] += '.'
        
        # Join the lines back together
        return '\n'.join(refined_lines)
    
    def evaluate_verse_quality(self, verse: str) -> Dict[str, float]:
        """
        Evaluate the quality of a generated verse using various metrics
        
        Args:
            verse: The verse to evaluate
            
        Returns:
            Dictionary with quality metrics
        """
        # Initialize scores
        scores = {
            'length': 0.0,       # Length appropriateness
            'structure': 0.0,    # Vietnamese poetic structure
            'rhyme': 0.0,        # Rhyming quality
            'tone_pattern': 0.0  # Vietnamese tone pattern adherence
        }
        
        # Split into lines
        lines = verse.strip().split('\n')
        
        # 1. Evaluate length appropriateness
        if len(lines) < 2:
            scores['length'] = 0.3  # Too short for a proper verse
        elif len(lines) % 2 == 1:
            scores['length'] = 0.7  # Odd number of lines might not be ideal for lục bát
        else:
            scores['length'] = 1.0  # Even number of lines is good for lục bát
        
        # 2. Evaluate structure
        structure_score = 0.0
        valid_luc_bat_pairs = 0
        
        for i in range(0, len(lines) - 1, 2):
            if i + 1 < len(lines):
                luc_line = lines[i].split()
                bat_line = lines[i + 1].split()
                
                # Check if we have a valid lục-bát pair
                if len(luc_line) == 6 and len(bat_line) == 8:
                    valid_luc_bat_pairs += 1
                    
                    # Check tone pattern
                    luc_valid = self._check_line_pattern(luc_line, self.luc_bat_patterns['luc'])
                    bat_valid = self._check_line_pattern(bat_line, self.luc_bat_patterns['bat'])
                    
                    if luc_valid and bat_valid:
                        structure_score += 1.0
                    elif luc_valid or bat_valid:
                        structure_score += 0.5
        
        if valid_luc_bat_pairs > 0:
            scores['structure'] = structure_score / valid_luc_bat_pairs
        
        # 3. Evaluate rhyming
        rhyme_score = 0.0
        rhyme_pairs = 0
        
        for i in range(0, len(lines) - 1, 2):
            if i + 1 < len(lines):
                luc_line = lines[i].split()
                bat_line = lines[i + 1].split()
                
                if len(luc_line) >= 6 and len(bat_line) >= 6:
                    # In lục bát, the 6th syllable of the lục line should rhyme with 
                    # the 6th syllable of the bát line
                    luc_last = luc_line[5] if len(luc_line) > 5 else ""
                    bat_sixth = bat_line[5] if len(bat_line) > 5 else ""
                    
                    # Simple rhyme check: same ending sounds
                    # This is a simplified approach - a proper Vietnamese rhyme checker would be more complex
                    if luc_last and bat_sixth:
                        # Extract the final vowel sounds (simplified approach)
                        luc_vowel = re.sub(r'[bcdfghjklmnpqrstvwxyz]$', '', luc_last)
                        bat_vowel = re.sub(r'[bcdfghjklmnpqrstvwxyz]$', '', bat_sixth)
                        
                        if luc_vowel == bat_vowel:
                            rhyme_score += 1.0
                        elif self._similar_vowel_sounds(luc_vowel, bat_vowel):
                            rhyme_score += 0.5
                            
                        rhyme_pairs += 1
        
        if rhyme_pairs > 0:
            scores['rhyme'] = rhyme_score / rhyme_pairs
        
        # 4. Evaluate tone pattern adherence
        tone_score = 0.0
        tone_checks = 0
        
        for i, line in enumerate(lines):
            words = line.split()
            
            if i % 2 == 0:  # lục line
                pattern = self.luc_bat_patterns['luc']
                if len(words) == 6:
                    # Check critical positions: 2, 4, 6
                    positions = [1, 3, 5]  # 0-indexed
                    expected_tones = [pattern[pos] for pos in positions if pos < len(pattern)]
                    actual_tones = [self.get_syllable_tone(words[pos]) for pos in positions if pos < len(words)]
                    
                    matches = sum(1 for exp, act in zip(expected_tones, actual_tones) 
                                 if exp is None or exp == act)
                    
                    if expected_tones:
                        tone_score += matches / len(expected_tones)
                        tone_checks += 1
            else:  # bát line
                pattern = self.luc_bat_patterns['bat']
                if len(words) == 8:
                    # Check critical positions: 2, 4, 6, 8
                    positions = [1, 3, 5, 7]  # 0-indexed
                    expected_tones = [pattern[pos] for pos in positions if pos < len(pattern)]
                    actual_tones = [self.get_syllable_tone(words[pos]) for pos in positions if pos < len(words)]
                    
                    matches = sum(1 for exp, act in zip(expected_tones, actual_tones) 
                                 if exp is None or exp == act)
                    
                    if expected_tones:
                        tone_score += matches / len(expected_tones)
                        tone_checks += 1
        
        if tone_checks > 0:
            scores['tone_pattern'] = tone_score / tone_checks
        
        # Calculate overall score
        scores['overall'] = (scores['length'] + scores['structure'] + 
                           scores['rhyme'] + scores['tone_pattern']) / 4
        
        return scores
    
    def _similar_vowel_sounds(self, vowel1: str, vowel2: str) -> bool:
        """Check if two vowel sounds are similar for rhyming purposes"""
        # Define groups of similar-sounding vowels in Vietnamese
        similar_vowels = [
            {'a', 'à', 'á', 'ả', 'ã', 'ạ'},
            {'ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ'},
            {'â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ'},
            {'e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ'},
            {'ê', 'ề', 'ế', 'ể', 'ễ', 'ệ'},
            {'i', 'ì', 'í', 'ỉ', 'ĩ', 'ị', 'y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ'},
            {'o', 'ò', 'ó', 'ỏ', 'õ', 'ọ'},
            {'ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ'},
            {'ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ'},
            {'u', 'ù', 'ú', 'ủ', 'ũ', 'ụ'},
            {'ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự'}
        ]
        
        # Check if the vowels are in the same group
        for group in similar_vowels:
            vowel1_in_group = any(char in group for char in vowel1)
            vowel2_in_group = any(char in group for char in vowel2)
            
            if vowel1_in_group and vowel2_in_group:
                return True
                
        return False
    
    def __del__(self):
        """Clean up resources when the object is deleted"""
        # Free GPU memory if using CUDA
        if hasattr(self, 'model') and self.device == "cuda":
            del self.model
            torch.cuda.empty_cache()