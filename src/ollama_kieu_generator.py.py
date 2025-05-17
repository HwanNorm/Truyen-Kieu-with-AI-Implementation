import os
import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import random
import requests

class OllamaKieuGenerator:
    """Vietnamese verse generator using Ollama's LLaMA for Truyện Kiều style poetry"""
    
    def __init__(self, model_name: str = "llama3", 
                 truyen_kieu_path: str = "data/truyen_kieu.txt",
                 ollama_base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url.rstrip('/')
        
        print(f"Using Ollama model: {model_name}")
        
        # Test connection to Ollama
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model.get("name") for model in models]
                
                if self.model_name in model_names:
                    print(f"Found {self.model_name} in Ollama models")
                else:
                    print(f"Model {self.model_name} not found. You may need to pull it: ollama pull {self.model_name}")
            else:
                print(f"Could not connect to Ollama API: {response.status_code}")
        except Exception as e:
            print(f"Error checking Ollama connection: {e}")
            print("Please make sure Ollama is running on your system")
        
        # Load Truyện Kiều data
        self.truyen_kieu_verses = self.load_truyen_kieu(truyen_kieu_path)
        print(f"Loaded {len(self.truyen_kieu_verses)} verses from Truyện Kiều")
        
        # Initialize Vietnamese language resources
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
        
        # Common ending words
        self.common_endings = ['ta', 'này', 'kia', 'đây', 'chi', 'sao', 'nào', 'thay', 
                              'chăng', 'vay', 'thôi', 'rồi', 'ra', 'vào', 
                              'ai', 'người', 'rằng', 'đà']
        
        # Patterns for lục bát structure
        self.luc_bat_patterns = {
            'luc': [None, 'flat', None, 'sharp', None, 'flat'],  # 6-syllable line
            'bat': [None, 'flat', None, 'flat', None, 'sharp', None, 'flat']  # 8-syllable line
        }
    
    def ollama_generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate text using Ollama API"""
        api_url = f"{self.ollama_base_url}/api/generate"
        
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "top_k": 50,
                "num_predict": max_tokens
            }
        }
        
        try:
            response = requests.post(api_url, json=data)
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                print(f"Ollama API error: {response.status_code}")
                print(response.text)
                return ""
                
        except Exception as e:
            print(f"Error calling Ollama API: {e}")
            return ""
    
    def load_truyen_kieu(self, filepath: str) -> List[str]:
        """Load Truyện Kiều verses from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            
            verses = []
            lines = text.strip().split('\n')
            
            # Process lines
            current_verse = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                line = re.sub(r'^\d+\.\s*', '', line)  # Remove line numbers
                current_verse.append(line)
                
                # If we have a complete lục-bát pair, save it
                if len(current_verse) >= 2:
                    if len(current_verse) % 2 == 0:  # Even number of lines
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
        
    def get_syllable_tone(self, syllable: str) -> str:
        """Determine tone group (flat or sharp) of a Vietnamese syllable"""
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
    
    def _check_line_pattern(self, words: List[str], pattern: List[Optional[str]]) -> bool:
        """Check if a line follows a specific tone pattern"""
        if len(words) != len(pattern):
            return False
            
        for i, (word, expected_tone) in enumerate(zip(words, pattern)):
            if expected_tone is not None:  # Only check positions where tone matters
                actual_tone = self.get_syllable_tone(word)
                if actual_tone != expected_tone:
                    return False
        
        return True
    
    def apply_poetic_constraints(self, verse: str, force_luc: bool = False, force_bat: bool = False) -> str:
        """Apply Vietnamese poetic structure constraints to verse"""
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
    
    def _select_reference_verses(self, initial_phrase: str = None) -> List[str]:
        """Select reference verses from Truyện Kiều for the initial phrase"""
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
        
    def generate_verse(self, initial_phrase: str = None, num_samples: int = 1, max_length: int = 100) -> List[str]:
        """Generate verses in the style of Truyện Kiều"""
        # Include reference verses from the actual Truyện Kiều
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
            # Generate text using Ollama
            generated_text = self.ollama_generate(prompt, max_tokens=max_length)
            
            # Extract just the verse part (after "Bài thơ:")
            verse_match = re.search(r'Bài thơ:(.*?)(?:$|\.)', generated_text, re.DOTALL)
            if verse_match:
                verse = verse_match.group(1).strip()
            else:
                # Try to extract any lines that look like verses
                lines = generated_text.split('\n')
                verse_lines = []
                for line in lines:
                    # Look for lines that might be verses
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
    


