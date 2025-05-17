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
    
    def apply_poetic_constraints(self, verse: str, force_luc: bool = False, force_bat: bool = False, force_luc_bat: bool = False) -> str:
        """Apply Vietnamese poetic structure constraints to verse, ensuring lục bát pattern"""
        # 1. Clean up the verse
        verse = verse.strip()
        if not verse:
            return ""
        
        # 2. Split into lines
        lines = verse.split('\n')
        # Handle force_luc_bat option
        if force_luc_bat:
            # Ensure strict lục bát pattern (6-8-6-8...)
            cleaned_lines = []
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Remove line numbers if present
                line = re.sub(r'^\d+\.\s*', '', line)
                
                # Process the line
                words = line.split()
                target_length = 6 if i % 2 == 0 else 8  # Alternate 6-8
                
                # Adjust length
                if len(words) > target_length:
                    words = words[:target_length]
                elif len(words) < target_length:
                    padding_needed = target_length - len(words)
                    if i % 2 == 0:  # lục line
                        padding = ["trong đời", "xưa nay", "này đây", "mà thôi"][:padding_needed]
                    else:  # bát line
                        padding = ["trên đời này đấy", "mà người đâu hay", "cho lòng nhẹ tênh", "biết làm sao đây"][:padding_needed]
                    words.extend(padding[:padding_needed])
                
                # Ensure proper tone endings
                if words and self.get_syllable_tone(words[-1]) != 'flat':
                    words[-1] = self.common_endings[i % len(self.common_endings)]
                
                cleaned_lines.append(' '.join(words))
            
            # Replace the original lines with the cleaned ones
            lines = cleaned_lines
        refined_lines = []
        
        # Process each line, enforcing alternating 6-8 pattern
        for i in range(0, len(lines) + (0 if len(lines) % 2 == 0 else 1)):  # Ensure even number of lines
            if i >= len(lines):
                # Add an additional line if needed to complete the pattern
                line = ""
            else:
                line = lines[i].strip()
                if not line:
                    continue
            
            # Remove line numbers if present
            line = re.sub(r'^\d+\.\s*', '', line)
            
            # Split into words/syllables
            words = line.split()
            
            # Strict lục bát pattern: Even lines (0, 2, 4...) have 6 syllables, odd lines have 8
            if i % 2 == 0:  # lục line (6 syllables)
                target_length = 6
                pattern = self.luc_bat_patterns['luc']
            else:  # bát line (8 syllables)
                target_length = 8
                pattern = self.luc_bat_patterns['bat']
            
            # Adjust to target length
            if len(words) > target_length:
                words = words[:target_length]
            elif len(words) < target_length:
                # For short lines, pad with common words to reach target length
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
        
        # Group lines in lục-bát pairs
        luc_bat_pairs = []
        for i in range(0, len(refined_lines), 2):
            if i + 1 < len(refined_lines):
                luc_line = refined_lines[i]
                bat_line = refined_lines[i + 1]
                
                # Add proper punctuation
                luc_line = luc_line.rstrip(',.:;!?') + ','
                bat_line = bat_line.rstrip(',.:;!?') + '.'
                
                luc_bat_pairs.append(f"{luc_line}\n{bat_line}")
        
        # Join the pairs with double newlines
        result = '\n'.join(luc_bat_pairs)
        
        return result
    
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
        """Generate verses in the style of Truyện Kiều, following the lục bát pattern"""
        # Include reference verses with examples of proper lục bát
        reference_verses = self._select_reference_verses(initial_phrase)
        reference_text = "\n\n".join(reference_verses)
        
        # Create a more specific prompt that emphasizes using the initial phrase
        if initial_phrase:
            # Using a more directive prompt
            prompt_template = f"""
            Hãy sáng tác một bài thơ lục bát trong phong cách Truyện Kiều của Nguyễn Du.

            YÊU CẦU NGHIÊM NGẶT:
            1. BẮT ĐẦU bài thơ với "{initial_phrase}" (chữ đầu tiên phải là "{initial_phrase.split()[0]}")
            2. Dòng đầu PHẢI có đúng 6 chữ
            3. Dòng thứ hai PHẢI có đúng 8 chữ
            4. Tiếp tục mô hình 6 chữ - 8 chữ cho các dòng tiếp theo
            5. KHÔNG sử dụng các từ đệm như "còn mà chăng thay"
            6. Giữ tính mạch lạc về chủ đề xuyên suốt bài thơ

            Dưới đây là ví dụ từ Truyện Kiều để học phong cách:
            {reference_text}

            Bài thơ (BẮT ĐẦU với "{initial_phrase}"):
            {initial_phrase}
            """
        else:
            # Default prompt
            prompt_template = """
            Hãy sáng tác một bài thơ lục bát trong phong cách Truyện Kiều của Nguyễn Du.

            YÊU CẦU NGHIÊM NGẶT:
            1. Dòng đầu PHẢI có đúng 6 chữ
            2. Dòng thứ hai PHẢI có đúng 8 chữ
            3. Tiếp tục mô hình 6 chữ - 8 chữ cho các dòng tiếp theo
            4. KHÔNG sử dụng các từ đệm như "còn mà chăng thay"
            5. Giữ tính mạch lạc về chủ đề xuyên suốt bài thơ

            Dưới đây là ví dụ từ Truyện Kiều để học phong cách:
            {reference_text}

            Bài thơ:
            """
        
        # Format the prompt
        prompt = prompt_template.format(reference_text=reference_text)
        
        # Generate verses
        generated_verses = []
        
        for _ in range(num_samples):
            # Generate text using Ollama with a higher temperature for creativity
            generated_text = self.ollama_generate(prompt, max_tokens=max_length, temperature=0.9)
            
            # Process the generated text
            processed_text = self._process_generated_text(generated_text, initial_phrase)
            
            # Apply lục bát structure enforcement
            verse = self._enforce_luc_bat_structure(processed_text, initial_phrase)
            
            # Only add if we have something valid
            if verse:
                generated_verses.append(verse)
        
        return generated_verses

    def _process_generated_text(self, text: str, initial_phrase: str = None) -> str:
        """Process the generated text to extract the verse"""
        lines = []
        capturing = False
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Start capturing if we see the initial phrase or the typical indicators
            if initial_phrase and initial_phrase in line:
                capturing = True
            elif "Bài thơ:" in line:
                capturing = True
                line = line.replace("Bài thơ:", "").strip()
                
            if capturing:
                # Skip lines that are obviously not verses
                if "YÊU CẦU" in line or "NGHIÊM NGẶT" in line:
                    continue
                    
                # Only include lines that look like verses
                words = line.split()
                if 3 <= len(words) <= 10:  # Reasonable length for a verse line
                    lines.append(line)
        
        return '\n'.join(lines)
    

    def _enforce_luc_bat_structure(self, verse: str, initial_phrase: str = None) -> str:
        """Enforce strict lục bát structure on a verse"""
        lines = verse.split('\n')
        formatted_lines = []
        
        # Always start with the initial phrase if provided
        if initial_phrase:
            # Create proper first line with initial phrase
            words = initial_phrase.split()
            if len(words) < 6:
                # Pad to 6 syllables
                if len(words) == 1:
                    words.extend(["bao", "thuở", "chốn", "trần", "gian"])
                elif len(words) == 2:
                    words.extend(["chốn", "trần", "gian", "này"])
                else:
                    words.extend(["trần", "gian", "này"][:6-len(words)])
            else:
                words = words[:6]  # Trim if too long
                
            formatted_lines.append(' '.join(words))
        
        # Add more lines from the generated verse
        for i, line in enumerate(lines):
            # Skip empty lines and the first line if we've already created it
            if not line.strip() or (i == 0 and initial_phrase and len(formatted_lines) > 0):
                continue
            
            # If it's our first line and we haven't added one yet, make sure to use initial_phrase
            if i == 0 and len(formatted_lines) == 0 and initial_phrase:
                words = initial_phrase.split()
                # Complete to 6 syllables
                if len(words) < 6:
                    words.extend(["bao", "thuở", "trần", "gian"][:6-len(words)])
                formatted_lines.append(' '.join(words[:6]))  # Ensure exactly 6 syllables
                continue
                
            # Process the line based on lục bát pattern
            words = line.split()
            target_length = 6 if len(formatted_lines) % 2 == 0 else 8
            
            # Skip if the line would be the same as the last added line
            if formatted_lines and ' '.join(words[:min(len(words), len(formatted_lines[-1].split()))]) == \
            ' '.join(formatted_lines[-1].split()[:min(len(words), len(formatted_lines[-1].split()))]):
                continue
            
            # Adjust length
            if len(words) > target_length:
                words = words[:target_length]
            elif len(words) < target_length:
                if target_length == 6:  # lục line
                    words.extend(["tình", "duyên", "nợ", "kiếp", "trần", "gian"][:target_length-len(words)])
                else:  # bát line
                    words.extend(["dường", "như", "duyên", "số", "đã", "an", "bài", "rồi"][:target_length-len(words)])
            
            # Make sure the line ends with flat tone
            if self.get_syllable_tone(words[-1]) != 'flat':
                if target_length == 6:
                    words[-1] = "ta"
                else:  # bát line
                    words[-1] = "rồi"
            
            formatted_lines.append(' '.join(words))
        
        # Make sure we have at least a lục-bát pair
        if len(formatted_lines) < 2:
            if len(formatted_lines) == 1:
                # Add a matching bát line
                formatted_lines.append("Duyên tình dường đã sắp bài từ lâu")
            else:
                # Should never happen if initial_phrase is provided, but just in case
                formatted_lines = [
                    initial_phrase if initial_phrase else "Trăm năm trong cõi người ta",
                    "Duyên tình dường đã sắp bài từ lâu"
                ]
        
        # Keep only even number of lines (complete lục-bát pairs)
        if len(formatted_lines) % 2 != 0:
            # Add one more line to complete the pair
            if len(formatted_lines[-1].split()) == 6:  # Last is lục
                formatted_lines.append("Duyên tình dường đã sắp bài từ lâu")
            else:  # Last is bát
                formatted_lines.append("Hồng nhan đa truân ta")
        
        # Add proper punctuation
        for i in range(len(formatted_lines)):
            if i < len(formatted_lines) - 1:
                formatted_lines[i] = formatted_lines[i].rstrip('.,:;!?') + ','
            else:
                formatted_lines[i] = formatted_lines[i].rstrip('.,:;!?') + '.'
        
        return '\n'.join(formatted_lines)
    


