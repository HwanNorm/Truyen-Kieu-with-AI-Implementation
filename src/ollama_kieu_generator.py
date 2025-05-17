import os
import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import random
import requests
import time

class OllamaKieuGenerator:
    """Vietnamese verse generator using Ollama's LLaMA for Truyện Kiều style poetry"""
    
    def __init__(self, model_name: str = "llama3", 
                 truyen_kieu_path: str = "data/truyen_kieu.txt",
                 ollama_base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url.rstrip('/')
        
        print(f"Using Ollama model: {model_name}")
        self.model_available = False
        
        # Test connection to Ollama and check model availability
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model.get("name") for model in models]
                
                if self.model_name in model_names:
                    print(f"Found {self.model_name} in Ollama models")
                    self.model_available = True
                else:
                    # Try with "latest" tag
                    full_model_name = f"{self.model_name}:latest"
                    if full_model_name in model_names:
                        print(f"Found {full_model_name} in Ollama models")
                        self.model_name = full_model_name
                        self.model_available = True
                    else:
                        print(f"Model {self.model_name} not found. You may need to pull it: ollama pull {self.model_name}")
                        # Try to use an available model as fallback
                        if models:
                            fallback_model = models[0].get("name")
                            print(f"Using {fallback_model} as a fallback model")
                            self.model_name = fallback_model
                            self.model_available = True
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
        
        # Extract real lục-bát pairs from Truyện Kiều for authentic fallbacks
        self.luc_bat_pairs = self._extract_luc_bat_pairs()
        
        # Load common ending words with flat tone
        self.common_endings = self._extract_common_endings() or [
            'ta', 'này', 'kia', 'đây', 'chi', 'sao', 'nào', 'thay', 
            'chăng', 'vay', 'thôi', 'rồi', 'ra', 'vào', 
            'ai', 'người', 'rằng', 'đà'
        ]
        
        # Additional words for padding lục lines (6 syllables)
        self.luc_line_padding = [
            ["trần", "gian"], 
            ["đời", "người"], 
            ["bể", "khổ"], 
            ["kiếp", "này"],
            ["duyên", "phận"],
            ["trần", "ai"],
            ["nhân", "gian"],
            ["duyên", "nợ"]
        ]
        
        # Additional words for padding bát lines (8 syllables)
        self.bat_line_padding = [
            ["duyên", "phận", "đã", "an", "bài"],
            ["kiếp", "người", "như", "thế", "thôi"],
            ["tình", "duyên", "trắc", "trở", "lắm"],
            ["hồng", "nhan", "bạc", "phận", "đấy"],
            ["chữ", "tình", "chữ", "hiếu", "sao"]
        ]
        
        # Patterns for lục bát structure
        self.luc_bat_patterns = {
            'luc': [None, 'flat', None, 'sharp', None, 'flat'],  # 6-syllable line
            'bat': [None, 'flat', None, 'flat', None, 'sharp', None, 'flat']  # 8-syllable line
        }
        
        # Complete bát lines (8 syllables) for fallback
        self.bat_line_templates = [
            "Duyên tình dường đã sắp bài từ lâu",
            "Bao nhiêu thương nhớ biết đâu mà cùng",
            "Tơ duyên đã se từ trời xa xăm",
            "Sắc tài phận bạc lỗi lầm xiết bao",
            "Chữ tình chữ hiếu vẹn trao một lòng",
            "Trăm năm biết có duyên cùng ai chăng",
            "Bến mê sóng vỗ thuyền trăng lỡ làng"
        ]
        
        # Complete lục lines (6 syllables) for fallback
        self.luc_line_templates = [
            "Trăm năm trong cõi người ta",
            "Cuộc đời dâu bể nhạt nhòa",
            "Duyên kia hồng phận mặn mà",
            "Hồng nhan bạc mệnh xót xa",
            "Đời người mấy chốc qua đi",
            "Trăng sao vằng vặc đêm khuya",
            "Chữ tình chữ hiếu khó bề"
        ]
    
    def _extract_luc_bat_pairs(self) -> List[Tuple[str, str]]:
        """Extract actual lục-bát pairs from Truyện Kiều for authentic verse structure"""
        pairs = []
        
        for verse in self.truyen_kieu_verses:
            lines = verse.strip().split('\n')
            if len(lines) >= 2:
                luc_line = lines[0].strip()
                bat_line = lines[1].strip()
                
                # Verify that they have the correct syllable counts
                luc_words = luc_line.split()
                bat_words = bat_line.split()
                
                if len(luc_words) == 6 and len(bat_words) == 8:
                    # Remove any punctuation
                    luc_line = luc_line.rstrip(',.;:!?')
                    bat_line = bat_line.rstrip(',.;:!?')
                    pairs.append((luc_line, bat_line))
        
        return pairs
    
    def _extract_common_endings(self) -> List[str]:
        """Extract common ending words with flat tone from the verses"""
        if not self.truyen_kieu_verses:
            return []
            
        # Find all words that end a line
        ending_words = []
        for verse in self.truyen_kieu_verses:
            lines = verse.strip().split('\n')
            for line in lines:
                words = line.strip().rstrip(',.;:!?').split()
                if words:
                    ending_words.append(words[-1])
        
        # Count frequencies
        from collections import Counter
        counter = Counter(ending_words)
        
        # Get common flat-tone endings
        common_endings = []
        for word, count in counter.most_common(50):
            if self.get_syllable_tone(word) == 'flat':
                common_endings.append(word)
                if len(common_endings) >= 25:
                    break
        
        return common_endings
    
    def ollama_generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, 
                       num_retries: int = 3, retry_delay: float = 2.0) -> str:
        """Generate text using Ollama API with retries"""
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
        
        for attempt in range(num_retries):
            try:
                response = requests.post(api_url, json=data, timeout=30)
                
                if response.status_code == 200:
                    return response.json().get("response", "")
                else:
                    print(f"Ollama API error (attempt {attempt+1}/{num_retries}): {response.status_code}")
                    print(response.text)
                    if attempt < num_retries - 1:
                        time.sleep(retry_delay)
                    
            except Exception as e:
                print(f"Error calling Ollama API (attempt {attempt+1}/{num_retries}): {e}")
                if attempt < num_retries - 1:
                    time.sleep(retry_delay)
        
        # If all retries failed, return empty string
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
    
    def count_vietnamese_syllables(self, text: str) -> int:
        """
        Count Vietnamese syllables in text
        In Vietnamese, words are separated by spaces and each word typically represents one syllable
        """
        # Remove punctuation and split by whitespace
        text = re.sub(r'[^\w\s]', '', text)
        words = text.strip().split()
        return len(words)
    
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
            2. Dòng đầu PHẢI có đúng 6 chữ (không nhiều hơn, không ít hơn)
            3. Dòng thứ hai PHẢI có đúng 8 chữ (không nhiều hơn, không ít hơn)
            4. Tiếp tục mô hình 6 chữ - 8 chữ cho các dòng tiếp theo
            5. Chủ đề phải liên quan đến: tình duyên, số phận, sắc đẹp, tài năng, hoặc trăm năm trong cõi người ta
            6. Giữ tính mạch lạc về chủ đề xuyên suốt bài thơ
            7. Mỗi dòng phải khác nhau, KHÔNG lặp lại câu đã viết
            8. KHÔNG sử dụng dấu phẩy (,) ở giữa dòng, chỉ đặt dấu phẩy cuối dòng

            Dưới đây là ví dụ từ Truyện Kiều để học phong cách:
            {reference_text}

            Bài thơ (BẮT ĐẦU với "{initial_phrase}" và theo cấu trúc 6-8 chữ mỗi dòng):
            """
        else:
            # Default prompt
            prompt_template = """
            Hãy sáng tác một bài thơ lục bát trong phong cách Truyện Kiều của Nguyễn Du.

            YÊU CẦU NGHIÊM NGẶT:
            1. Dòng đầu PHẢI có đúng 6 chữ (không nhiều hơn, không ít hơn)
            2. Dòng thứ hai PHẢI có đúng 8 chữ (không nhiều hơn, không ít hơn)
            3. Tiếp tục mô hình 6 chữ - 8 chữ cho các dòng tiếp theo
            4. Chủ đề phải liên quan đến: tình duyên, số phận, sắc đẹp, tài năng, hoặc trăm năm trong cõi người ta
            5. Giữ tính mạch lạc về chủ đề xuyên suốt bài thơ
            6. Mỗi dòng phải khác nhau, KHÔNG lặp lại câu đã viết
            7. KHÔNG sử dụng dấu phẩy (,) ở giữa dòng, chỉ đặt dấu phẩy cuối dòng

            Dưới đây là ví dụ từ Truyện Kiều để học phong cách:
            {reference_text}

            Bài thơ (theo cấu trúc 6-8 chữ mỗi dòng):
            """
        
        # Generate verses
        generated_verses = []
        
        for _ in range(num_samples):
            if not self.model_available:
                # If model is not available, use a fallback approach with original Truyện Kiều verses
                verse = self._generate_fallback_verse(initial_phrase)
                generated_verses.append(verse)
                continue
                
            # Generate text using Ollama with a higher temperature for creativity
            prompt = prompt_template
            generated_text = self.ollama_generate(prompt, max_tokens=max_length, temperature=0.8)
            
            # Process the generated text
            processed_text = self._process_generated_text(generated_text, initial_phrase)
            
            # Apply lục bát structure enforcement
            verse = self._enforce_luc_bat_structure(processed_text, initial_phrase)
            
            # Only add if we have something valid
            if verse:
                generated_verses.append(verse)
        
        return generated_verses

    def _generate_fallback_verse(self, initial_phrase: str = None) -> str:
        """Generate a verse using patterns from Truyện Kiều when Ollama is not available"""
        if not initial_phrase:
            # Without initial phrase, use a complete lục-bát pair from Truyện Kiều
            if self.luc_bat_pairs:
                luc_line, bat_line = random.choice(self.luc_bat_pairs)
                return f"{luc_line},\n{bat_line}."
            else:
                # Fallback if no pairs extracted
                return "Trăm năm trong cõi người ta,\nChữ tài chữ mệnh khéo là ghét nhau."
        
        # Start with the initial phrase
        words = initial_phrase.split()
        
        # Complete to exactly 6 syllables for lục line
        if len(words) < 6:
            # Add padding words to reach 6 syllables
            padding = random.choice(self.luc_line_padding)
            words.extend(padding[:6-len(words)])
            
        # Ensure exactly 6 syllables
        words = words[:6]
        
        # Make sure the last word has a flat tone
        if self.get_syllable_tone(words[-1]) != 'flat':
            words[-1] = random.choice(self.common_endings)
            
        luc_line = ' '.join(words)
        
        # Choose a matching bat line from the extracted pairs or templates
        if self.luc_bat_pairs:
            # Try to find a contextually relevant bat line
            relevant_pairs = []
            for pair in self.luc_bat_pairs:
                pair_luc = pair[0].lower()
                for word in words:
                    if word.lower() in pair_luc:
                        relevant_pairs.append(pair)
                        break
            
            if relevant_pairs:
                _, bat_line = random.choice(relevant_pairs)
            else:
                # No contextually relevant pairs, use a random one
                _, bat_line = random.choice(self.luc_bat_pairs)
        else:
            # Use a template as fallback
            bat_line = random.choice(self.bat_line_templates)
            
        # Format with correct punctuation
        return f"{luc_line},\n{bat_line}."

    def _clean_line(self, line: str) -> str:
        """Clean a line by removing internal punctuation and extraneous spaces"""
        # Remove any internal punctuation
        line = re.sub(r'[,.;:!?]', ' ', line)
        # Normalize spaces
        line = re.sub(r'\s+', ' ', line)
        return line.strip()

    def _process_generated_text(self, text: str, initial_phrase: str = None) -> str:
        """Process the generated text to extract the verse"""
        lines = []
        capturing = False
        
        # Handle empty response or initial phrase
        if not text and initial_phrase:
            return initial_phrase
            
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Start capturing if we see the initial phrase or the typical indicators
            if initial_phrase and initial_phrase in line:
                capturing = True
                # Extract from the initial phrase onwards
                idx = line.find(initial_phrase)
                if idx >= 0:
                    line = line[idx:]
            elif "Bài thơ:" in line:
                capturing = True
                line = line.replace("Bài thơ:", "").strip()
                
            if capturing:
                # Skip lines that are obviously not verses
                if any(x in line for x in ["YÊU CẦU", "NGHIÊM NGẶT", "BẮT ĐẦU", "PHẢI", "MỖI", "TIẾP TỤC"]):
                    continue
                    
                # Clean the line
                line = self._clean_line(line)
                
                # Only include lines that look like verses (reasonable length)
                words = line.split()
                if 3 <= len(words) <= 10:  # Reasonable length for a verse line
                    lines.append(line)
        
        # If we didn't find any valid lines but have an initial phrase, use it
        if not lines and initial_phrase:
            lines.append(initial_phrase)
            
        return '\n'.join(lines)
    
    def _create_exact_syllable_line(self, words: List[str], target_count: int, 
                                   ensure_flat_ending: bool = True) -> List[str]:
        """Create a line with exactly the target number of syllables"""
        # Remove any empty strings
        words = [w for w in words if w]
        
        if not words:
            # Return default words if no input
            if target_count == 6:
                return random.choice(self.luc_line_templates).split()
            else:
                return random.choice(self.bat_line_templates).split()
        
        # Adjust length
        if len(words) > target_count:
            # Truncate to target length
            words = words[:target_count]
        elif len(words) < target_count:
            # Add padding to reach target length
            if target_count == 6:  # Lục line
                padding_options = self.luc_line_padding
            else:  # Bát line
                padding_options = self.bat_line_padding
                
            # Select random padding
            padding = random.choice(padding_options)
            # Add only as many words as needed
            needed = target_count - len(words)
            words.extend(padding[:needed])
        
        # Ensure we have exactly the target number of syllables
        words = words[:target_count]
        
        # Ensure the last word has a flat tone if required
        if ensure_flat_ending and words and self.get_syllable_tone(words[-1]) != 'flat':
            words[-1] = random.choice(self.common_endings)
            
        return words

    def _enforce_luc_bat_structure(self, verse: str, initial_phrase: str = None) -> str:
        """Strictly enforce lục bát structure on verse"""
        verse_lines = []
        
        # Split and clean the text
        raw_lines = verse.strip().split('\n')
        cleaned_lines = [self._clean_line(line) for line in raw_lines if line.strip()]
        
        # Start with the initial phrase if provided
        first_line_words = []
        if initial_phrase:
            for line in cleaned_lines:
                if initial_phrase in line:
                    # Extract words starting from the initial phrase
                    idx = line.lower().find(initial_phrase.lower())
                    if idx >= 0:
                        text_from_phrase = line[idx:]
                        first_line_words = text_from_phrase.split()
                        break
            
            # If not found in any line, use just the initial phrase
            if not first_line_words:
                first_line_words = initial_phrase.split()
        elif cleaned_lines:
            # Use the first line from generated text
            first_line_words = cleaned_lines[0].split()
        
        # Create the first lục line (exactly 6 syllables)
        luc_line_words = self._create_exact_syllable_line(first_line_words, 6, True)
        verse_lines.append(' '.join(luc_line_words) + ',')
        
        # Prepare other lines for processing
        remaining_lines = []
        if cleaned_lines:
            # Skip the first line if we've used it
            if initial_phrase and any(initial_phrase in line for line in cleaned_lines):
                remaining_lines = [line for line in cleaned_lines if initial_phrase not in line]
            else:
                remaining_lines = cleaned_lines[1:]
        
        # Add alternating bát and lục lines
        line_idx = 0
        while line_idx < len(remaining_lines) and len(verse_lines) < 8:  # Limit to 4 pairs
            current_line = remaining_lines[line_idx].split()
            line_idx += 1
            
            if len(verse_lines) % 2 == 1:  # Need a bát line (8 syllables)
                bat_line_words = self._create_exact_syllable_line(current_line, 8, True)
                if len(verse_lines) == len(verse_lines) - 1:  # Last line
                    verse_lines.append(' '.join(bat_line_words) + '.')
                else:
                    verse_lines.append(' '.join(bat_line_words) + ';')
            else:  # Need a lục line (6 syllables)
                luc_line_words = self._create_exact_syllable_line(current_line, 6, True)
                verse_lines.append(' '.join(luc_line_words) + ',')
        
        # If we don't have at least one complete lục-bát pair, add missing lines
        if len(verse_lines) == 1:
            # Add a bát line
            if self.luc_bat_pairs:
                # Try to find a matching bat line from authentic pairs
                for luc, bat in self.luc_bat_pairs:
                    if any(word in luc for word in luc_line_words):
                        verse_lines.append(bat + '.')
                        break
                else:
                    # No match found, use random
                    verse_lines.append(random.choice(self.bat_line_templates) + '.')
            else:
                # Use template
                verse_lines.append(random.choice(self.bat_line_templates) + '.')
        
        # Ensure we end with a complete pair (even number of lines)
        if len(verse_lines) % 2 != 0:
            verse_lines = verse_lines[:-1]
            
        # Make sure the last line ends with a period
        if verse_lines:
            verse_lines[-1] = verse_lines[-1].rstrip(',;:') + '.'
        
        return '\n'.join(verse_lines)
    
    def _select_reference_verses(self, initial_phrase: str = None) -> List[str]:
        """Select reference verses from Truyện Kiều relevant to the initial phrase"""
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
            
            # Bonus points for exact phrase match
            if initial_phrase.lower() in verse.lower():
                score += 10
                
            scored_verses.append((verse, score))
        
        # Sort by score (descending) and get top matches
        scored_verses.sort(key=lambda x: x[1], reverse=True)
        
        # Get top matching verses, then add some random ones for diversity
        top_verses = [v[0] for v in scored_verses[:3] if v[1] > 0]
        
        # If we don't have enough matching verses, add some random ones
        remaining_needed = num_references - len(top_verses)
        if remaining_needed > 0:
            remaining_verses = [v for v in self.truyen_kieu_verses if v not in top_verses]
            random_verses = random.sample(remaining_verses, min(remaining_needed, len(remaining_verses)))
            top_verses.extend(random_verses)
        
        return top_verses
