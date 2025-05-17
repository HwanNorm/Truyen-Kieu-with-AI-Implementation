import os
from typing import Dict, List, Optional, Tuple

class CulturalContextEnhancer:
    """
    Enhancer for adding cultural context to language models and image generation
    for Truyện Kiều.
    """
    
    def __init__(self, context_file: Optional[str] = None):
        """
        Initialize the cultural context enhancer
        
        Args:
            context_file: Optional path to a JSON file with additional context data
        """
        self.context_db = self._initialize_context_db()
        
        # Load additional context if provided
        if context_file and os.path.exists(context_file):
            self._load_context_file(context_file)
    
    def _initialize_context_db(self) -> Dict:
        """
        Initialize the database of cultural context
        
        Returns:
            Dictionary containing cultural context information
        """
        return {
            "characters": {
                "thúy kiều": "The beautiful and talented main character who suffers many hardships",
                "kim trọng": "Kiều's first love and eventual husband",
                "thúc sinh": "Kiều's second husband who is already married to Hoạn Thư",
                "từ hải": "The rebel leader who respects Kiều's talents",
                "hoạn thư": "The jealous first wife of Thúc Sinh who torments Kiều",
                "đạm tiên": "The ghost of a beautiful courtesan that Kiều meets at her grave",
                "sở khanh": "The conman who tricks Kiều into escaping with him",
                "tú bà": "The madam of the brothel where Kiều is first sold",
                "mã giám sinh": "The slave trader who buys Kiều from her family",
                "thúy vân": "Kiều's younger sister who later marries Kim Trọng",
                "vương ông": "Kiều's father",
                "giác duyên": "The Buddhist nun who helps Kiều"
            },
            "locations": {
                "lâm tri": "City where Kiều is forced into prostitution",
                "quan âm các": "The Buddhist temple where Kiều takes refuge",
                "chiêu ẩn am": "The Buddhist hermitage where Kiều stays after her suicide attempt",
                "châu thai": "Place where Từ Hải meets Kiều",
                "vô tích": "City where Hoạn Thư lives",
                "sông tiền đường": "The river where Kiều attempts suicide"
            },
            "themes": {
                "talent_and_fate": "The conflict between talent and destiny",
                "beauty_and_misfortune": "The connection between beauty and suffering",
                "loyalty": "Dedication to promises and relationships",
                "filial_piety": "Devotion to parents and family",
                "buddhist_salvation": "The role of Buddhism in escaping suffering",
                "karma": "The consequences of actions across lifetimes"
            },
            "symbols": {
                "đàn": "Musical instrument (lute) representing Kiều's talent",
                "hoa": "Flowers symbolizing beauty and its transience",
                "trăng": "Moon representing beauty, purity, and the passage of time",
                "liễu": "Willow trees symbolizing feminine grace and sadness",
                "nước": "Water representing the flow of life and tears",
                "sen": "Lotus symbolizing purity emerging from mud (Buddhist symbol)",
                "mưa": "Rain representing sadness and tears",
                "cầu": "Bridge symbolizing connections and transitions"
            },
            "poetic_forms": {
                "lục bát": "Six-eight verse form used in Truyện Kiều",
                "song thất": "Dual seven-syllable lines",
                "thất ngôn": "Seven-syllable lines",
                "ngũ ngôn": "Five-syllable lines"
            },
            "cultural_concepts": {
                "âm dương": "Yin and yang - complementary opposites",
                "mệnh": "Fate or destiny",
                "tài": "Talent or ability",
                "tình": "Love or sentiment",
                "hiếu": "Filial piety",
                "nghĩa": "Righteousness or duty",
                "trung": "Loyalty",
                "tiết": "Chastity or moral integrity"
            }
        }
    
    def _load_context_file(self, context_file: str) -> None:
        """
        Load additional context from a JSON file
        
        Args:
            context_file: Path to a JSON file with additional context
        """
        import json
        
        try:
            with open(context_file, 'r', encoding='utf-8') as f:
                additional_context = json.load(f)
                
            # Merge with existing context
            for category, items in additional_context.items():
                if category in self.context_db:
                    self.context_db[category].update(items)
                else:
                    self.context_db[category] = items
                    
            print(f"Loaded additional context from {context_file}")
            
        except Exception as e:
            print(f"Error loading context file: {e}")
    
    def enhance_text_prompt(self, text: str, target: str = "image") -> str:
        """
        Enhance a text prompt with cultural context
        
        Args:
            text: The original text prompt
            target: Target usage ('image', 'verse_generation', etc.)
            
        Returns:
            Enhanced text prompt
        """
        text_lower = text.lower()
        context_elements = []
        
        # Check for characters
        for character, desc in self.context_db["characters"].items():
            if character in text_lower:
                if target == "image":
                    context_elements.append(f"{character} ({desc})")
                else:
                    context_elements.append(character)
        
        # Check for locations
        for location, desc in self.context_db["locations"].items():
            if location in text_lower:
                if target == "image":
                    context_elements.append(f"at {location} ({desc})")
                else:
                    context_elements.append(location)
        
        # Check for symbols
        for symbol, meaning in self.context_db["symbols"].items():
            if symbol in text_lower:
                if target == "image":
                    context_elements.append(f"{symbol} symbolizing {meaning}")
                else:
                    context_elements.append(symbol)
        
        # Build enhanced prompt
        if not context_elements:
            # No specific elements found, add general context
            if target == "image":
                return f"{text}, in the style of traditional Vietnamese art depicting scenes from Truyện Kiều"
            else:
                return text
        
        # Add found elements to prompt
        if target == "image":
            context_str = ", ".join(context_elements)
            return f"{text}, featuring {context_str}, in the style of traditional Vietnamese art depicting scenes from Truyện Kiều"
        else:
            return text
    
    def get_thematic_elements(self, theme: str) -> Dict[str, List[str]]:
        """
        Get thematic elements for a given theme
        
        Args:
            theme: The theme to look up
            
        Returns:
            Dictionary of thematic elements
        """
        theme_lower = theme.lower()
        elements = {}
        
        # Look for direct theme match
        for theme_key, desc in self.context_db["themes"].items():
            if theme_lower in theme_key or theme_key in theme_lower:
                elements["theme_description"] = desc
                break
        
        # Collect relevant symbols
        elements["symbols"] = []
        for symbol, meaning in self.context_db["symbols"].items():
            if theme_lower in meaning.lower():
                elements["symbols"].append(symbol)
        
        # Collect relevant characters
        elements["characters"] = []
        for character, desc in self.context_db["characters"].items():
            if theme_lower in desc.lower():
                elements["characters"].append(character)
        
        return elements
    
    def suggest_verse_pairings(self, verse: str) -> List[str]:
        """
        Suggest themes or other verses that pair well with the input verse
        
        Args:
            verse: Input verse
            
        Returns:
            List of suggestions
        """
        verse_lower = verse.lower()
        suggestions = []
        found_themes = []
        
        # Find themes present in the verse
        for theme, desc in self.context_db["themes"].items():
            theme_keywords = theme.replace('_', ' ').split()
            if any(keyword in verse_lower for keyword in theme_keywords):
                found_themes.append(theme)
                
        # Find symbols present in the verse
        found_symbols = []
        for symbol, meaning in self.context_db["symbols"].items():
            if symbol in verse_lower:
                found_symbols.append(symbol)
                
        # Generate suggestions based on themes and symbols
        if found_themes:
            theme_str = ", ".join(found_themes).replace('_', ' ')
            suggestions.append(f"Theme: This verse explores {theme_str}")
            
        if found_symbols:
            symbol_meanings = []
            for symbol in found_symbols:
                meaning = self.context_db["symbols"][symbol]
                symbol_meanings.append(f"{symbol} ({meaning})")
                
            suggestions.append(f"Symbols: {', '.join(symbol_meanings)}")
            
        # Suggest related concepts
        related_concepts = []
        for concept, desc in self.context_db["cultural_concepts"].items():
            if concept in verse_lower:
                related_concepts.append(f"{concept} - {desc}")
                
        if related_concepts:
            suggestions.append(f"Cultural concepts: {', '.join(related_concepts)}")
            
        return suggestions
    
    def get_contextual_info(self, key: str, category: Optional[str] = None) -> Dict:
        """
        Get contextual information for a specific key
        
        Args:
            key: The key to look up
            category: Optional category to search in
            
        Returns:
            Dictionary with contextual information
        """
        key_lower = key.lower()
        
        if category:
            # Search in specific category
            if category in self.context_db:
                for term, info in self.context_db[category].items():
                    if key_lower in term or term in key_lower:
                        return {
                            "category": category,
                            "term": term,
                            "info": info
                        }
        else:
            # Search in all categories
            for category, items in self.context_db.items():
                for term, info in items.items():
                    if key_lower in term or term in key_lower:
                        return {
                            "category": category,
                            "term": term,
                            "info": info
                        }
        
        # Nothing found
        return {}


# Test the cultural context enhancer if run directly
if __name__ == "__main__":
    enhancer = CulturalContextEnhancer()
    
    # Test text enhancement
    test_verses = [
        "Trăm năm trong cõi người ta,",
        "Thúy Kiều tài sắc ai bì,",
        "Mây trôi bèo nổi thiếu gì là nơi!"
    ]
    
    print("Testing text enhancement for image generation:")
    for verse in test_verses:
        enhanced = enhancer.enhance_text_prompt(verse, "image")
        print(f"Original: {verse}")
        print(f"Enhanced: {enhanced}")
        print()
    
    # Test thematic elements
    test_themes = ["fate", "beauty", "loyalty"]
    
    print("Testing thematic elements:")
    for theme in test_themes:
        elements = enhancer.get_thematic_elements(theme)
        print(f"Theme: {theme}")
        print(f"Elements: {elements}")
        print()
    
    # Test verse pairings
    print("Testing verse pairings:")
    for verse in test_verses:
        suggestions = enhancer.suggest_verse_pairings(verse)
        print(f"Verse: {verse}")
        print("Suggestions:")
        for suggestion in suggestions:
            print(f"- {suggestion}")
        print()