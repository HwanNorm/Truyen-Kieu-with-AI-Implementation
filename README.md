# Truyện Kiều with AI Implementation

![Vietnamese Literature AI](https://img.shields.io/badge/vietnamese-literature-green)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)

This project implements various AI techniques to analyze, understand, and generate content related to Truyện Kiều, the most significant work in Vietnamese literature.

## 📝 Overview

Truyện Kiều (The Tale of Kieu) by Nguyễn Du is written in lục bát (six-eight) verse form and tells the story of Thúy Kiều, exploring the relationship between talent and fate through 3,254 verses.

This project includes five main components:

1. **Vector Space Model & Search Engine** - Find relevant verses based on keywords
2. **Authorship Attribution** - Determine if a verse was likely written by Nguyễn Du
3. **Language Modeling & Verse Generation** - Generate new verses in the style of Truyện Kiều
4. **Image Generation** - Convert verses into visual representations
5. **Image-to-Verse Retrieval** - Find verses that best match a given image

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/HwanNorm/Truyen-Kieu-with-AI-Implementation.git
   cd Truyen-Kieu-with-AI-Implementation
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. For verse generation using Ollama, install Ollama separately:
   - Visit [https://ollama.ai/](https://ollama.ai/) to download and install
   - Pull the Llama model: `ollama pull llama3`

## 🛠️ Usage

### Interactive Interface (Recommended)

Run the interactive menu:
```bash
python menu_app.py
```

This provides a user-friendly menu to access all features.

### Command-line Interface

For specific tasks:

1. **Search Engine**:
   ```bash
   python app.py --mode search --query "vầng trăng"
   ```
   
   Interactive mode:
   ```bash
   python app.py --mode search
   ```

2. **Authorship Attribution**:
   ```bash
   python app.py --mode train-authorship
   ```

3. **Verse Generation using Ollama**:
   ```bash
   python app.py --mode ollama-verse --prompt "Trăm năm" --num-samples 3
   ```

4. **Image Generation**:
   ```bash
   python app.py --mode generate-image --verse "Trăm năm trong cõi người ta" --style traditional
   ```

5. **Image-to-Verse Retrieval**:
   ```bash
   python app.py --mode image-to-verse --image "path/to/image.jpg" --enhanced
   ```

## 📂 Project Structure

```
truyen-kieu-project/
├── data/
│   ├── truyen_kieu.txt                # Raw text file of the poem
│   ├── vietnamese_stopwords.txt       # List of Vietnamese stopwords
│   └── comparison_texts/              # For authorship attribution
├── src/
│   ├── preprocessor.py                # Text preprocessing functions
│   ├── vectorizer.py                  # TF-IDF implementation
│   ├── search_engine.py               # Search functionality
│   ├── nguyen_du_classifier.py        # Authorship attribution
│   ├── language_model.py              # Language modeling
│   ├── verse_generator.py             # Verse generation
│   ├── ollama_kieu_generator.py       # Ollama-based generator
│   ├── image_generator.py             # Image generation
│   ├── multimodal_retriever.py        # Image-to-verse retrieval
│   └── cultural_context.py            # Vietnamese cultural context
├── models/                            # For saving trained models
├── output/                            # Generated outputs
├── app.py                             # Command-line interface
├── menu_app.py                        # Interactive menu
├── requirements.txt                   # Project dependencies
└── README.md                          # Project documentation
```

## 📊 Features

### Vector Space Model & Search Engine

- TF-IDF vectorization of verses
- Cosine similarity search
- Hybrid search combining vector-based and string-based matching
- Interactive query interface

### Authorship Attribution

- Binary classification: Nguyễn Du vs. other authors
- Exact verse matching with fallback to classifier
- Random Forest model for stylistic analysis
- Confidence scores for predictions

### Language Modeling & Verse Generation

- Multiple model options: N-gram, LSTM, and Ollama integration
- Generation with cultural constraints
- Lục bát structure enforcement
- Post-processing for Vietnamese poetic rules

### Image Generation

- Stable Diffusion-based visualization
- Vietnamese poetic imagery translation
- Multiple artistic styles
- Cultural symbol enhancement

### Image-to-Verse Retrieval

- CLIP-based multimodal matching
- Color analysis for Vietnamese poetic elements
- Cultural context enhancement
- User feedback collection and learning

## 🔍 Example Usage

### Search for verses about the moon:

```bash
python app.py --mode search --query "vầng trăng"
```

Output:
```
1. [Line 42] Lần thâu gương giọt đồ trang, Nước đi đều đều, vành trăng bóng lồng. (score: 0.8976)
2. [Line 1248] Đêm thu một khắc một chầy, Bâng khuâng như tỉnh như say, vành trăng. (score: 0.8834)
...
```

### Generate a verse starting with "Thúy Kiều":

```bash
python app.py --mode ollama-verse --prompt "Thúy Kiều"
```

Output:
```
=== Generated Verse ===
Thúy Kiều là đóa hoa kiếp,
Sắc sảo khôn ngoan trong tâm hồn tình;
Trên đường tình duyên cô đơn,
Mặc mặc lặng lẽ không cần người xa;
Tình yêu buồn bã như gió,
Làm lòng người đau thương sâu hồng nhan;
Đôi mắt rạng rỡ như vàng,
Cầu mong muốn được gần nhau chữ tình.
```


