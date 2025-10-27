# Book Translator

A powerful machine learning-based tool for translating books and documents from English to various Indian languages using the IndicTrans2 model.

## Features

- Supports translation to multiple Indian languages:

  - Hindi (hin_Deva)
  - Marathi (mar_Deva)
  - Gujarati (guj_Gujr)
  - Tamil (tam_Taml)
  - Telugu (tel_Telu)

- Supports multiple file formats:
  - PDF files (with formatting preservation)
  - Text files (.txt)
  - DOCX files (coming soon)

## Model Details

The project uses the AI4Bharat/IndicTrans2 model for translations:

- Base model: `ai4bharat/indictrans2-en-indic-1B`
- High accuracy for Indian language translations
- Preserves formatting and layout of source documents
- Supports batch processing for efficient translation

## Installation

1. Clone the required repositories:

```bash
git clone https://github.com/AI4Bharat/IndicTrans2.git
```

2. Install the required Python packages:

```bash
pip install nltk sacremoses pandas regex mock transformers>=4.33.2 mosestokenizer
pip install bitsandbytes scipy accelerate datasets
pip install sentencepiece
pip install pymupdf
pip install evaluate
```

3. Install NLTK data:

```python
import nltk
nltk.download('punkt')
```

4. Install the IndicTransToolkit:

```bash
git clone https://github.com/VarunGumma/IndicTransToolkit.git
cd IndicTransToolkit
pip install --editable ./
```

## Usage

1. Place your source document in the project directory
2. Use the `convertFile()` function to translate your document:

```python
# For PDF files
convertFile("document_name", "pdf", "hin_Deva")  # For Hindi translation

# For text files
convertFile("document_name", "txt", "tam_Taml")  # For Tamil translation
```

### Supported Language Codes

- Hindi: "hin_Deva"
- Marathi: "mar_Deva"
- Gujarati: "guj_Gujr"
- Tamil: "tam_Taml"
- Telugu: "tel_Telu"

## Project Structure

```
Book-Translator/
├── Book-Translator.ipynb    # Main notebook with model and translation code
├── Input Samples/          # Sample input files for testing
│   └── sample.txt
├── output/                 # Directory for translated output
└── README.md
```

## Model Evaluation

The translation model has been evaluated using multiple metrics:

- BLEU Score
- TER (Translation Edit Rate)
- METEOR Score

Comparisons have been made against translations from:

- Google Translate API
- ChatGPT
- Gemini

## Technical Requirements

- Python 3.x
- CUDA-compatible GPU (recommended for better performance)
- Sufficient RAM for model loading
- Disk space for model weights

## License

This project uses the IndicTrans2 model which is released under the MIT License.
