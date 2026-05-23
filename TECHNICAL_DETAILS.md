# Technical Documentation: Book Translator

This document provides a technical overview of the Book Translator project, detailing the architecture, model implementation, and file processing pipeline.

## 1. Core Architecture

The project is built as a document translation pipeline that leverages state-of-the-art Transformer models to convert English text into 25+ Indian languages. It consists of two primary interfaces:
- **Gradio Web UI (`app.py`)**: A user-friendly interface for file uploads, settings adjustment, and side-by-side previews.
- **Research Notebook (`Book-Translator.ipynb`)**: An experimental environment for model evaluation (BLEU, TER, METEOR) and pipeline testing.

## 2. Model Implementation

### 2.1 Model Details
- **Base Model**: `ai4bharat/indictrans2-en-indic-1B`
- **Architecture**: Sequence-to-Sequence Transformer (Encoder-Decoder).
- **Tokenizer**: AutoTokenizer (HF Transformers) with `trust_remote_code=True`.
- **Processor**: `IndicProcessor` from `IndicTransToolkit` for specialized Indic language handling.

### 2.2 Memory Optimization (Quantization)
To support varying hardware capabilities, the system implements `BitsAndBytesConfig` for model quantization:
- **Full Precision (FP16/BF16)**: Highest accuracy.
- **8-bit Quantization**: Reduced memory footprint with minimal accuracy loss.
- **4-bit Quantization**: Maximum compression for low-VRAM GPUs or CPU-heavy environments.

```python
# Quantization logic implemented in load_model()
qconfig = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

## 3. Translation Pipeline

### 3.1 Text Processing
The translation utilizes `batch_translate` to handle sentences in parallel, optimizing GPU throughput.
1. **Preprocessing**: `IndicProcessor.preprocess_batch` formats text for the model.
2. **Inference**: `model.generate` produces translated tokens using Beam Search (default `num_beams=5`).
3. **Postprocessing**: `IndicProcessor.postprocess_batch` cleans up tokens into human-readable text.

### 3.2 PDF Handling (Formatting Preservation)
PDF translation is powered by **PyMuPDF (fitz)**. Unlike standard text extractors, the pipeline:
1. Extracts text blocks with coordinate metadata (`bbox`), font size, and color.
2. Filters non-translatable elements (numbers, bullets, white space).
3. Translates text in batches.
4. **Layout Reconstruction**: 
    - Draws white rectangles over original text to "erase" it.
    - Re-inserts translated text into the exact same coordinates using `insert_htmlbox`.
    - Matches original font size and color using dynamically generated CSS.

### 3.3 Text File Handling
Text files are processed line-by-line, maintaining original paragraph breaks and UTF-8 encoding.

## 4. UI & Preview System

The Gradio interface implements a sophisticated preview system:
- **PDF Preview**: Renders the first 4 pages of the PDF as base64-encoded PNG images for immediate visual verification of layout preservation.
- **Side-by-Side View**: Uses custom HTML/CSS cards to display original and translated content together.
- **Progress Tracking**: Real-time progress bars update based on page completion (PDF) or batch completion (TXT).


## 5. Dependencies

- `transformers`: Core model loading and inference.
- `IndicTransToolkit`: Specialized preprocessing for Indic languages.
- `pymupdf`: Advanced PDF manipulation and rendering.
- `gradio`: Web interface components.
- `bitsandbytes`: Quantization backend.
- `torch`: Deep learning framework.
