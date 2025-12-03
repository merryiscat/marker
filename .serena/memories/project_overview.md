# Marker Project Overview

## Purpose
Marker is a Python library that converts documents (PDF, images, PPTX, DOCX, XLSX, HTML, EPUB) to Markdown, JSON, HTML, and chunks with high speed and accuracy. It uses a deep learning model pipeline for OCR, layout detection, and text formatting, with optional LLM integration for enhanced accuracy.

## Tech Stack
- **Language**: Python 3.10+
- **Package Manager**: Poetry
- **Deep Learning**: PyTorch (GPU, CPU, MPS support)
- **OCR**: Surya OCR models
- **LLM Integration**: Gemini (default), Claude, OpenAI, Vertex, Ollama, Azure OpenAI
- **Key Libraries**: 
  - transformers (Hugging Face)
  - Pillow (image processing)
  - pdftext (PDF parsing)
  - markdownify (HTML to Markdown)
  - scikit-learn (ML utilities)
  - anthropic, google-genai, openai (LLM services)

## Installation
```bash
# Basic installation
pip install marker-pdf

# Full document format support (DOCX, PPTX, XLSX, EPUB, HTML)
pip install marker-pdf[full]

# Development environment
poetry install
```

## License
- Code: GPL-3.0-or-later
- Models: Modified AI Pubs Open Rail-M (free for research, personal use, startups <$2M funding/revenue)
