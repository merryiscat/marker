# Marker Suggested Commands

## Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/converters/test_pdf_converter.py

# Run specific test function
pytest tests/converters/test_pdf_converter.py::test_function_name

# Run tests with custom PDF filename marker
pytest -m "filename('your_file.pdf')"
```

## Code Quality
```bash
# Run pre-commit hooks (Ruff linting and formatting)
pre-commit run --all-files

# Manual Ruff commands
ruff check --fix .    # Lint and auto-fix
ruff format .         # Format code
```

## Conversion Commands
```bash
# Single file conversion
marker_single /path/to/file.pdf

# Multiple files in folder
marker /path/to/folder

# Multi-GPU batch conversion
NUM_DEVICES=4 NUM_WORKERS=15 marker_chunk_convert ../pdf_in ../md_out

# GUI app
pip install streamlit streamlit-ace
marker_gui

# API server
pip install -U uvicorn fastapi python-multipart
marker_server --port 8001
```

## Common Flags
```bash
# Use LLM for enhanced accuracy
marker_single file.pdf --use_llm

# Force OCR for all content
marker_single file.pdf --force_ocr

# Strip existing OCR text and re-OCR
marker_single file.pdf --strip_existing_ocr

# Specify output format
marker_single file.pdf --output_format [markdown|json|html|chunks]

# Debug mode (save layout/text images)
marker_single file.pdf --debug
```

## Environment Variables
```bash
# Force specific torch device
TORCH_DEVICE=cuda  # or cpu, mps

# Google API key for Gemini (hybrid mode)
GOOGLE_API_KEY=your_key
```

## Windows-Specific Commands
```bash
# Directory listing
dir
ls  # if using PowerShell or Git Bash

# Change directory
cd path\to\directory

# Find files
dir /s /b *.py  # recursive search for .py files

# Grep equivalent (PowerShell)
Select-String -Path "*.py" -Pattern "search_term"

# Git commands (standard)
git status
git add .
git commit -m "message"
```
