# Marker Architecture

## Modular Pipeline Architecture

### Directory Structure
```
marker/
├── providers/      # Source file providers (PdfProvider, ImageProvider, DocumentProvider)
├── builders/       # Document structure builders (DocumentBuilder, LayoutBuilder, OcrBuilder, LineBuilder, StructureBuilder)
├── processors/     # Block type processors (table, equation, footnote, sectionheader, llm/)
├── renderers/      # Output format renderers (MarkdownRenderer, JsonRenderer, HtmlRenderer, ChunkRenderer, ExtractionRenderer)
├── schema/         # Block type definitions (BlockTypes enum, Pydantic models)
├── converters/     # End-to-end pipeline (PdfConverter, TableConverter, OCRConverter, ExtractionConverter)
├── services/       # LLM service integrations (Gemini, Claude, OpenAI, Vertex, Ollama, Azure)
├── config/         # Configuration management (ConfigParser)
├── scripts/        # CLI entry points
└── utils/          # Utility functions
```

## Conversion Pipeline Flow
```
Input File
  ↓
Provider (file reading and parsing)
  ↓
Builders (document structure and block creation)
  ↓
Processors (block formatting and transformation)
  ↓
Renderer (final output format)
  ↓
Output (Markdown/JSON/HTML/Chunks)
```

## Block Structure
- Document → Pages → Blocks (Text, Table, Figure, Equation, etc.)
- Each block has: `block_type`, `polygon` (bounding box), `html`, `children`
- Programmatic block manipulation:
  ```python
  document = converter.build_document("FILEPATH")
  forms = document.contained_blocks((BlockTypes.Form,))
  ```
