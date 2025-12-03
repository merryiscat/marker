# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

Marker는 PDF, 이미지, PPTX, DOCX, XLSX, HTML, EPUB 문서를 마크다운, JSON, HTML, 청크로 고속·고정확도 변환하는 Python 라이브러리입니다. 딥러닝 모델 파이프라인을 사용하여 OCR, 레이아웃 감지, 텍스트 포맷팅을 수행하며, 선택적으로 LLM을 활용해 정확도를 향상시킬 수 있습니다.

## 개발 환경 설정

**필수 요구사항**:
- Python 3.10 이상
- PyTorch (GPU, CPU, MPS 지원)
- Poetry (의존성 관리)

**설치**:
```bash
# 기본 설치
pip install marker-pdf

# 전체 문서 형식 지원 (DOCX, PPTX, XLSX, EPUB, HTML)
pip install marker-pdf[full]

# 개발 환경 설치
poetry install
```

## 핵심 명령어

### 테스트
```bash
# 전체 테스트 실행
pytest

# 특정 테스트 파일 실행
pytest tests/converters/test_pdf_converter.py

# 특정 테스트 함수 실행
pytest tests/converters/test_pdf_converter.py::test_function_name

# 마커 사용 (커스텀 PDF 파일명 지정)
pytest -m "filename('your_file.pdf')"
```

### 코드 품질
```bash
# Ruff를 사용한 린팅 및 포맷팅
pre-commit run --all-files

# 또는 수동 실행
ruff check --fix .
ruff format .
```

### 실행
```bash
# 단일 파일 변환
marker_single /path/to/file.pdf

# 폴더 내 여러 파일 변환
marker /path/to/folder

# 다중 GPU 사용
NUM_DEVICES=4 NUM_WORKERS=15 marker_chunk_convert ../pdf_in ../md_out

# GUI 앱 실행
pip install streamlit streamlit-ace
marker_gui

# API 서버 실행
pip install -U uvicorn fastapi python-multipart
marker_server --port 8001
```

## 아키텍처

Marker는 모듈화된 파이프라인 아키텍처를 사용합니다:

### 핵심 컴포넌트

1. **Providers** (`marker/providers/`)
   - 소스 파일(PDF, 이미지, DOCX 등)에서 정보를 제공
   - `PdfProvider`, `ImageProvider`, `DocumentProvider` 등
   - 각 파일 형식에 대한 저수준 데이터 추출 처리

2. **Builders** (`marker/builders/`)
   - 초기 문서 블록 생성 및 텍스트 채우기
   - `DocumentBuilder`: 문서 구조 생성
   - `LayoutBuilder`: 레이아웃 감지 및 읽기 순서 결정
   - `OcrBuilder`: OCR이 필요한 경우 텍스트 추출
   - `LineBuilder`: 텍스트 라인 생성
   - `StructureBuilder`: 문서 구조 분석

3. **Processors** (`marker/processors/`)
   - 특정 블록 타입 처리 및 변환
   - `table.py`: 테이블 포맷팅
   - `equation.py`: 수식 처리
   - `footnote.py`: 각주 처리
   - `sectionheader.py`: 섹션 헤더 인식
   - `llm/`: LLM 기반 프로세서 (하이브리드 모드)

4. **Renderers** (`marker/renderers/`)
   - 블록을 최종 출력 형식으로 렌더링
   - `MarkdownRenderer`: 마크다운 출력
   - `JsonRenderer`: JSON 트리 구조 출력
   - `HtmlRenderer`: HTML 출력
   - `ChunkRenderer`: RAG를 위한 평면화된 청크
   - `ExtractionRenderer`: 구조화된 추출 (베타)

5. **Schema** (`marker/schema/`)
   - 모든 블록 타입의 클래스 정의
   - `BlockTypes`: 블록 타입 열거형 (Page, Text, Table, Figure, Equation 등)
   - Pydantic 모델을 사용한 타입 안전성 보장

6. **Converters** (`marker/converters/`)
   - 전체 엔드투엔드 파이프라인 실행
   - `PdfConverter`: 전체 PDF 변환 (기본)
   - `TableConverter`: 테이블만 추출
   - `OCRConverter`: OCR 전용
   - `ExtractionConverter`: 구조화된 데이터 추출 (베타)

### 변환 파이프라인 흐름

```
입력 파일
  ↓
Provider (파일 읽기 및 파싱)
  ↓
Builders (문서 구조 및 블록 생성)
  ↓
Processors (블록 포맷팅 및 변환)
  ↓
Renderer (최종 출력 형식)
  ↓
출력 (Markdown/JSON/HTML/Chunks)
```

### 블록 구조

문서는 트리 구조로 구성:
- **Document** → **Pages** → **Blocks** (Text, Table, Figure 등)
- 각 블록은 `block_type`, `polygon` (경계 상자), `html`, `children`을 가짐
- 프로그래밍 방식으로 블록 조작 가능:
  ```python
  document = converter.build_document("FILEPATH")
  forms = document.contained_blocks((BlockTypes.Form,))
  ```

## 주요 설정

### ConfigParser 사용

`marker/config/parser.py`의 `ConfigParser` 클래스를 통해 설정 관리:

```python
from marker.config.parser import ConfigParser

config = {
    "output_format": "json",
    "force_ocr": True,
    "use_llm": True,
    "gemini_api_key": "YOUR_KEY"
}
config_parser = ConfigParser(config)

converter = PdfConverter(
    config=config_parser.generate_config_dict(),
    artifact_dict=create_model_dict(),
    processor_list=config_parser.get_processors(),
    renderer=config_parser.get_renderer(),
    llm_service=config_parser.get_llm_service()
)
```

### 환경 변수

- `TORCH_DEVICE`: torch 디바이스 강제 지정 (cuda, cpu, mps)
- `GOOGLE_API_KEY`: Gemini API 키 (하이브리드 모드)

### 중요 플래그

- `--use_llm`: LLM으로 정확도 향상 (테이블 병합, 인라인 수식, 양식 추출)
- `--force_ocr`: 전체 문서 OCR 강제 실행
- `--strip_existing_ocr`: 기존 OCR 텍스트 제거 후 재OCR
- `--output_format [markdown|json|html|chunks]`: 출력 형식 지정
- `--debug`: 디버그 모드 (레이아웃/텍스트 이미지 저장)

## LLM 서비스 설정

하이브리드 모드(`--use_llm`)에서 지원하는 서비스:
- **Gemini** (기본): `--gemini_api_key`
- **Google Vertex**: `--llm_service=marker.services.vertex.GoogleVertexService --vertex_project_id=PROJECT_ID`
- **Ollama**: `--llm_service=marker.services.ollama.OllamaService --ollama_base_url=URL`
- **Claude**: `--llm_service=marker.services.claude.ClaudeService --claude_api_key=KEY`
- **OpenAI**: `--llm_service=marker.services.openai.OpenAIService --openai_api_key=KEY`
- **Azure OpenAI**: `--llm_service=marker.services.azure_openai.AzureOpenAIService`

## 확장 방법

### 커스텀 프로세서 추가

`marker/processors/` 패턴을 따라 새 프로세서 작성:

```python
from marker.processors import BaseProcessor

class MyCustomProcessor(BaseProcessor):
    def __call__(self, document):
        # 문서 블록 처리 로직
        return document

# 사용
converter = PdfConverter(
    processor_list=config_parser.get_processors() + [MyCustomProcessor()]
)
```

### 새 출력 형식 추가

`marker/renderers/`에 새 렌더러 작성 후 `PdfConverter`에 전달

### 새 입력 형식 지원

`marker/providers/`에 새 프로바이더 작성 및 `registry.py`에 등록

## 모델 및 가중치

- Surya OCR 모델: 다국어 OCR 및 레이아웃 감지
- Texify: 수식 변환
- 모델 라이선스: AI Pubs Open Rail-M (연구/개인/스타트업 무료, 상업용은 별도 라이선스)

## 디버깅 팁

- `--debug` 플래그로 각 페이지의 레이아웃/텍스트 이미지와 JSON 저장
- 텍스트가 깨지면 `--force_ocr` 사용
- 메모리 부족 시 워커 수 감소 또는 PDF 분할
- `--use_llm`으로 정확도 문제 해결

## 참고 자료

- 벤치마크 실행: `python benchmarks/overall.py --methods marker --scores heuristic,llm`
- API 문서: `localhost:8001/docs` (서버 실행 시)
- Discord 커뮤니티: https://discord.gg//KuZwXNGnfH
