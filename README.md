# Marker (한글 번역)

Marker는 문서를 마크다운, JSON, 청크, HTML로 빠르고 정확하게 변환합니다.

- 모든 언어의 PDF, 이미지, PPTX, DOCX, XLSX, HTML, EPUB 파일 변환
- 표, 양식, 수식, 인라인 수식, 링크, 참조, 코드 블록 포맷팅
- 이미지 추출 및 저장
- 머리글/바닥글 및 기타 아티팩트 제거
- 자체 포맷팅 및 로직으로 확장 가능
- JSON 스키마를 통한 구조화된 추출 수행 (베타)
- LLM을 활용한 정확도 향상 옵션 (커스텀 프롬프트 지원)
- GPU, CPU, MPS에서 작동

관리형 API 또는 온프레미스 문서 인텔리전스 솔루션은 [플랫폼](https://datalab.to?utm_source=gh-marker)을 확인하세요.

## 성능

<img src="data/images/overall.png" width="800px"/>

Marker는 Llamaparse, Mathpix 같은 클라우드 서비스 및 기타 오픈소스 도구와 비교하여 우수한 벤치마크 결과를 보여줍니다.

위 결과는 단일 PDF 페이지를 순차적으로 실행한 것입니다. Marker는 배치 모드에서 훨씬 빠르며, H100에서 초당 25페이지의 처리량을 달성할 것으로 예상됩니다.

자세한 속도 및 정확도 벤치마크와 자체 벤치마크 실행 방법은 [아래](#benchmarks)를 참조하세요.

## 하이브리드 모드

최고의 정확도를 위해서는 `--use_llm` 플래그를 전달하여 marker와 함께 LLM을 사용하세요. 이를 통해 페이지 간 표 병합, 인라인 수식 처리, 표 올바른 포맷팅, 양식에서 값 추출 등을 수행할 수 있습니다. gemini 또는 ollama 모델을 사용할 수 있으며, 기본적으로 `gemini-2.0-flash`를 사용합니다. 자세한 내용은 [아래](#llm-services)를 참조하세요.

다음은 marker, gemini flash 단독, marker + use_llm을 비교한 표 벤치마크입니다:

<img src="data/images/table.png" width="400px"/>

보시다시피 use_llm 모드는 marker 또는 gemini 단독보다 더 높은 정확도를 제공합니다.

## 예제

| PDF | 파일 타입 | Markdown                                                                                                                     | JSON                                                                                                   |
|-----|-----------|------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| [Think Python](https://greenteapress.com/thinkpython/thinkpython.pdf) | 교과서 | [보기](https://github.com/VikParuchuri/marker/blob/master/data/examples/markdown/thinkpython/thinkpython.md)                 | [보기](https://github.com/VikParuchuri/marker/blob/master/data/examples/json/thinkpython.json)         |
| [Switch Transformers](https://arxiv.org/pdf/2101.03961.pdf) | arXiv 논문 | [보기](https://github.com/VikParuchuri/marker/blob/master/data/examples/markdown/switch_transformers/switch_trans.md) | [보기](https://github.com/VikParuchuri/marker/blob/master/data/examples/json/switch_trans.json) |
| [Multi-column CNN](https://arxiv.org/pdf/1804.07821.pdf) | arXiv 논문 | [보기](https://github.com/VikParuchuri/marker/blob/master/data/examples/markdown/multicolcnn/multicolcnn.md)                 | [보기](https://github.com/VikParuchuri/marker/blob/master/data/examples/json/multicolcnn.json)         |

# 상업적 사용

모델 가중치는 수정된 AI Pubs Open Rail-M 라이선스(연구, 개인 사용, 200만 달러 미만의 자금/수익을 가진 스타트업의 경우 무료)를 사용하며 코드는 GPL입니다. 더 광범위한 상업 라이선스 또는 GPL 요구사항 제거를 위해서는 [가격 페이지](https://www.datalab.to/pricing?utm_source=gh-marker)를 방문하세요.

# 호스팅 API 및 온프레미스

marker를 위한 [호스팅 API](https://www.datalab.to?utm_source=gh-marker) 및 [간편한 온프레미스 솔루션](https://www.datalab.to/blog/self-serve-on-prem-licensing)이 있습니다 - 무료로 가입할 수 있으며 테스트를 위한 크레딧을 제공합니다.

API 특징:
- PDF, 이미지, PPT, PPTX, DOC, DOCX, XLS, XLSX, HTML, EPUB 파일 지원
- 주요 클라우드 기반 경쟁사의 1/4 가격
- 빠름 - 250페이지 PDF 약 15초
- LLM 모드 지원
- 높은 가동시간 (99.99%)

# 커뮤니티

[Discord](https://discord.gg//KuZwXNGnfH)에서 향후 개발에 대해 논의합니다.

# 설치

python 3.10+ 및 [PyTorch](https://pytorch.org/get-started/locally/)가 필요합니다.

다음 명령으로 설치:

```shell
pip install marker-pdf
```

PDF 외의 문서에 marker를 사용하려면 추가 의존성을 설치해야 합니다:

```shell
pip install marker-pdf[full]
```

# 사용법

먼저 몇 가지 설정:

- torch 디바이스는 자동으로 감지되지만 재정의할 수 있습니다. 예: `TORCH_DEVICE=cuda`.
- 일부 PDF는 디지털 문서라도 잘못된 텍스트를 포함할 수 있습니다. 모든 라인에 OCR을 강제하려면 `--force_ocr`을 설정하거나, 모든 디지털 텍스트를 유지하고 기존 OCR 텍스트를 제거하려면 `strip_existing_ocr`을 설정하세요.
- 인라인 수식이 중요한 경우 `force_ocr`을 설정하여 인라인 수식을 LaTeX로 변환하세요.

## 인터랙티브 앱

기본 옵션으로 marker를 대화식으로 사용할 수 있는 streamlit 앱이 포함되어 있습니다. 다음과 같이 실행:

```shell
pip install streamlit streamlit-ace
marker_gui
```

## 단일 파일 변환

```shell
marker_single /path/to/file.pdf
```

PDF 또는 이미지를 전달할 수 있습니다.

옵션:
- `--page_range TEXT`: 처리할 페이지를 지정합니다. 쉼표로 구분된 페이지 번호 및 범위를 허용합니다. 예: `--page_range "0,5-10,20"`은 페이지 0, 5~10, 20을 처리합니다.
- `--output_format [markdown|json|html|chunks]`: 출력 결과의 형식을 지정합니다.
- `--output_dir PATH`: 출력 파일이 저장될 디렉토리입니다. 기본값은 settings.OUTPUT_DIR에 지정된 값입니다.
- `--paginate_output`: `\n\n{PAGE_NUMBER}` 다음에 `-` * 48, 그 다음 `\n\n`을 사용하여 출력을 페이지로 나눕니다.
- `--use_llm`: LLM을 사용하여 정확도를 향상시킵니다. LLM 백엔드를 구성해야 합니다 - [아래](#llm-services) 참조.
- `--force_ocr`: 추출 가능한 텍스트를 포함할 수 있는 페이지에도 전체 문서에 OCR 처리를 강제합니다. 이는 인라인 수식도 올바르게 포맷합니다.
- `--block_correction_prompt`: LLM 모드가 활성화된 경우, marker 출력을 수정하는 데 사용될 선택적 프롬프트입니다. 출력에 적용하려는 커스텀 포맷팅이나 로직에 유용합니다.
- `--strip_existing_ocr`: 문서의 모든 기존 OCR 텍스트를 제거하고 surya로 재OCR합니다.
- `--redo_inline_math`: 최고 품질의 인라인 수식 변환을 원하는 경우 `--use_llm`과 함께 사용하세요.
- `--disable_image_extraction`: PDF에서 이미지를 추출하지 않습니다. `--use_llm`도 지정하면 이미지가 설명으로 대체됩니다.
- `--debug`: 추가 로깅 및 진단 정보를 위한 디버그 모드를 활성화합니다.
- `--processors TEXT`: 쉼표로 구분된 전체 모듈 경로를 제공하여 기본 프로세서를 재정의합니다. 예: `--processors "module1.processor1,module2.processor2"`
- `--config_json PATH`: 추가 설정을 포함하는 JSON 구성 파일의 경로입니다.
- `config --help`: 사용 가능한 모든 빌더, 프로세서, 변환기 및 관련 구성을 나열합니다. 이 값들은 marker 기본값을 추가로 조정하기 위한 JSON 구성 파일을 만드는 데 사용할 수 있습니다.
- `--converter_cls`: `marker.converters.pdf.PdfConverter`(기본값) 또는 `marker.converters.table.TableConverter` 중 하나입니다. `PdfConverter`는 전체 PDF를 변환하고, `TableConverter`는 표만 추출 및 변환합니다.
- `--llm_service`: `--use_llm`이 전달되는 경우 사용할 llm 서비스입니다. 기본값은 `marker.services.gemini.GoogleGeminiService`입니다.
- `--help`: marker에 전달할 수 있는 모든 플래그를 확인합니다. (위에 나열된 것보다 훨씬 더 많은 옵션을 지원합니다)

surya OCR에 대해 지원되는 언어 목록은 [여기](https://github.com/VikParuchuri/surya/blob/master/surya/recognition/languages.py)에 있습니다. OCR이 필요하지 않으면 marker는 모든 언어에서 작동할 수 있습니다.

## 여러 파일 변환

```shell
marker /path/to/input/folder
```

- `marker`는 위의 `marker_single`과 동일한 모든 옵션을 지원합니다.
- `--workers`는 동시에 실행할 변환 워커의 수입니다. 이는 기본적으로 자동으로 설정되지만, CPU/GPU 사용량을 늘리는 대신 처리량을 늘리기 위해 증가시킬 수 있습니다. Marker는 피크 시 워커당 5GB의 VRAM을 사용하며 평균 3.5GB입니다.

## 여러 GPU에서 여러 파일 변환

```shell
NUM_DEVICES=4 NUM_WORKERS=15 marker_chunk_convert ../pdf_in ../md_out
```

- `NUM_DEVICES`는 사용할 GPU의 수입니다. `2` 이상이어야 합니다.
- `NUM_WORKERS`는 각 GPU에서 실행할 병렬 프로세스의 수입니다.

## Python에서 사용

전달할 수 있는 추가 인수는 `marker/converters/pdf.py`의 `PdfConverter` 클래스를 참조하세요.

```python
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

converter = PdfConverter(
    artifact_dict=create_model_dict(),
)
rendered = converter("FILEPATH")
text, _, images = text_from_rendered(rendered)
```

`rendered`는 요청된 출력 타입에 따라 다른 속성을 가진 pydantic basemodel입니다. 마크다운 출력(기본값)의 경우 `markdown`, `metadata`, `images` 속성을 갖습니다. json 출력의 경우 `children`, `block_type`, `metadata`를 갖습니다.

### 커스텀 구성

`ConfigParser`를 사용하여 구성을 전달할 수 있습니다. 사용 가능한 모든 옵션을 보려면 `marker_single --help`를 수행하세요.

```python
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser

config = {
    "output_format": "json",
    "ADDITIONAL_KEY": "VALUE"
}
config_parser = ConfigParser(config)

converter = PdfConverter(
    config=config_parser.generate_config_dict(),
    artifact_dict=create_model_dict(),
    processor_list=config_parser.get_processors(),
    renderer=config_parser.get_renderer(),
    llm_service=config_parser.get_llm_service()
)
rendered = converter("FILEPATH")
```

### 블록 추출

각 문서는 하나 이상의 페이지로 구성됩니다. 페이지는 블록을 포함하며, 블록 자체가 다른 블록을 포함할 수 있습니다. 프로그래밍 방식으로 이러한 블록을 조작할 수 있습니다.

다음은 문서에서 모든 양식을 추출하는 예입니다:

```python
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.schema import BlockTypes

converter = PdfConverter(
    artifact_dict=create_model_dict(),
)
document = converter.build_document("FILEPATH")
forms = document.contained_blocks((BlockTypes.Form,))
```

블록 추출 및 조작의 더 많은 예는 프로세서를 참조하세요.

## 기타 변환기

다른 변환 파이프라인을 정의하는 다른 변환기도 사용할 수 있습니다:

### 표 추출

`TableConverter`는 표만 변환하고 추출합니다:

```python
from marker.converters.table import TableConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

converter = TableConverter(
    artifact_dict=create_model_dict(),
)
rendered = converter("FILEPATH")
text, _, images = text_from_rendered(rendered)
```

이것은 PdfConverter와 동일한 모든 구성을 사용합니다. 레이아웃 감지를 피하고 대신 모든 페이지를 표로 가정하려면 `force_layout_block=Table` 구성을 지정할 수 있습니다. 셀 경계 상자도 얻으려면 `output_format=json`을 설정하세요.

CLI를 통해 다음과 같이 실행할 수도 있습니다:
```shell
marker_single FILENAME --use_llm --force_layout_block Table --converter_cls marker.converters.table.TableConverter --output_format json
```

### OCR 전용

OCR만 실행하려면 `OCRConverter`를 통해 수행할 수도 있습니다. 개별 문자와 경계 상자를 유지하려면 `--keep_chars`를 설정하세요.

```python
from marker.converters.ocr import OCRConverter
from marker.models import create_model_dict

converter = OCRConverter(
    artifact_dict=create_model_dict(),
)
rendered = converter("FILEPATH")
```

이것은 PdfConverter와 동일한 모든 구성을 사용합니다.

CLI를 통해 다음과 같이 실행할 수도 있습니다:
```shell
marker_single FILENAME --converter_cls marker.converters.ocr.OCRConverter
```

### 구조화된 추출 (베타)

`ExtractionConverter`를 통해 구조화된 추출을 실행할 수 있습니다. 이를 위해서는 먼저 llm 서비스를 설정해야 합니다(자세한 내용은 [여기](#llm-services) 참조). 추출된 값이 포함된 JSON 출력을 얻을 수 있습니다.

```python
from marker.converters.extraction import ExtractionConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from pydantic import BaseModel

class Links(BaseModel):
    links: list[str]

schema = Links.model_json_schema()
config_parser = ConfigParser({
    "page_schema": schema
})

converter = ExtractionConverter(
    artifact_dict=create_model_dict(),
    config=config_parser.generate_config_dict(),
    llm_service=config_parser.get_llm_service(),
)
rendered = converter("FILEPATH")
```

Rendered는 `original_markdown` 필드를 가집니다. 다음번에 변환기를 실행할 때 이를 `existing_markdown` 구성 키로 다시 전달하면 문서 재파싱을 건너뛸 수 있습니다.

# 출력 형식

## Markdown

Markdown 출력에는 다음이 포함됩니다:

- 이미지 링크 (이미지는 동일한 폴더에 저장됨)
- 포맷된 표
- 내장된 LaTeX 수식 (`$$`로 펜싱됨)
- 코드는 삼중 백틱으로 펜싱됨
- 각주를 위한 위첨자

## HTML

HTML 출력은 마크다운 출력과 유사합니다:

- 이미지는 `img` 태그를 통해 포함됨
- 수식은 `<math>` 태그로 펜싱됨
- 코드는 `pre` 태그에 있음

## JSON

JSON 출력은 트리 구조로 구성되며, 리프 노드가 블록입니다. 리프 노드의 예로는 단일 목록 항목, 텍스트 단락 또는 이미지가 있습니다.

출력은 리스트이며, 각 리스트 항목은 페이지를 나타냅니다. 각 페이지는 내부 marker 스키마에서 블록으로 간주됩니다. 다양한 요소를 나타내는 다양한 블록 타입이 있습니다.

페이지에는 다음 키가 있습니다:

- `id` - 블록의 고유 ID입니다.
- `block_type` - 블록의 타입입니다. 가능한 블록 타입은 `marker/schema/__init__.py`에서 확인할 수 있습니다. 작성 시점에 ["Line", "Span", "FigureGroup", "TableGroup", "ListGroup", "PictureGroup", "Page", "Caption", "Code", "Figure", "Footnote", "Form", "Equation", "Handwriting", "TextInlineMath", "ListItem", "PageFooter", "PageHeader", "Picture", "SectionHeader", "Table", "Text", "TableOfContents", "Document"]입니다.
- `html` - 페이지의 HTML입니다. 자식에 대한 재귀 참조가 있습니다. 전체 html을 원하는 경우 `content-ref` 태그를 자식 콘텐츠로 교체해야 합니다. 이에 대한 예는 `marker/output.py:json_to_html`에서 볼 수 있습니다. 이 함수는 json 출력에서 단일 블록을 가져와 HTML로 변환합니다.
- `polygon` - (x1,y1), (x2,y2), (x3, y3), (x4, y4) 형식의 페이지의 4모서리 다각형입니다. (x1,y1)은 왼쪽 위이며 좌표는 시계 방향으로 진행됩니다.
- `children` - 자식 블록입니다.

자식 블록에는 두 개의 추가 키가 있습니다:

- `section_hierarchy` - 블록이 속한 섹션을 나타냅니다. `1`은 h1 태그, `2`는 h2 등을 나타냅니다.
- `images` - base64로 인코딩된 이미지입니다. 키는 블록 ID이고 데이터는 인코딩된 이미지입니다.

페이지의 자식 블록도 자체 자식을 가질 수 있습니다(트리 구조).

```json
{
      "id": "/page/10/Page/366",
      "block_type": "Page",
      "html": "<content-ref src='/page/10/SectionHeader/0'></content-ref><content-ref src='/page/10/SectionHeader/1'></content-ref><content-ref src='/page/10/Text/2'></content-ref><content-ref src='/page/10/Text/3'></content-ref><content-ref src='/page/10/Figure/4'></content-ref><content-ref src='/page/10/SectionHeader/5'></content-ref><content-ref src='/page/10/SectionHeader/6'></content-ref><content-ref src='/page/10/TextInlineMath/7'></content-ref><content-ref src='/page/10/TextInlineMath/8'></content-ref><content-ref src='/page/10/Table/9'></content-ref><content-ref src='/page/10/SectionHeader/10'></content-ref><content-ref src='/page/10/Text/11'></content-ref>",
      "polygon": [[0.0, 0.0], [612.0, 0.0], [612.0, 792.0], [0.0, 792.0]],
      "children": [
        {
          "id": "/page/10/SectionHeader/0",
          "block_type": "SectionHeader",
          "html": "<h1>Supplementary Material for <i>Subspace Adversarial Training</i> </h1>",
          "polygon": [
            [217.845703125, 80.630859375], [374.73046875, 80.630859375],
            [374.73046875, 107.0],
            [217.845703125, 107.0]
          ],
          "children": null,
          "section_hierarchy": {
            "1": "/page/10/SectionHeader/1"
          },
          "images": {}
        },
        ...
        ]
    }


```

## Chunks

Chunks 형식은 JSON과 유사하지만 트리 대신 모든 것을 단일 리스트로 평면화합니다. 각 페이지의 최상위 블록만 표시됩니다. 또한 각 블록의 전체 HTML이 내부에 있으므로 재구성하기 위해 트리를 탐색할 필요가 없습니다. 이를 통해 RAG를 위한 유연하고 쉬운 청킹이 가능합니다.

## 메타데이터

모든 출력 형식은 다음 필드를 가진 메타데이터 딕셔너리를 반환합니다:

```json
{
    "table_of_contents": [
      {
        "title": "Introduction",
        "heading_level": 1,
        "page_id": 0,
        "polygon": [...]
      }
    ], // 계산된 PDF 목차
    "page_stats": [
      {
        "page_id":  0,
        "text_extraction_method": "pdftext",
        "block_counts": [("Span", 200), ...]
      },
      ...
    ]
}
```

# LLM 서비스

`--use_llm` 플래그로 실행할 때 사용할 수 있는 서비스 선택:

- `Gemini` - 기본적으로 Gemini 개발자 API를 사용합니다. 구성에 `--gemini_api_key`를 전달해야 합니다.
- `Google Vertex` - 더 신뢰할 수 있는 vertex를 사용합니다. `--vertex_project_id`를 전달해야 합니다. 사용하려면 `--llm_service=marker.services.vertex.GoogleVertexService`를 설정하세요.
- `Ollama` - 로컬 모델을 사용합니다. `--ollama_base_url` 및 `--ollama_model`을 구성할 수 있습니다. 사용하려면 `--llm_service=marker.services.ollama.OllamaService`를 설정하세요.
- `Claude` - anthropic API를 사용합니다. `--claude_api_key` 및 `--claude_model_name`을 구성할 수 있습니다. 사용하려면 `--llm_service=marker.services.claude.ClaudeService`를 설정하세요.
- `OpenAI` - 모든 openai 유사 엔드포인트를 지원합니다. `--openai_api_key`, `--openai_model`, `--openai_base_url`을 구성할 수 있습니다. 사용하려면 `--llm_service=marker.services.openai.OpenAIService`를 설정하세요.
- `Azure OpenAI` - Azure OpenAI 서비스를 사용합니다. `--azure_endpoint`, `--azure_api_key`, `--deployment_name`을 구성할 수 있습니다. 사용하려면 `--llm_service=marker.services.azure_openai.AzureOpenAIService`를 설정하세요.

이러한 서비스에는 추가 선택적 구성이 있을 수 있습니다 - 클래스를 보면 확인할 수 있습니다.

# 내부 구조

Marker는 쉽게 확장할 수 있습니다. marker의 핵심 단위는:

- `Providers`, `marker/providers`에 있습니다. PDF와 같은 소스 파일에서 정보를 제공합니다.
- `Builders`, `marker/builders`에 있습니다. 프로바이더의 정보를 사용하여 초기 문서 블록을 생성하고 텍스트를 채웁니다.
- `Processors`, `marker/processors`에 있습니다. 특정 블록을 처리합니다. 예를 들어 표 포맷터는 프로세서입니다.
- `Renderers`, `marker/renderers`에 있습니다. 블록을 사용하여 출력을 렌더링합니다.
- `Schema`, `marker/schema`에 있습니다. 모든 블록 타입에 대한 클래스입니다.
- `Converters`, `marker/converters`에 있습니다. 전체 엔드투엔드 파이프라인을 실행합니다.

처리 동작을 사용자 정의하려면 `processors`를 재정의하세요. 새 출력 형식을 추가하려면 새 `renderer`를 작성하세요. 추가 입력 형식을 위해서는 새 `provider`를 작성하세요.

프로세서와 렌더러는 기본 `PDFConverter`에 직접 전달할 수 있으므로 자신만의 커스텀 처리를 쉽게 지정할 수 있습니다.

## API 서버

다음과 같이 실행할 수 있는 매우 간단한 API 서버가 있습니다:

```shell
pip install -U uvicorn fastapi python-multipart
marker_server --port 8001
```

이렇게 하면 `localhost:8001`에서 액세스할 수 있는 fastapi 서버가 시작됩니다. `localhost:8001/docs`로 이동하여 엔드포인트 옵션을 확인할 수 있습니다.

다음과 같이 요청을 보낼 수 있습니다:

```
import requests
import json

post_data = {
    'filepath': 'FILEPATH',
    # 여기에 다른 매개변수 추가
}

requests.post("http://localhost:8001/marker", data=json.dumps(post_data)).json()
```

이것은 매우 견고한 API가 아니며 소규모 사용만을 위한 것입니다. 이 서버를 사용하고 싶지만 더 견고한 변환 옵션을 원하는 경우 호스팅된 [Datalab API](https://www.datalab.to/plans)를 사용할 수 있습니다.

# 문제 해결

예상대로 작동하지 않는 경우 유용할 수 있는 몇 가지 설정이 있습니다:

- 정확도에 문제가 있는 경우 `--use_llm`을 설정하여 LLM을 사용하여 품질을 향상시키세요. 이것이 작동하려면 `GOOGLE_API_KEY`를 Gemini API 키로 설정해야 합니다.
- 깨진 텍스트가 보이면 `force_ocr`을 설정하세요 - 이렇게 하면 문서를 재OCR합니다.
- `TORCH_DEVICE` - marker가 추론을 위해 지정된 torch 디바이스를 사용하도록 강제하려면 이를 설정하세요.
- 메모리 부족 오류가 발생하면 워커 수를 줄이세요. 긴 PDF를 여러 파일로 분할해 볼 수도 있습니다.

## 디버깅

디버그 모드를 활성화하려면 `debug` 옵션을 전달하세요. 이렇게 하면 감지된 레이아웃 및 텍스트가 있는 각 페이지의 이미지와 추가 경계 상자 정보가 있는 json 파일이 저장됩니다.

# 벤치마크

## 전체 PDF 변환

Common Crawl에서 단일 PDF 페이지를 추출하여 [벤치마크 세트](https://huggingface.co/datasets/datalab-to/marker_benchmark)를 만들었습니다. 우리는 텍스트를 실측 텍스트 세그먼트와 정렬하는 휴리스틱과 판단 방법으로서의 LLM을 기반으로 점수를 매겼습니다.

| 방법       | 평균 시간 | 휴리스틱 점수 | LLM 점수  |
|------------|----------|---------------|-----------|
| marker     | 2.83837  | 95.6709       | 4.23916   |
| llamaparse | 23.348   | 84.2442       | 3.97619   |
| mathpix    | 6.36223  | 86.4281       | 4.15626   |
| docling    | 3.69949  | 86.7073       | 3.70429   |

벤치마크는 marker와 docling의 경우 H100에서 실행되었습니다 - llamaparse와 mathpix는 클라우드 서비스를 사용했습니다. 문서 타입별로도 확인할 수 있습니다:

<img src="data/images/per_doc.png" width="1000px"/>

| 문서 타입              | Marker 휴리스틱 | Marker LLM | Llamaparse 휴리스틱 | Llamaparse LLM | Mathpix 휴리스틱 | Mathpix LLM | Docling 휴리스틱 | Docling LLM |
|----------------------|------------------|------------|----------------------|----------------|-------------------|-------------|-------------------|-------------|
| 과학 논문            | 96.6737          | 4.34899    | 87.1651              | 3.96421        | 91.2267           | 4.46861     | 92.135            | 3.72422     |
| 책 페이지            | 97.1846          | 4.16168    | 90.9532              | 4.07186        | 93.8886           | 4.35329     | 90.0556           | 3.64671     |
| 기타                 | 95.1632          | 4.25076    | 81.1385              | 4.01835        | 79.6231           | 4.00306     | 83.8223           | 3.76147     |
| 양식                 | 88.0147          | 3.84663    | 66.3081              | 3.68712        | 64.7512           | 3.33129     | 68.3857           | 3.40491     |
| 프레젠테이션         | 95.1562          | 4.13669    | 81.2261              | 4              | 83.6737           | 3.95683     | 84.8405           | 3.86331     |
| 재무 문서            | 95.3697          | 4.39106    | 82.5812              | 4.16111        | 81.3115           | 4.05556     | 86.3882           | 3.8         |
| 편지                 | 98.4021          | 4.5        | 93.4477              | 4.28125        | 96.0383           | 4.45312     | 92.0952           | 4.09375     |
| 공학 문서            | 93.9244          | 4.04412    | 77.4854              | 3.72059        | 80.3319           | 3.88235     | 79.6807           | 3.42647     |
| 법률 문서            | 96.689           | 4.27759    | 86.9769              | 3.87584        | 91.601            | 4.20805     | 87.8383           | 3.65552     |
| 신문 페이지          | 98.8733          | 4.25806    | 84.7492              | 3.90323        | 96.9963           | 4.45161     | 92.6496           | 3.51613     |
| 잡지 페이지          | 98.2145          | 4.38776    | 87.2902              | 3.97959        | 93.5934           | 4.16327     | 93.0892           | 4.02041     |

## 처리량

[단일 긴 PDF](https://www.greenteapress.com/thinkpython/thinkpython.pdf)를 사용하여 처리량을 벤치마킹했습니다.

| 방법    | 페이지당 시간 | 문서당 시간 | 사용 VRAM |
|---------|---------------|-------------|-----------|
| marker  | 0.18          | 43.42       | 3.17GB    |

예상 처리량은 H100에서 초당 122페이지입니다 - 사용된 VRAM을 고려하면 22개의 개별 프로세스를 실행할 수 있습니다.

## 표 변환

Marker는 `marker.converters.table.TableConverter`를 사용하여 PDF에서 표를 추출할 수 있습니다. 표 추출 성능은 [FinTabNet](https://developer.ibm.com/exchanges/data/all/fintabnet/)의 테스트 분할을 사용하여 원본 HTML 표현과 추출된 HTML 표현을 비교하여 측정됩니다. HTML 표현은 구조와 내용을 모두 판단하기 위해 트리 편집 거리 기반 메트릭을 사용하여 비교됩니다. Marker는 PDF 페이지의 모든 표의 구조를 감지하고 식별하며 다음 점수를 달성합니다:

| 방법             | 평균 점수 | 총 표 수 |
|------------------|-----------|----------|
| marker           | 0.816     | 99       |
| marker w/use_llm | 0.907     | 99       |
| gemini           | 0.829     | 99       |

보시다시피 `--use_llm` 플래그는 표 인식 성능을 크게 향상시킬 수 있습니다.

fintabnet과 레이아웃 모델의 감지 방법이 약간 다르기 때문에(이로 인해 일부 표가 분할/병합됨) 실측과 정렬할 수 없는 표를 필터링합니다.

## 자체 벤치마크 실행

머신에서 marker의 성능을 벤치마킹할 수 있습니다. 다음과 같이 marker를 수동으로 설치하세요:

```shell
git clone https://github.com/VikParuchuri/marker.git
poetry install
```

### 전체 PDF 변환

[여기](https://drive.google.com/file/d/1ZSeWDo2g1y0BRLT7KnbmytV2bjWARWba/view?usp=sharing)에서 벤치마크 데이터를 다운로드하고 압축을 풉니다. 그런 다음 다음과 같이 전체 벤치마크를 실행합니다:

```shell
python benchmarks/overall.py --methods marker --scores heuristic,llm
```

옵션:

- `--use_llm` marker 결과를 개선하기 위해 llm을 사용합니다.
- `--max_rows` 벤치마크를 위해 처리할 행 수입니다.
- `--methods` `llamaparse`, `mathpix`, `docling`, `marker`가 될 수 있습니다. 쉼표로 구분됩니다.
- `--scores` 사용할 점수 함수는 `llm`, `heuristic`이 될 수 있습니다. 쉼표로 구분됩니다.

### 표 변환
처리된 FinTabNet 데이터세트는 [여기](https://huggingface.co/datasets/datalab-to/fintabnet-test)에 호스팅되어 있으며 자동으로 다운로드됩니다. 다음과 같이 벤치마크를 실행합니다:

```shell
python benchmarks/table/table.py --max_rows 100
```

옵션:

- `--use_llm` marker와 함께 llm을 사용하여 정확도를 향상시킵니다.
- `--use_gemini` gemini 2.0 flash도 벤치마킹합니다.

# 작동 방식

Marker는 딥러닝 모델의 파이프라인입니다:

- 텍스트 추출, 필요한 경우 OCR (휴리스틱, [surya](https://github.com/VikParuchuri/surya))
- 페이지 레이아웃 감지 및 읽기 순서 찾기 ([surya](https://github.com/VikParuchuri/surya))
- 각 블록 정리 및 포맷 (휴리스틱, [texify](https://github.com/VikParuchuri/texify), [surya](https://github.com/VikParuchuri/surya))
- 선택적으로 LLM을 사용하여 품질 향상
- 블록 결합 및 전체 텍스트 후처리

필요한 경우에만 모델을 사용하므로 속도와 정확도가 향상됩니다.

# 제한사항

PDF는 까다로운 형식이므로 marker가 항상 완벽하게 작동하지는 않습니다. 다음은 로드맵에 있는 몇 가지 알려진 제한사항입니다:

- 중첩된 표와 양식이 있는 매우 복잡한 레이아웃은 작동하지 않을 수 있습니다
- 양식이 제대로 렌더링되지 않을 수 있습니다

참고: `--use_llm` 및 `--force_ocr` 플래그를 전달하면 대부분 이러한 문제가 해결됩니다.

# 사용 및 배포 예제

`marker`를 항상 로컬에서 실행할 수 있지만 API로 노출하려는 경우 몇 가지 옵션이 있습니다:
- `marker`와 `surya`로 구동되며 테스트하기 쉬운 플랫폼 API - 무료로 가입할 수 있으며 크레딧을 포함합니다, [여기서 사용해 보세요](https://datalab.to)
- 상업적 사용을 위한 간편한 온프레미스 솔루션, [여기에서 읽어보세요](https://www.datalab.to/blog/self-serve-on-prem-licensing) - 개인정보 보호 보장과 높은 처리량 추론 최적화를 제공합니다.
- [`Modal`](https://modal.com)을 사용하여 웹 엔드포인트를 통해 `marker`를 배포하고 액세스하는 방법을 보여주는 [Modal을 사용한 배포 예제](./examples/README_MODAL.md). Modal은 개발자가 몇 분 만에 GPU에 모델을 배포하고 확장할 수 있게 하는 AI 컴퓨팅 플랫폼입니다.

---

# Marker (Original English)

Marker converts documents to markdown, JSON, chunks, and HTML quickly and accurately.

- Converts PDF, image, PPTX, DOCX, XLSX, HTML, EPUB files in all languages
- Formats tables, forms, equations, inline math, links, references, and code blocks
- Extracts and saves images
- Removes headers/footers/other artifacts
- Extensible with your own formatting and logic
- Does structured extraction, given a JSON schema (beta)
- Optionally boost accuracy with LLMs (and your own prompt)
- Works on GPU, CPU, or MPS

For our managed API or on-prem document intelligence solution, check out [our platform here](https://datalab.to?utm_source=gh-marker).

## Performance

<img src="data/images/overall.png" width="800px"/>

Marker benchmarks favorably compared to cloud services like Llamaparse and Mathpix, as well as other open source tools.

The above results are running single PDF pages serially.  Marker is significantly faster when running in batch mode, with a projected throughput of 25 pages/second on an H100.

See [below](#benchmarks) for detailed speed and accuracy benchmarks, and instructions on how to run your own benchmarks.

## Hybrid Mode

For the highest accuracy, pass the `--use_llm` flag to use an LLM alongside marker.  This will do things like merge tables across pages, handle inline math, format tables properly, and extract values from forms.  It can use any gemini or ollama model.  By default, it uses `gemini-2.0-flash`.  See [below](#llm-services) for details.

Here is a table benchmark comparing marker, gemini flash alone, and marker with use_llm:

<img src="data/images/table.png" width="400px"/>

As you can see, the use_llm mode offers higher accuracy than marker or gemini alone.

## Examples

| PDF | File type | Markdown                                                                                                                     | JSON                                                                                                   |
|-----|-----------|------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| [Think Python](https://greenteapress.com/thinkpython/thinkpython.pdf) | Textbook | [View](https://github.com/VikParuchuri/marker/blob/master/data/examples/markdown/thinkpython/thinkpython.md)                 | [View](https://github.com/VikParuchuri/marker/blob/master/data/examples/json/thinkpython.json)         |
| [Switch Transformers](https://arxiv.org/pdf/2101.03961.pdf) | arXiv paper | [View](https://github.com/VikParuchuri/marker/blob/master/data/examples/markdown/switch_transformers/switch_trans.md) | [View](https://github.com/VikParuchuri/marker/blob/master/data/examples/json/switch_trans.json) |
| [Multi-column CNN](https://arxiv.org/pdf/1804.07821.pdf) | arXiv paper | [View](https://github.com/VikParuchuri/marker/blob/master/data/examples/markdown/multicolcnn/multicolcnn.md)                 | [View](https://github.com/VikParuchuri/marker/blob/master/data/examples/json/multicolcnn.json)         |

# Commercial usage

Our model weights use a modified AI Pubs Open Rail-M license (free for research, personal use, and startups under $2M funding/revenue) and our code is GPL. For broader commercial licensing or to remove GPL requirements, visit our pricing page [here](https://www.datalab.to/pricing?utm_source=gh-marker).

# Hosted API & On-prem

There's a [hosted API](https://www.datalab.to?utm_source=gh-marker) and [painless on-prem solution](https://www.datalab.to/blog/self-serve-on-prem-licensing) for marker - it's free to sign up, and we'll throw in credits for you to test it out.

The API:
- Supports PDF, image, PPT, PPTX, DOC, DOCX, XLS, XLSX, HTML, EPUB files
- Is 1/4th the price of leading cloud-based competitors
- Fast - ~15s for a 250 page PDF
- Supports LLM mode
- High uptime (99.99%)

# Community

[Discord](https://discord.gg//KuZwXNGnfH) is where we discuss future development.

# Installation

You'll need python 3.10+ and [PyTorch](https://pytorch.org/get-started/locally/).

Install with:

```shell
pip install marker-pdf
```

If you want to use marker on documents other than PDFs, you will need to install additional dependencies with:

```shell
pip install marker-pdf[full]
```

# Usage

First, some configuration:

- Your torch device will be automatically detected, but you can override this.  For example, `TORCH_DEVICE=cuda`.
- Some PDFs, even digital ones, have bad text in them.  Set `--force_ocr` to force OCR on all lines, or the `strip_existing_ocr` to keep all digital text, and strip out any existing OCR text.
- If you care about inline math, set `force_ocr` to convert inline math to LaTeX.

## Interactive App

I've included a streamlit app that lets you interactively try marker with some basic options.  Run it with:

```shell
pip install streamlit streamlit-ace
marker_gui
```

## Convert a single file

```shell
marker_single /path/to/file.pdf
```

You can pass in PDFs or images.

Options:
- `--page_range TEXT`: Specify which pages to process. Accepts comma-separated page numbers and ranges. Example: `--page_range "0,5-10,20"` will process pages 0, 5 through 10, and page 20.
- `--output_format [markdown|json|html|chunks]`: Specify the format for the output results.
- `--output_dir PATH`: Directory where output files will be saved. Defaults to the value specified in settings.OUTPUT_DIR.
- `--paginate_output`: Paginates the output, using `\n\n{PAGE_NUMBER}` followed by `-` * 48, then `\n\n`
- `--use_llm`: Uses an LLM to improve accuracy.  You will need to configure the LLM backend - see [below](#llm-services).
- `--force_ocr`: Force OCR processing on the entire document, even for pages that might contain extractable text.  This will also format inline math properly.
- `--block_correction_prompt`: if LLM mode is active, an optional prompt that will be used to correct the output of marker.  This is useful for custom formatting or logic that you want to apply to the output.
- `--strip_existing_ocr`: Remove all existing OCR text in the document and re-OCR with surya.
- `--redo_inline_math`: If you want the absolute highest quality inline math conversion, use this along with `--use_llm`.
- `--disable_image_extraction`: Don't extract images from the PDF.  If you also specify `--use_llm`, then images will be replaced with a description.
- `--debug`: Enable debug mode for additional logging and diagnostic information.
- `--processors TEXT`: Override the default processors by providing their full module paths, separated by commas. Example: `--processors "module1.processor1,module2.processor2"`
- `--config_json PATH`: Path to a JSON configuration file containing additional settings.
- `config --help`: List all available builders, processors, and converters, and their associated configuration.  These values can be used to build a JSON configuration file for additional tweaking of marker defaults.
- `--converter_cls`: One of `marker.converters.pdf.PdfConverter` (default) or `marker.converters.table.TableConverter`.  The `PdfConverter` will convert the whole PDF, the `TableConverter` will only extract and convert tables.
- `--llm_service`: Which llm service to use if `--use_llm` is passed.  This defaults to `marker.services.gemini.GoogleGeminiService`.
- `--help`: see all of the flags that can be passed into marker.  (it supports many more options then are listed above)

The list of supported languages for surya OCR is [here](https://github.com/VikParuchuri/surya/blob/master/surya/recognition/languages.py).  If you don't need OCR, marker can work with any language.

## Convert multiple files

```shell
marker /path/to/input/folder
```

- `marker` supports all the same options from `marker_single` above.
- `--workers` is the number of conversion workers to run simultaneously.  This is automatically set by default, but you can increase it to increase throughput, at the cost of more CPU/GPU usage.  Marker will use 5GB of VRAM per worker at the peak, and 3.5GB average.

## Convert multiple files on multiple GPUs

```shell
NUM_DEVICES=4 NUM_WORKERS=15 marker_chunk_convert ../pdf_in ../md_out
```

- `NUM_DEVICES` is the number of GPUs to use.  Should be `2` or greater.
- `NUM_WORKERS` is the number of parallel processes to run on each GPU.

## Use from python

See the `PdfConverter` class at `marker/converters/pdf.py` function for additional arguments that can be passed.

```python
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

converter = PdfConverter(
    artifact_dict=create_model_dict(),
)
rendered = converter("FILEPATH")
text, _, images = text_from_rendered(rendered)
```

`rendered` will be a pydantic basemodel with different properties depending on the output type requested.  With markdown output (default), you'll have the properties `markdown`, `metadata`, and `images`.  For json output, you'll have `children`, `block_type`, and `metadata`.

### Custom configuration

You can pass configuration using the `ConfigParser`.  To see all available options, do `marker_single --help`.

```python
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser

config = {
    "output_format": "json",
    "ADDITIONAL_KEY": "VALUE"
}
config_parser = ConfigParser(config)

converter = PdfConverter(
    config=config_parser.generate_config_dict(),
    artifact_dict=create_model_dict(),
    processor_list=config_parser.get_processors(),
    renderer=config_parser.get_renderer(),
    llm_service=config_parser.get_llm_service()
)
rendered = converter("FILEPATH")
```

### Extract blocks

Each document consists of one or more pages.  Pages contain blocks, which can themselves contain other blocks.  It's possible to programmatically manipulate these blocks.

Here's an example of extracting all forms from a document:

```python
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.schema import BlockTypes

converter = PdfConverter(
    artifact_dict=create_model_dict(),
)
document = converter.build_document("FILEPATH")
forms = document.contained_blocks((BlockTypes.Form,))
```

Look at the processors for more examples of extracting and manipulating blocks.

## Other converters

You can also use other converters that define different conversion pipelines:

### Extract tables

The `TableConverter` will only convert and extract tables:

```python
from marker.converters.table import TableConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

converter = TableConverter(
    artifact_dict=create_model_dict(),
)
rendered = converter("FILEPATH")
text, _, images = text_from_rendered(rendered)
```

This takes all the same configuration as the PdfConverter.  You can specify the configuration `force_layout_block=Table` to avoid layout detection and instead assume every page is a table.  Set `output_format=json` to also get cell bounding boxes.

You can also run this via the CLI with
```shell
marker_single FILENAME --use_llm --force_layout_block Table --converter_cls marker.converters.table.TableConverter --output_format json
```

### OCR Only

If you only want to run OCR, you can also do that through the `OCRConverter`.  Set `--keep_chars` to keep individual characters and bounding boxes.

```python
from marker.converters.ocr import OCRConverter
from marker.models import create_model_dict

converter = OCRConverter(
    artifact_dict=create_model_dict(),
)
rendered = converter("FILEPATH")
```

This takes all the same configuration as the PdfConverter.

You can also run this via the CLI with
```shell
marker_single FILENAME --converter_cls marker.converters.ocr.OCRConverter
```

### Structured Extraction (beta)

You can run structured extraction via the `ExtractionConverter`.  This requires an llm service to be setup first (see [here](#llm-services) for details).  You'll get a JSON output with the extracted values.

```python
from marker.converters.extraction import ExtractionConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from pydantic import BaseModel

class Links(BaseModel):
    links: list[str]

schema = Links.model_json_schema()
config_parser = ConfigParser({
    "page_schema": schema
})

converter = ExtractionConverter(
    artifact_dict=create_model_dict(),
    config=config_parser.generate_config_dict(),
    llm_service=config_parser.get_llm_service(),
)
rendered = converter("FILEPATH")
```

Rendered will have an `original_markdown` field.  If you pass this back in next time you run the converter, as the `existing_markdown` config key, you can skip re-parsing the document.

# Output Formats

## Markdown

Markdown output will include:

- image links (images will be saved in the same folder)
- formatted tables
- embedded LaTeX equations (fenced with `$$`)
- Code is fenced with triple backticks
- Superscripts for footnotes

## HTML

HTML output is similar to markdown output:

- Images are included via `img` tags
- equations are fenced with `<math>` tags
- code is in `pre` tags

## JSON

JSON output will be organized in a tree-like structure, with the leaf nodes being blocks.  Examples of leaf nodes are a single list item, a paragraph of text, or an image.

The output will be a list, with each list item representing a page.  Each page is considered a block in the internal marker schema.  There are different types of blocks to represent different elements.

Pages have the keys:

- `id` - unique id for the block.
- `block_type` - the type of block. The possible block types can be seen in `marker/schema/__init__.py`.  As of this writing, they are ["Line", "Span", "FigureGroup", "TableGroup", "ListGroup", "PictureGroup", "Page", "Caption", "Code", "Figure", "Footnote", "Form", "Equation", "Handwriting", "TextInlineMath", "ListItem", "PageFooter", "PageHeader", "Picture", "SectionHeader", "Table", "Text", "TableOfContents", "Document"]
- `html` - the HTML for the page.  Note that this will have recursive references to children.  The `content-ref` tags must be replaced with the child content if you want the full html.  You can see an example of this at `marker/output.py:json_to_html`.  That function will take in a single block from the json output, and turn it into HTML.
- `polygon` - the 4-corner polygon of the page, in (x1,y1), (x2,y2), (x3, y3), (x4, y4) format.  (x1,y1) is the top left, and coordinates go clockwise.
- `children` - the child blocks.

The child blocks have two additional keys:

- `section_hierarchy` - indicates the sections that the block is part of.  `1` indicates an h1 tag, `2` an h2, and so on.
- `images` - base64 encoded images.  The key will be the block id, and the data will be the encoded image.

Note that child blocks of pages can have their own children as well (a tree structure).

```json
{
      "id": "/page/10/Page/366",
      "block_type": "Page",
      "html": "<content-ref src='/page/10/SectionHeader/0'></content-ref><content-ref src='/page/10/SectionHeader/1'></content-ref><content-ref src='/page/10/Text/2'></content-ref><content-ref src='/page/10/Text/3'></content-ref><content-ref src='/page/10/Figure/4'></content-ref><content-ref src='/page/10/SectionHeader/5'></content-ref><content-ref src='/page/10/SectionHeader/6'></content-ref><content-ref src='/page/10/TextInlineMath/7'></content-ref><content-ref src='/page/10/TextInlineMath/8'></content-ref><content-ref src='/page/10/Table/9'></content-ref><content-ref src='/page/10/SectionHeader/10'></content-ref><content-ref src='/page/10/Text/11'></content-ref>",
      "polygon": [[0.0, 0.0], [612.0, 0.0], [612.0, 792.0], [0.0, 792.0]],
      "children": [
        {
          "id": "/page/10/SectionHeader/0",
          "block_type": "SectionHeader",
          "html": "<h1>Supplementary Material for <i>Subspace Adversarial Training</i> </h1>",
          "polygon": [
            [217.845703125, 80.630859375], [374.73046875, 80.630859375],
            [374.73046875, 107.0],
            [217.845703125, 107.0]
          ],
          "children": null,
          "section_hierarchy": {
            "1": "/page/10/SectionHeader/1"
          },
          "images": {}
        },
        ...
        ]
    }


```

## Chunks

Chunks format is similar to JSON, but flattens everything into a single list instead of a tree.  Only the top level blocks from each page show up. It also has the full HTML of each block inside, so you don't need to crawl the tree to reconstruct it.  This enable flexible and easy chunking for RAG.

## Metadata

All output formats will return a metadata dictionary, with the following fields:

```json
{
    "table_of_contents": [
      {
        "title": "Introduction",
        "heading_level": 1,
        "page_id": 0,
        "polygon": [...]
      }
    ], // computed PDF table of contents
    "page_stats": [
      {
        "page_id":  0,
        "text_extraction_method": "pdftext",
        "block_counts": [("Span", 200), ...]
      },
      ...
    ]
}
```

# LLM Services

When running with the `--use_llm` flag, you have a choice of services you can use:

- `Gemini` - this will use the Gemini developer API by default.  You'll need to pass `--gemini_api_key` to configuration.
- `Google Vertex` - this will use vertex, which can be more reliable.  You'll need to pass `--vertex_project_id`.  To use it, set `--llm_service=marker.services.vertex.GoogleVertexService`.
- `Ollama` - this will use local models.  You can configure `--ollama_base_url` and `--ollama_model`. To use it, set `--llm_service=marker.services.ollama.OllamaService`.
- `Claude` - this will use the anthropic API.  You can configure `--claude_api_key`, and `--claude_model_name`.  To use it, set `--llm_service=marker.services.claude.ClaudeService`.
- `OpenAI` - this supports any openai-like endpoint. You can configure `--openai_api_key`, `--openai_model`, and `--openai_base_url`. To use it, set `--llm_service=marker.services.openai.OpenAIService`.
- `Azure OpenAI` - this uses the Azure OpenAI service. You can configure `--azure_endpoint`, `--azure_api_key`, and `--deployment_name`. To use it, set `--llm_service=marker.services.azure_openai.AzureOpenAIService`.

These services may have additional optional configuration as well - you can see it by viewing the classes.

# Internals

Marker is easy to extend.  The core units of marker are:

- `Providers`, at `marker/providers`.  These provide information from a source file, like a PDF.
- `Builders`, at `marker/builders`.  These generate the initial document blocks and fill in text, using info from the providers.
- `Processors`, at `marker/processors`.  These process specific blocks, for example the table formatter is a processor.
- `Renderers`, at `marker/renderers`. These use the blocks to render output.
- `Schema`, at `marker/schema`.  The classes for all the block types.
- `Converters`, at `marker/converters`.  They run the whole end to end pipeline.

To customize processing behavior, override the `processors`.  To add new output formats, write a new `renderer`.  For additional input formats, write a new `provider.`

Processors and renderers can be directly passed into the base `PDFConverter`, so you can specify your own custom processing easily.

## API server

There is a very simple API server you can run like this:

```shell
pip install -U uvicorn fastapi python-multipart
marker_server --port 8001
```

This will start a fastapi server that you can access at `localhost:8001`.  You can go to `localhost:8001/docs` to see the endpoint options.

You can send requests like this:

```
import requests
import json

post_data = {
    'filepath': 'FILEPATH',
    # Add other params here
}

requests.post("http://localhost:8001/marker", data=json.dumps(post_data)).json()
```

Note that this is not a very robust API, and is only intended for small-scale use.  If you want to use this server, but want a more robust conversion option, you can use the hosted [Datalab API](https://www.datalab.to/plans).

# Troubleshooting

There are some settings that you may find useful if things aren't working the way you expect:

- If you have issues with accuracy, try setting `--use_llm` to use an LLM to improve quality.  You must set `GOOGLE_API_KEY` to a Gemini API key for this to work.
- Make sure to set `force_ocr` if you see garbled text - this will re-OCR the document.
- `TORCH_DEVICE` - set this to force marker to use a given torch device for inference.
- If you're getting out of memory errors, decrease worker count.  You can also try splitting up long PDFs into multiple files.

## Debugging

Pass the `debug` option to activate debug mode.  This will save images of each page with detected layout and text, as well as output a json file with additional bounding box information.

# Benchmarks

## Overall PDF Conversion

We created a [benchmark set](https://huggingface.co/datasets/datalab-to/marker_benchmark) by extracting single PDF pages from common crawl.  We scored based on a heuristic that aligns text with ground truth text segments, and an LLM as a judge scoring method.

| Method     | Avg Time | Heuristic Score | LLM Score |
|------------|----------|-----------------|-----------|
| marker     | 2.83837  | 95.6709         | 4.23916   |
| llamaparse | 23.348   | 84.2442         | 3.97619   |
| mathpix    | 6.36223  | 86.4281         | 4.15626   |
| docling    | 3.69949  | 86.7073         | 3.70429   |

Benchmarks were run on an H100 for markjer and docling - llamaparse and mathpix used their cloud services.  We can also look at it by document type:

<img src="data/images/per_doc.png" width="1000px"/>

| Document Type        | Marker heuristic | Marker LLM | Llamaparse Heuristic | Llamaparse LLM | Mathpix Heuristic | Mathpix LLM | Docling Heuristic | Docling LLM |
|----------------------|------------------|------------|----------------------|----------------|-------------------|-------------|-------------------|-------------|
| Scientific paper     | 96.6737          | 4.34899    | 87.1651              | 3.96421        | 91.2267           | 4.46861     | 92.135            | 3.72422     |
| Book page            | 97.1846          | 4.16168    | 90.9532              | 4.07186        | 93.8886           | 4.35329     | 90.0556           | 3.64671     |
| Other                | 95.1632          | 4.25076    | 81.1385              | 4.01835        | 79.6231           | 4.00306     | 83.8223           | 3.76147     |
| Form                 | 88.0147          | 3.84663    | 66.3081              | 3.68712        | 64.7512           | 3.33129     | 68.3857           | 3.40491     |
| Presentation         | 95.1562          | 4.13669    | 81.2261              | 4              | 83.6737           | 3.95683     | 84.8405           | 3.86331     |
| Financial document   | 95.3697          | 4.39106    | 82.5812              | 4.16111        | 81.3115           | 4.05556     | 86.3882           | 3.8         |
| Letter               | 98.4021          | 4.5        | 93.4477              | 4.28125        | 96.0383           | 4.45312     | 92.0952           | 4.09375     |
| Engineering document | 93.9244          | 4.04412    | 77.4854              | 3.72059        | 80.3319           | 3.88235     | 79.6807           | 3.42647     |
| Legal document       | 96.689           | 4.27759    | 86.9769              | 3.87584        | 91.601            | 4.20805     | 87.8383           | 3.65552     |
| Newspaper page       | 98.8733          | 4.25806    | 84.7492              | 3.90323        | 96.9963           | 4.45161     | 92.6496           | 3.51613     |
| Magazine page        | 98.2145          | 4.38776    | 87.2902              | 3.97959        | 93.5934           | 4.16327     | 93.0892           | 4.02041     |

## Throughput

We benchmarked throughput using a [single long PDF](https://www.greenteapress.com/thinkpython/thinkpython.pdf).

| Method  | Time per page | Time per document | VRAM used |
|---------|---------------|-------------------|---------- |
| marker  | 0.18          | 43.42             |  3.17GB   |

The projected throughput is 122 pages per second on an H100 - we can run 22 individual processes given the VRAM used.

## Table Conversion

Marker can extract tables from PDFs using `marker.converters.table.TableConverter`. The table extraction performance is measured by comparing the extracted HTML representation of tables against the original HTML representations using the test split of [FinTabNet](https://developer.ibm.com/exchanges/data/all/fintabnet/). The HTML representations are compared using a tree edit distance based metric to judge both structure and content. Marker detects and identifies the structure of all tables in a PDF page and achieves these scores:

| Method           | Avg score | Total tables |
|------------------|-----------|--------------|
| marker           | 0.816     | 99           |
| marker w/use_llm | 0.907     | 99           |
| gemini           | 0.829     | 99           |

The `--use_llm` flag can significantly improve table recognition performance, as you can see.

We filter out tables that we cannot align with the ground truth, since fintabnet and our layout model have slightly different detection methods (this results in some tables being split/merged).

## Running your own benchmarks

You can benchmark the performance of marker on your machine. Install marker manually with:

```shell
git clone https://github.com/VikParuchuri/marker.git
poetry install
```

### Overall PDF Conversion

Download the benchmark data [here](https://drive.google.com/file/d/1ZSeWDo2g1y0BRLT7KnbmytV2bjWARWba/view?usp=sharing) and unzip. Then run the overall benchmark like this:

```shell
python benchmarks/overall.py --methods marker --scores heuristic,llm
```

Options:

- `--use_llm` use an llm to improve the marker results.
- `--max_rows` how many rows to process for the benchmark.
- `--methods` can be `llamaparse`, `mathpix`, `docling`, `marker`.  Comma separated.
- `--scores` which scoring functions to use, can be `llm`, `heuristic`.  Comma separated.

### Table Conversion
The processed FinTabNet dataset is hosted [here](https://huggingface.co/datasets/datalab-to/fintabnet-test) and is automatically downloaded. Run the benchmark with:

```shell
python benchmarks/table/table.py --max_rows 100
```

Options:

- `--use_llm` uses an llm with marker to improve accuracy.
- `--use_gemini` also benchmarks gemini 2.0 flash.

# How it works

Marker is a pipeline of deep learning models:

- Extract text, OCR if necessary (heuristics, [surya](https://github.com/VikParuchuri/surya))
- Detect page layout and find reading order ([surya](https://github.com/VikParuchuri/surya))
- Clean and format each block (heuristics, [texify](https://github.com/VikParuchuri/texify), [surya](https://github.com/VikParuchuri/surya))
- Optionally use an LLM to improve quality
- Combine blocks and postprocess complete text

It only uses models where necessary, which improves speed and accuracy.

# Limitations

PDF is a tricky format, so marker will not always work perfectly.  Here are some known limitations that are on the roadmap to address:

- Very complex layouts, with nested tables and forms, may not work
- Forms may not be rendered well

Note: Passing the `--use_llm` and `--force_ocr` flags will mostly solve these issues.

# Usage and Deployment Examples

You can always run `marker` locally, but if you wanted to expose it as an API, we have a few options:
- Our platform API which is powered by `marker` and `surya` and is easy to test out - it's free to sign up, and we'll include credits, [try it out here](https://datalab.to)
- Our painless on-prem solution for commercial use, which you can [read about here](https://www.datalab.to/blog/self-serve-on-prem-licensing) and gives you privacy guarantees with high throughput inference optimizations.
- [Deployment example with Modal](./examples/README_MODAL.md) that shows you how to deploy and access `marker` through a web endpoint using [`Modal`](https://modal.com). Modal is an AI compute platform that enables developers to deploy and scale models on GPUs in minutes.
