# vLLM SmolDocling Client Documentation

## Overview

The vLLM SmolDocling client provides advanced PDF parsing capabilities using GPU acceleration (GPU Workload 1). It's designed to handle complex document layouts, extract tables with structure preservation, analyze images, and recognize mathematical formulas.

## Features

### 🚀 Advanced PDF Processing
- **Complex Layout Analysis**: Multi-column text, headers, footers
- **Table Extraction**: Structure-preserving table parsing with headers and captions
- **Image Analysis**: Image detection with automated caption generation
- **Formula Recognition**: Mathematical formulas in LaTeX format
- **GPU Optimization**: Dedicated GPU utilization for maximum performance

### 🔧 Configuration Options
- **Flexible Processing**: Configurable extraction features
- **Scalable Performance**: Batch processing and timeout controls
- **Quality Control**: Confidence scoring and error handling

## Quick Start

### Basic Usage

```python
import asyncio
from pathlib import Path
from core.clients.vllm_smoldocling import VLLMSmolDoclingClient

async def parse_pdf():
    async with VLLMSmolDoclingClient() as client:
        result = await client.parse_pdf(Path("document.pdf"))
        
        if result.success:
            print(f"Processed {result.total_pages} pages")
            document = client.convert_to_document(result, Path("document.pdf"))
            print(f"Extracted {len(document.segments)} segments")

asyncio.run(parse_pdf())
```

### Custom Configuration

```python
from core.clients.vllm_smoldocling import SmolDoclingConfig

config = SmolDoclingConfig(
    max_pages=50,
    extract_tables=True,
    extract_images=True,
    extract_formulas=False,
    preserve_layout=True,
    timeout_seconds=120
)

async with VLLMSmolDoclingClient(config) as client:
    result = await client.parse_pdf(pdf_path)
```

## Configuration Options

### SmolDoclingConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_pages` | int | 100 | Maximum pages to process |
| `extract_tables` | bool | True | Extract table structures |
| `extract_images` | bool | True | Analyze and describe images |
| `extract_formulas` | bool | True | Extract mathematical formulas |
| `preserve_layout` | bool | True | Maintain document layout info |
| `output_format` | str | "structured" | Output format preference |
| `gpu_optimization` | bool | True | Enable GPU-specific optimizations |
| `batch_size` | int | 1 | Documents per batch |
| `timeout_seconds` | int | 300 | Processing timeout |

## Data Structures

### SmolDoclingResult

```python
@dataclass
class SmolDoclingResult:
    pages: List[SmolDoclingPage]
    metadata: Dict[str, Any]
    processing_time_seconds: float
    model_version: str
    total_pages: int
    success: bool
    error_message: Optional[str] = None
```

### SmolDoclingPage

```python
@dataclass
class SmolDoclingPage:
    page_number: int
    text: str
    tables: List[TableData]
    images: List[ImageData]
    formulas: List[FormulaData]
    layout_info: Dict[str, Any]
    confidence_score: float = 0.0
```

### TableData

```python
@dataclass
class TableData:
    caption: Optional[str]
    headers: List[str]
    rows: List[List[str]]
    page_number: int
    bbox: Optional[Dict[str, float]] = None
```

### ImageData

```python
@dataclass
class ImageData:
    caption: Optional[str]
    description: Optional[str]
    page_number: int
    bbox: Optional[Dict[str, float]] = None
    image_type: str = "figure"
```

### FormulaData

```python
@dataclass
class FormulaData:
    latex: Optional[str]
    mathml: Optional[str]
    description: Optional[str]
    page_number: int
    bbox: Optional[Dict[str, float]] = None
```

## Advanced Usage

### Batch Processing

```python
pdf_files = [Path("doc1.pdf"), Path("doc2.pdf"), Path("doc3.pdf")]

async with VLLMSmolDoclingClient() as client:
    results = await client.batch_parse_pdfs(pdf_files)
    
    for pdf_file, result in zip(pdf_files, results):
        if result.success:
            print(f"✅ {pdf_file.name}: {result.total_pages} pages")
        else:
            print(f"❌ {pdf_file.name}: {result.error_message}")
```

### Health Monitoring

```python
async with VLLMSmolDoclingClient() as client:
    health = await client.health_check()
    
    print(f"Status: {health['status']}")
    print(f"Response time: {health['response_time_ms']}ms")
    print(f"GPU info: {health['gpu_info']}")
```

### Document Conversion

```python
# Convert SmolDocling result to standard Document format
document = client.convert_to_document(result, pdf_path)

# Access different content types
text_segments = document.get_segments_by_type("text")
table_segments = document.get_segments_by_type("table")
image_segments = document.get_segments_by_type("image_caption")
formula_segments = document.get_segments_by_type("formula")
```

## Error Handling

### Common Exceptions

```python
from plugins.parsers.base_parser import ParseError

try:
    result = await client.parse_pdf(pdf_path)
except ParseError as e:
    print(f"Parsing failed: {e}")
except TimeoutError:
    print("Processing timed out")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Graceful Degradation

```python
# Check result success before processing
if result.success:
    document = client.convert_to_document(result, pdf_path)
    # Process successfully parsed content
else:
    logger.error(f"Parsing failed: {result.error_message}")
    # Handle failure case
```

## Performance Optimization

### GPU Memory Management

```python
# For large documents, reduce batch size
config = SmolDoclingConfig(
    batch_size=1,          # Process one document at a time
    max_pages=50,          # Limit pages per document
    gpu_optimization=True  # Enable GPU optimizations
)
```

### Timeout Configuration

```python
# Adjust timeouts based on document complexity
config = SmolDoclingConfig(
    timeout_seconds=600,    # 10 minutes for complex documents
    max_pages=200          # Large document support
)
```

### Selective Extraction

```python
# For faster processing, disable unused features
config = SmolDoclingConfig(
    extract_tables=True,    # Keep tables
    extract_images=False,   # Skip images for speed
    extract_formulas=False, # Skip formulas for speed
    preserve_layout=False   # Skip layout for speed
)
```

## Integration with Pipeline

### Document Processing Pipeline

The vLLM SmolDocling client is automatically integrated into the document processing pipeline:

1. **File Upload**: PDF files are uploaded via `/documents/upload`
2. **Pipeline Trigger**: Processing starts with `/documents/{id}/process`
3. **GPU Assignment**: PDF parsing uses GPU 1 (vLLM SmolDocling)
4. **Stage Tracking**: Progress is tracked through processing stages
5. **Result Storage**: Parsed content flows to chunking and triple extraction

### Pipeline Status

```python
# Check processing status
GET /documents/{document_id}/status

# Response includes GPU workload information
{
    "document_id": "123",
    "status": "processing",
    "stage": "document_parsing",
    "progress": 25.0,
    "details": {
        "parser": "vLLM SmolDocling",
        "gpu": "GPU 1"
    }
}
```

## Service Requirements

### vLLM SmolDocling Service

The client expects a vLLM service running SmolDocling model:

```bash
# Example vLLM command
python -m vllm.entrypoints.openai.api_server \
    --model /models/smoldocling \
    --gpu-memory-utilization 0.9 \
    --port 8002
```

### Configuration

Update `config/default.yaml`:

```yaml
parsing:
  pdf:
    provider: "vllm_smoldocling"
    vllm_endpoint: "http://localhost:8002"
    gpu_optimization: true
    max_pages: 100
```

### Environment Variables

```bash
# Set in .env file
VLLM_SMOLDOCLING_URL=http://localhost:8002
```

## Troubleshooting

### Common Issues

1. **Connection Refused**
   - Check if vLLM service is running
   - Verify endpoint URL in configuration
   - Check network connectivity

2. **Timeout Errors**
   - Increase `timeout_seconds` in configuration
   - Reduce `max_pages` for large documents
   - Check GPU memory availability

3. **GPU Memory Issues**
   - Reduce `batch_size` to 1
   - Lower `gpu_memory_utilization` in vLLM
   - Process smaller documents

4. **Parsing Failures**
   - Check PDF file integrity
   - Verify file is not password-protected
   - Review error messages in logs

### Debugging

Enable detailed logging:

```python
import logging
logging.getLogger("core.clients.vllm_smoldocling").setLevel(logging.DEBUG)
```

### Health Checks

Regular health monitoring:

```python
# In production, check service health periodically
health = await client.health_check()
if health['status'] != 'healthy':
    logger.warning(f"vLLM SmolDocling unhealthy: {health}")
```

## Examples

See `examples/vllm_smoldocling_example.py` for comprehensive usage examples including:

- Single PDF parsing
- Batch processing
- Health monitoring
- Configuration options
- Error handling patterns

## API Reference

For complete API documentation, see the docstrings in:
- `core/clients/vllm_smoldocling.py`
- Auto-generated docs at `/docs` when running the FastAPI server