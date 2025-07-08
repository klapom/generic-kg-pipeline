# Hochschul-LLM Client Documentation

## Overview

The Hochschul-LLM client provides knowledge graph triple extraction capabilities via OpenAI-compatible API. It connects to external Hochschul GPU infrastructure running Qwen1.5-based models for high-performance semantic analysis and relationship extraction.

## Features

### 🚀 Triple Extraction
- **Semantic Analysis**: Advanced understanding of text relationships
- **RDF Triple Generation**: Subject-Predicate-Object extraction
- **Confidence Scoring**: Quality assessment for each extracted triple
- **Domain Adaptation**: Context-aware extraction for specific domains
- **Batch Processing**: Efficient processing of multiple text chunks

### 🔧 OpenAI API Compatibility
- **Standard Interface**: Uses OpenAI Python client
- **JSON Mode**: Structured response format
- **Rate Limiting**: Built-in batch management
- **Error Handling**: Robust retry and fallback mechanisms

## Quick Start

### Basic Usage

```python
import asyncio
from core.clients.hochschul_llm import HochschulLLMClient

async def extract_triples():
    text = "Einstein proposed the Theory of Relativity in 1905."
    
    async with HochschulLLMClient() as client:
        result = await client.extract_triples(text)
        
        if result.success:
            for triple in result.triples:
                print(f"{triple.subject} → {triple.predicate} → {triple.object}")
                print(f"Confidence: {triple.confidence}")

asyncio.run(extract_triples())
```

### Configuration

```python
from core.clients.hochschul_llm import TripleExtractionConfig

config = TripleExtractionConfig(
    temperature=0.1,
    max_tokens=4000,
    confidence_threshold=0.7,
    batch_size=5
)

async with HochschulLLMClient(config) as client:
    result = await client.extract_triples(text)
```

## Configuration Options

### TripleExtractionConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | "qwen1.5-72b" | Model identifier for Hochschul-LLM |
| `temperature` | float | 0.1 | Sampling temperature (0.0-1.0) |
| `max_tokens` | int | 4000 | Maximum tokens in response |
| `timeout_seconds` | int | 60 | Request timeout |
| `max_retries` | int | 3 | Maximum retry attempts |
| `retry_delay_seconds` | float | 1.0 | Delay between retries |
| `confidence_threshold` | float | 0.7 | Minimum confidence for validation |
| `batch_size` | int | 5 | Concurrent requests in batch |

### Environment Configuration

Required environment variables:

```bash
# Set in .env file
HOCHSCHUL_LLM_ENDPOINT=https://llm.hochschule.example/api/v1
HOCHSCHUL_LLM_API_KEY=your-api-key-here
```

Configuration in `config/default.yaml`:

```yaml
llm:
  provider: "hochschul"
  hochschul:
    endpoint: "${HOCHSCHUL_LLM_ENDPOINT}"
    api_key: "${HOCHSCHUL_LLM_API_KEY}"
    model: "qwen1.5-72b"
    temperature: 0.1
    max_tokens: 4000
    timeout: 60
```

## Data Structures

### Triple

```python
@dataclass
class Triple:
    subject: str
    predicate: str
    object: str
    confidence: float = 0.0
    source_chunk: Optional[str] = None
    context: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]
    def to_ntriples(self) -> str
```

### ExtractionResult

```python
@dataclass
class ExtractionResult:
    triples: List[Triple]
    source_text: str
    processing_time_seconds: float
    model_used: str
    confidence_scores: Dict[str, float]
    metadata: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None
    
    @property
    def triple_count(self) -> int
    @property
    def average_confidence(self) -> float
    def filter_by_confidence(self, threshold: float) -> List[Triple]
```

## Advanced Usage

### Domain-Specific Extraction

```python
async with HochschulLLMClient() as client:
    result = await client.extract_triples(
        text,
        domain_context="scientific research and publications",
        ontology_hints=["hasAuthor", "publishedIn", "cites", "hasSubject"]
    )
```

### Batch Processing

```python
text_chunks = ["Text 1", "Text 2", "Text 3"]

async with HochschulLLMClient() as client:
    results = await client.extract_triples_batch(text_chunks)
    
    total_triples = sum(r.triple_count for r in results if r.success)
    print(f"Extracted {total_triples} triples from {len(results)} chunks")
```

### Triple Validation

```python
async with HochschulLLMClient() as client:
    # Extract triples
    result = await client.extract_triples(text)
    
    # Validate quality
    validation = await client.validate_triples(result.triples)
    
    print(f"Quality score: {validation['quality_score']}")
    print(f"Valid triples: {validation['valid_triples']}/{validation['total_triples']}")
```

## Prompt Engineering

### Extraction Prompt Structure

The client automatically builds prompts with:

1. **JSON Schema**: Required response format
2. **Guidelines**: Extraction rules and best practices
3. **Domain Context**: Optional domain-specific instructions
4. **Ontology Hints**: Preferred relationship types
5. **Source Text**: The content to analyze

### Custom Prompt Guidelines

The system includes built-in guidelines:

- Extract factual relationships, not opinions
- Use clear, specific predicates
- Ensure well-defined entities
- Assign confidence based on evidence
- Include relevant context
- Focus on verifiable relationships

### Domain Adaptation

```python
# Scientific domain
result = await client.extract_triples(
    text,
    domain_context="scientific literature and research",
    ontology_hints=["hasAuthor", "publishedIn", "investigates", "usesMethod"]
)

# Business domain
result = await client.extract_triples(
    text,
    domain_context="business and organizational relationships",
    ontology_hints=["worksAt", "hasCEO", "locatedIn", "ownedBy"]
)
```

## Error Handling

### Exception Types

```python
from core.clients.hochschul_llm import HochschulLLMClient

try:
    async with HochschulLLMClient() as client:
        result = await client.extract_triples(text)
        
        if not result.success:
            print(f"Extraction failed: {result.error_message}")
            
except ValueError as e:
    # Configuration errors
    print(f"Configuration error: {e}")
except Exception as e:
    # Network, API, or other errors
    print(f"Unexpected error: {e}")
```

### Graceful Degradation

```python
async with HochschulLLMClient() as client:
    results = await client.extract_triples_batch(text_chunks)
    
    successful_results = [r for r in results if r.success]
    failed_results = [r for r in results if not r.success]
    
    print(f"Processed: {len(successful_results)}/{len(results)} chunks")
    
    if failed_results:
        print("Failed chunks:")
        for result in failed_results:
            print(f"  - {result.error_message}")
```

## Performance Optimization

### Batch Configuration

```python
# High-throughput configuration
config = TripleExtractionConfig(
    batch_size=10,           # More concurrent requests
    retry_delay_seconds=0.5, # Faster retries
    timeout_seconds=30       # Shorter timeouts
)
```

### Rate Limiting

The client automatically manages rate limits:

- Processes in configurable batches
- Adds delays between batches
- Implements exponential backoff for retries

### Confidence Filtering

```python
# Filter for high-confidence triples only
high_confidence_triples = result.filter_by_confidence(0.8)

# Adjust threshold in config
config = TripleExtractionConfig(confidence_threshold=0.8)
```

## Integration with Pipeline

### Document Processing Integration

The Hochschul-LLM client is integrated into the document processing pipeline:

1. **Text Chunking**: Document segments are prepared for extraction
2. **Batch Processing**: Multiple chunks processed efficiently
3. **Quality Control**: Validation and confidence filtering
4. **Storage**: Results stored in Fuseki triple store

### Pipeline Status Tracking

```python
# Check processing status
GET /documents/{document_id}/status

# Response includes extraction details
{
    "stage": "triple_extraction",
    "progress": 80.0,
    "triples_count": 42,
    "extraction_details": {
        "chunks_processed": 10,
        "successful_chunks": 9,
        "failed_chunks": 1,
        "average_confidence": 0.85
    },
    "details": {
        "llm_provider": "Hochschul-LLM",
        "api": "OpenAI-compatible"
    }
}
```

## API Reference

### Core Methods

```python
# Health check
health = await client.health_check()

# Single extraction
result = await client.extract_triples(text, domain_context, ontology_hints)

# Batch extraction
results = await client.extract_triples_batch(text_chunks, domain_context, ontology_hints)

# Validation
validation = await client.validate_triples(triples)
```

### Health Monitoring

```python
async with HochschulLLMClient() as client:
    health = await client.health_check()
    
    if health['status'] == 'healthy':
        print(f"✅ Service operational")
        print(f"   Response time: {health['response_time_ms']}ms")
        print(f"   Model: {health['model']}")
    else:
        print(f"❌ Service unavailable: {health['error']}")
```

## Troubleshooting

### Common Issues

1. **Configuration Not Found**
   ```
   ValueError: Hochschul-LLM configuration not found
   ```
   - Check environment variables are set
   - Verify .env file is loaded
   - Confirm config/default.yaml is correct

2. **API Connection Failed**
   ```
   status: "unhealthy", error: "Connection failed"
   ```
   - Verify endpoint URL is accessible
   - Check API key is valid
   - Confirm network connectivity

3. **Rate Limiting**
   ```
   HTTP 429: Too Many Requests
   ```
   - Reduce batch_size in configuration
   - Increase retry_delay_seconds
   - Check API usage limits

4. **Low Quality Extractions**
   ```
   quality_score: 0.4
   ```
   - Review domain_context setting
   - Adjust confidence_threshold
   - Provide better ontology_hints

### Debugging

Enable detailed logging:

```python
import logging
logging.getLogger("core.clients.hochschul_llm").setLevel(logging.DEBUG)
```

Check service status:

```bash
curl -X GET "https://llm.hochschule.example/api/v1/models" \
     -H "Authorization: Bearer your-api-key"
```

## Examples

See `examples/hochschul_llm_example.py` for comprehensive examples:

- Single text extraction
- Batch processing
- Domain-specific extraction
- Triple validation
- Health monitoring
- Configuration options
- Error handling patterns

## OpenAI API Compatibility

The client uses the standard OpenAI Python client with:

- **Chat Completions**: `/v1/chat/completions` endpoint
- **JSON Mode**: `response_format={"type": "json_object"}`
- **System Messages**: Extraction instructions
- **Token Counting**: Usage tracking and optimization
- **Error Handling**: Standard HTTP error codes

This ensures compatibility with any OpenAI-compatible LLM service.

## Security Considerations

- **API Keys**: Store securely in environment variables
- **Rate Limiting**: Respect service limits
- **Data Privacy**: Consider data handling policies
- **Network Security**: Use HTTPS endpoints
- **Logging**: Avoid logging sensitive content

For complete API documentation, see the docstrings in `core/clients/hochschul_llm.py`.