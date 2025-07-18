# SmolDocling Integration - Quick Troubleshooting

## Most Common Issues

### 1. Token Repetition Bug
**Problem**: Tags repeat endlessly on pages with many similar elements
```xml
<paragraph>Fahrzeugdaten</paragraph>
<paragraph>Fahrzeugdaten</paragraph>  <!-- Repeats 100s of times -->
```
**Solution**: System auto-detects & truncates. Check logs for "Repetition bug detected"

### 2. Empty Document Error
**Problem**: "DoclingDocument has no text content"  
**Cause**: SmolDocling outputs `<paragraph>`, docling-core expects `<text>`  
**Solution**: Automatic transformation applied in VLLMSmolDoclingFinalClient

### 3. vLLM Connection Failed
**Problem**: "Failed to connect to vLLM server"  
**Fix**: 
```bash
# Check server
curl http://localhost:8088/health

# Start if needed
vllm serve vidore/SmolDocling-1.8b-Instruct --port 8088
```

### 4. Missing Dependencies
**Problem**: "Docling not available"  
**Fix**: 
```bash
uv pip install docling>=2.26.0 docling-core>=2.0.0
```

## Quick Debug Commands

```python
# Enable debug logging
import logging
logging.getLogger("core.clients.vllm_smoldocling_final").setLevel(logging.DEBUG)

# Test single page
from core.clients import VLLMSmolDoclingFinalClient
client = VLLMSmolDoclingFinalClient()
result = client.parse_pdf(Path("test.pdf"), max_pages=1)
```

## Key Configuration

```yaml
# config.yaml
services:
  vllm:
    url: "http://localhost:8088/v1"
    timeout: 300

models:
  vision:
    smoldocling:
      dpi: 144  # NOT 300!
      gpu_memory_utilization: 0.8
```

## Error Quick Reference

| Error | Fix |
|-------|-----|
| "Repetition bug detected" | Automatic - check if enough content before truncation |
| "No text content" | Tag transformation handles this automatically |
| "GPU out of memory" | Reduce gpu_memory_utilization |
| "Connection refused" | Start vLLM server |
| "Static method error" | Use `DoclingDocument.load_from_doctags()` not instance method |