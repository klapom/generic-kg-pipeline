# Docling & SmolDocling Troubleshooting Guide

## Common Issues and Solutions

### 1. SmolDocling Token Repetition Bug

**Symptom**: 
- Pages with many similar elements (e.g., specification tables) show repeating tags
- Same `<paragraph>` or `<table>` tags appear multiple times
- Parsing appears stuck in a loop

**Example**:
```xml
<paragraph>Fahrzeugdaten</paragraph>
<paragraph>Fahrzeugdaten</paragraph>
<paragraph>Fahrzeugdaten</paragraph>
<!-- Repeats hundreds of times -->
```

**Cause**: Known issue with SmolDocling's autoregressive generation

**Solution**: 
- The system automatically detects and truncates at repetition point
- If repetition occurs too early (< 2 meaningful tags), page is marked as unparseable
- Check logs for "Repetition bug detected" messages

**Affected Documents**: 
- BMW technical specifications (pages 3, 8, 13, 14)
- Documents with dense tabular data

---

### 2. DocTags Compatibility Issues

**Symptom**:
- Error: "DoclingDocument has no text content"
- Empty document despite successful SmolDocling parsing

**Cause**: 
- SmolDocling uses `<paragraph>` tags
- docling-core expects `<text>` tags

**Solution**: 
- Automatic transformation is applied in `VLLMSmolDoclingFinalClient`
- Tag mapping:
  - `<paragraph>` → `<text>`
  - `<section_header>` → `<section_header_level_1>`

---

### 3. Docling Not Available

**Symptom**:
```
Error: Docling not available, cannot proceed with parsing
```

**Cause**: 
- docling or docling-core not installed
- Import errors

**Solution**:
```bash
# Install required packages
uv pip install docling>=2.26.0 docling-core>=2.0.0

# Verify installation
python -c "import docling, docling_core; print('Docling OK')"
```

---

### 4. vLLM Server Connection Issues

**Symptom**:
- "Failed to connect to vLLM server"
- Timeout errors during SmolDocling parsing

**Cause**: 
- vLLM server not running
- Incorrect URL configuration

**Solution**:
1. Check vLLM server status:
```bash
curl http://localhost:8088/health
```

2. Verify configuration in `config.yaml`:
```yaml
services:
  vllm:
    url: "http://localhost:8088/v1"
    timeout: 300
```

3. Start vLLM server if needed:
```bash
# See deployment documentation for vLLM setup
vllm serve vidore/SmolDocling-1.8b-Instruct --port 8088
```

---

### 5. Large PDF Processing Issues

**Symptom**:
- Memory errors
- Extremely slow processing
- Timeout errors

**Cause**: 
- PDFs with 100+ pages
- High-resolution images

**Solution**:
1. Increase timeout in config:
```yaml
pdf_parser:
  timeout_seconds: 600  # 10 minutes
```

2. For very large PDFs:
- Consider splitting into smaller parts
- Process in batches

---

### 6. Missing Visual Elements

**Symptom**:
- No images extracted from PDF
- `visual_elements` list is empty

**Possible Causes**:
1. **PDF has no embedded images** - Check with PDF reader
2. **Images are vector graphics** - Not supported
3. **Docling extraction disabled**

**Solution**:
Verify configuration:
```yaml
pdf_parser:
  docling_config:
    extract_images_directly: true
```

---

### 7. Static Method Errors

**Symptom**:
```
TypeError: load_from_doctags() takes 1 positional argument but 2 were given
```

**Cause**: 
- `DoclingDocument.load_from_doctags()` is a static method
- Called incorrectly on instance

**Solution**: 
Always call as static method:
```python
# Correct
doc = DoclingDocument.load_from_doctags(doctags_xml)

# Wrong
doc = DoclingDocument()
doc = doc.load_from_doctags(doctags_xml)  # Error!
```

---

## Debugging Tips

### Enable Debug Logging

```python
import logging
logging.getLogger("core.clients.vllm_smoldocling_final").setLevel(logging.DEBUG)
```

### Check SmolDocling Raw Output

To see raw DocTags before transformation:
```python
# In VLLMSmolDoclingFinalClient._process_with_docling()
logger.debug(f"Raw DocTags:\n{doctags[:500]}")  # First 500 chars
```

### Monitor GPU Memory

For vLLM GPU issues:
```bash
nvidia-smi -l 1  # Update every second
```

### Test Individual Pages

```python
# Test specific problematic page
from pathlib import Path
from core.clients import VLLMSmolDoclingFinalClient

client = VLLMSmolDoclingFinalClient()
result = client.parse_pdf(Path("problem.pdf"), max_pages=1)
```

---

## Error Messages Reference

| Error | Meaning | Action |
|-------|---------|--------|
| "Docling not available" | Missing dependencies | Install docling |
| "Repetition bug detected" | SmolDocling token loop | Automatic handling |
| "ParseError: No content extracted" | Empty DocTags | Check PDF content |
| "GPU out of memory" | vLLM memory issue | Reduce batch size |
| "Connection refused" | vLLM server down | Start vLLM server |

---

## Performance Considerations

- **DPI Setting**: SmolDocling works best at 144 DPI (not 300)
- **GPU Memory**: Allocate 0.8-0.9 for SmolDocling
- **Timeout**: Allow 30-60s per page for complex documents

---

## Getting Help

1. **Check Logs**: Look for detailed error messages
2. **Test Files**: Try with known-good PDFs first
3. **Minimal Example**: Isolate the issue with a single page
4. **Report Issues**: Include full error trace and PDF sample

---

## Known Limitations

1. **Vector Graphics**: Not extracted as images
2. **Scanned PDFs**: Need OCR pre-processing
3. **Complex Tables**: May be split incorrectly
4. **Mathematical Formulas**: Limited support
5. **Non-Latin Scripts**: Variable quality

---

## Version Compatibility

Tested with:
- docling >= 2.26.0
- docling-core >= 2.0.0
- vLLM >= 0.6.0
- SmolDocling-1.8b-Instruct

Always check version compatibility when upgrading.