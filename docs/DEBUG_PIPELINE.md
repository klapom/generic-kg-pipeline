# Debug Pipeline Documentation

## Overview

The Debug Pipeline provides comprehensive logging, tracking, and visualization capabilities for the production document processing pipeline. It tracks segments, chunks, VLM descriptions, and generates detailed HTML reports for analysis.

## Features

### 1. **Multi-Level Debugging**
- **none**: Production mode with minimal logging
- **basic**: Track metrics and timings
- **detailed**: Full tracking of segments, chunks, and VLM processing
- **full**: Complete debugging with all intermediate data and images

### 2. **HTML Report Generation**
Generates interactive HTML reports showing:
- Pipeline flow visualization
- Segment-by-segment analysis
- VLM descriptions with confidence scores
- Chunk formation and context inheritance
- Performance metrics for each stage
- Error and warning tracking

### 3. **VLM Processing Tracking**
- Two-stage VLM processing (Qwen2.5-VL → Pixtral)
- Confidence evaluation for each analysis
- Automatic fallback for low-confidence results
- Model performance comparison

### 4. **Performance Monitoring**
- Stage-wise timing breakdown
- Memory usage tracking
- Processing bottleneck identification
- Batch processing statistics

## Usage

### Basic Usage

```bash
# Process with detailed debugging and HTML report
python scripts/process_documents_debug.py --debug-level detailed --html-report

# Process specific file with full debugging
python scripts/process_documents_debug.py --file BMW_X5.pdf --debug-level full

# Process pattern with basic debugging
python scripts/process_documents_debug.py --pattern "BMW*.pdf" --debug-level basic
```

### Command Line Options

```
--debug-level {none,basic,detailed,full}
    Debug level for pipeline execution (default: basic)

--html-report
    Generate HTML analysis report

--debug-dir DIR
    Directory for debug output (default: data/debug)

--file FILE
    Process specific file

--pattern PATTERN
    File pattern to match (e.g., 'BMW*.pdf')

--no-vlm
    Disable VLM processing

--save-intermediate
    Save intermediate processing results
```

### Python API

```python
from core.pipeline_debugger import PipelineDebugConfig, DebugLevel
from scripts.process_documents_debug import DebugDocumentProcessor

# Create debug configuration
debug_config = PipelineDebugConfig(
    debug_level=DebugLevel.DETAILED,
    generate_html_report=True,
    track_segments=True,
    track_chunks=True,
    track_vlm_descriptions=True,
    save_intermediate_results=True,
    output_dir=Path("data/debug")
)

# Initialize processor
processor = DebugDocumentProcessor(debug_config)

# Process document
result = await processor.process_file_with_debug(
    Path("document.pdf"), 
    enable_vlm=True
)
```

## HTML Report Structure

The generated HTML report includes:

### 1. **Processing Summary**
- Total processing time
- Stage-wise timing breakdown
- Document statistics (pages, segments, chunks)
- VLM analysis summary

### 2. **Pipeline Flow Visualization**
Interactive visualization showing:
```
PDF Input → SmolDocling → VLM Analysis → Chunking → Output
```

### 3. **Segments Analysis**
- Filterable segment list (All, With VLM, Text Only, Visual)
- Content preview for each segment
- VLM descriptions with confidence scores
- Processing metadata

### 4. **Chunks Analysis**
- Chunk content and token counts
- Source segment mapping
- Context inheritance visualization
- Chunk type classification

### 5. **Issues Section**
- Errors encountered during processing
- Warnings and recommendations
- Performance bottlenecks

## Production Use Cases

### 1. **Quality Assurance**
```bash
# Check document processing quality
python scripts/process_documents_debug.py --file critical_document.pdf --debug-level full --html-report
```

### 2. **Performance Optimization**
```bash
# Identify bottlenecks
python scripts/process_documents_debug.py --pattern "*.pdf" --debug-level basic --html-report
```

### 3. **VLM Model Comparison**
```bash
# Compare VLM model performance
python scripts/process_documents_debug.py --file test_doc.pdf --debug-level detailed --html-report
```

### 4. **Troubleshooting**
```bash
# Debug processing issues
python scripts/process_documents_debug.py --file problematic.pdf --debug-level full --save-intermediate
```

## Output Files

### Debug Directory Structure
```
data/debug/
├── document_analysis_20240715_143022.html    # HTML report
├── document_debug_20240715_143022.json       # Debug data (if enabled)
└── batch/                                     # Batch processing results
    ├── doc1_analysis_20240715_143022.html
    ├── doc2_analysis_20240715_143022.html
    └── processing_summary.json
```

### JSON Debug Data
When `save_intermediate_results` is enabled:
```json
{
  "document_id": "document_20240715_143022",
  "timings": {
    "parsing": 2.34,
    "vlm_processing": 15.67,
    "chunking": 1.23,
    "total": 19.24
  },
  "stats": {
    "total_segments": 45,
    "total_visual_elements": 8,
    "total_chunks": 23
  },
  "segments": [...],
  "chunks": [...],
  "errors": [],
  "warnings": []
}
```

## Best Practices

### 1. **Development**
- Use `detailed` level during development
- Enable HTML reports for visual debugging
- Save intermediate results for troubleshooting

### 2. **Testing**
- Use `basic` level for performance testing
- Compare results across different debug levels
- Monitor memory usage with full debugging

### 3. **Production**
- Use `none` or `basic` level in production
- Enable detailed debugging only for critical documents
- Set up automated report generation for QA

### 4. **Troubleshooting**
- Start with `basic` level to identify issues
- Increase to `detailed` or `full` for specific problems
- Use pattern matching to isolate problematic documents

## Integration with CI/CD

```yaml
# Example GitHub Actions workflow
- name: Process Documents with Debug
  run: |
    python scripts/process_documents_debug.py \
      --pattern "test/*.pdf" \
      --debug-level basic \
      --html-report \
      --debug-dir artifacts/debug

- name: Upload Debug Reports
  uses: actions/upload-artifact@v3
  with:
    name: debug-reports
    path: artifacts/debug/*.html
```

## Performance Considerations

- **Debug Level Impact**:
  - `none`: ~0% overhead
  - `basic`: ~5% overhead
  - `detailed`: ~10-15% overhead
  - `full`: ~20-30% overhead

- **Memory Usage**:
  - Basic tracking: +50-100MB
  - Detailed tracking: +200-500MB
  - Full with images: +1-2GB per document

## Future Enhancements

1. **Real-time Monitoring**
   - WebSocket-based live tracking
   - Progress visualization
   - Resource usage graphs

2. **Advanced Analytics**
   - Cross-document comparisons
   - Trend analysis over time
   - Model performance benchmarking

3. **Export Options**
   - PDF report generation
   - CSV data export
   - Integration with monitoring tools

4. **Configuration Profiles**
   - Pre-defined debug profiles
   - Environment-specific settings
   - Automatic profile selection