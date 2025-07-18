# Current Features Overview - Generic Knowledge Graph Pipeline

*Last updated: July 2025*

## 🎯 Core Capabilities

### 1. Document Processing Pipeline
- **Multi-format Support**: PDF, DOCX, XLSX, PPTX, TXT
- **Unified Parser Factory**: Single entry point for all document types
- **Hybrid PDF Parser**: SmolDocling + fallback extractors

### 2. Advanced PDF Processing with vLLM
- **SmolDocling Integration**: 
  - Vision-Language Model for document layout understanding
  - DocTags format for structured extraction
  - Automatic repetition bug detection and handling
- **Direct Image Extraction**: Via docling integration
- **Bounding Box Preservation**: Accurate visual element positioning

### 3. Visual Language Models (VLM)
- **Two-Stage VLM Strategy**:
  - Stage 1: Fast triage with smaller models
  - Stage 2: Deep analysis with specialized models
- **Supported Models**:
  - Qwen2.5-VL (primary)
  - LLaVA (comprehensive analysis)
  - Pixtral (fallback)
- **Smart Model Selection**: Based on confidence and element type

### 4. Content Chunking & Context
- **Intelligent Chunking**:
  - Respects document boundaries
  - Token-based sizing with overlap
  - Context inheritance between chunks
- **Context Groups**: Maintains relationships between related chunks
- **Performance Optimization**: Async processing with batching

### 5. Knowledge Graph Generation
- **Hochschul-LLM Integration**: 
  - Triple extraction from text
  - Domain-specific ontology support
  - Batch processing capabilities
- **TableToTripleConverter**: Structured data to RDF/Turtle (available but not integrated)

### 6. API & Integration
- **FastAPI-based REST API**:
  - Document upload endpoints
  - Pipeline control
  - Query and export functionality
- **WebSocket Support**: Real-time updates
- **Async Architecture**: High performance processing

## 🔧 Technical Infrastructure

### Configuration System
- **Unified config.yaml**: Single source of truth
- **Environment Variable Support**: `${VAR:default}` syntax
- **Hot-reload Capability**: Changes without restart
- **Type-safe with Pydantic**: Validation and defaults

### Debugging & Monitoring
- **Multi-level Debug Pipeline**:
  - Basic: Metrics and timings
  - Detailed: Full content tracking
  - Full: Complete data dumps
- **HTML Report Generation**: Visual analysis reports
- **Performance Tracking**: Processing times per stage

### Error Handling
- **SmolDocling Repetition Bug**: Automatic detection and truncation
- **DocTags Compatibility**: Automatic tag transformation
- **Graceful Fallbacks**: No hard failures

## 📚 Available Guides

1. **Setup & Configuration**:
   - VLLM_SETUP.md - Complete vLLM installation
   - VLLM_CONFIGURATION.md - Quick configuration guide
   - HOCHSCHUL_LLM.md - LLM service setup

2. **Implementation Guides**:
   - DEBUG_PIPELINE.md - Debug system usage
   - vlm_optimization_strategy.md - VLM deployment optimization
   - chunking_implementation_plan.md - Chunking system details

3. **Troubleshooting**:
   - DOCLING_TROUBLESHOOTING.md - Comprehensive issue resolution
   - SMOLDOCLING_QUICK_TROUBLESHOOTING.md - Quick reference

4. **Technical References**:
   - ARCHITECTURE.md - System architecture overview
   - API_DOCUMENTATION.md - REST API reference
   - document_structure_analysis.md - Data model documentation

## 🚀 Next Potential Features

Based on INTEGRATION_PLAN.md analysis:

### High Value Additions:
1. **TableToTripleConverter Integration** - Already implemented, just needs activation
2. **Enhanced Monitoring** - Performance metrics and error tracking
3. **Batch Processing UI** - Web interface for bulk document processing

### Lower Priority:
- Camelot table extraction (SmolDocling handles tables well)
- Regex-based extractors (current extraction is sufficient)
- ML-based parser selection (premature optimization)

## 📊 Current Status

- **Core Pipeline**: ✅ Fully operational
- **Legacy Code**: ✅ Completely removed
- **Parser Unification**: ✅ Completed
- **Documentation**: ✅ Cleaned and organized
- **Tests**: ✅ All passing after consolidation