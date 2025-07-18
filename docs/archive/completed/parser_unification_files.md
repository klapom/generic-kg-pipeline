# Files Requiring Parser Unification Updates

## Core Parser Files
1. **core/parsers/hybrid_pdf_parser.py** - Main hybrid parser implementation
2. **core/parsers/parser_factory.py** - Factory for creating parsers
3. **core/parsers/__init__.py** - Parser module exports
4. **core/parsers/implementations/__init__.py** - Implementation exports
5. **core/parsers/implementations/pdf/__init__.py** - PDF parser exports
6. **core/parsers/implementations/pdf/standard_pdf_parser.py** - Standard PDF parser
7. **core/parsers/implementations/pdf/hybrid_pdf_parser.py** - Hybrid PDF parser implementation

## Scripts
1. **scripts/process_documents.py** - Main document processing script
   - Uses ParserFactory to create parsers
   - Calls parse_document() method

2. **scripts/process_documents_debug.py** - Debug version of processing script
   - Creates HybridPDFParser directly for debugging
   - Uses ParserFactory for normal processing

3. **scripts/process_documents_debug_enhanced.py** - Enhanced debug script
   - Similar to debug script, creates HybridPDFParser directly
   - Uses ParserFactory

4. **scripts/test_content_extraction.py** - Content extraction test script
   - Creates HybridPDFParser directly
   - Calls parse() method

5. **scripts/process_documents_vllm.py** - vLLM document processing script
   - Uses VLLMBatchProcessor which internally uses parsers

## Batch Processors
1. **core/batch_processor.py** - Main batch processing logic
   - Creates ParserFactory instance
   - Calls parse_document() from factory

2. **core/vllm_batch_processor.py** - vLLM-specific batch processor
   - Creates ParserFactory instance
   - Has special handling for PDFs with SmolDocling
   - Falls back to parser factory for other formats

3. **core/vlm/batch_processor.py** - Visual Language Model batch processor
   - Creates HybridPDFParser directly
   - Calls parse() method

## API Endpoints
1. **api/routers/documents.py** - Document upload and processing API
   - Currently has placeholder for parser integration
   - Needs to use unified parser for document processing

## Archive/Legacy Files (May need consideration)
1. **archiv/process_bmw_docs.py** - BMW document processing
   - Uses ParserFactory

## Key Changes Needed

### 1. Standardize Parser Creation
- All files should use ParserFactory instead of creating parsers directly
- Exception: VLM batch processor may need special handling

### 2. Standardize Parse Method Calls
- Use `parse_document()` from ParserFactory consistently
- Avoid direct `parse()` calls on parser instances

### 3. Update Imports
- Change from: `from core.parsers.implementations.pdf import HybridPDFParser`
- To: `from core.parsers import ParserFactory`

### 4. Configuration Handling
- Ensure consistent configuration passing through ParserFactory
- Handle VLM enablement through factory configuration

### 5. Special Cases
- **VLLMBatchProcessor**: Has special SmolDocling integration for PDFs
- **VLM Batch Processor**: May need direct parser access for visual processing
- **Debug Scripts**: May keep direct parser creation for debugging purposes

## Priority Files for Update
1. Core parser implementations (hybrid_pdf_parser.py, standard_pdf_parser.py)
2. ParserFactory - ensure it properly handles all cases
3. Batch processors - main entry points for document processing
4. Scripts - user-facing interfaces
5. API endpoints - external interfaces