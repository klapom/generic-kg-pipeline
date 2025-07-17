# Test Organization

## Directory Structure

### `/tests/unit/`
Unit tests that test individual components in isolation.
- `parsers/` - Tests for individual parser implementations
- `test_content_chunker.py` - Tests for content chunking
- `test_document_parsers.py` - Tests for parser factory
- `test_vllm_smoldocling.py` - Tests for vLLM client

### `/tests/integration/`
Integration tests that test multiple components working together.
- `test_runner.py` - Full pipeline test runner
- `test_complete_bmw_processing.py` - End-to-end BMW document processing
- `test_hybrid_parser.py` - Hybrid parser integration
- `test_api_endpoints.py` - API endpoint tests

### `/tests/debugging/`
Debugging and development tests for specific issues. Not included in CI/CD.
- `page_issues/` - Tests for specific page problems
- `extraction/` - Tests for extraction methods
- `processing/` - Tests for processing pipelines

### `/tests/archive/`
Archived vLLM debugging tests from initial development. Kept for reference but not actively maintained.

### `/tests/production/`
Production-ready processing scripts.
- `process_all_pdfs_private.py` - Process all PDFs with privacy protection

### `/tests/fixtures/`
Shared test fixtures and utilities (to be implemented).

## Running Tests

```bash
# Run all unit tests
pytest tests/unit/

# Run all integration tests  
pytest tests/integration/

# Run specific test category
pytest tests/unit/parsers/

# Run with coverage
pytest tests/unit/ tests/integration/ --cov=core --cov=plugins

# Run debugging tests (not in CI)
pytest tests/debugging/
```

## Test Conventions

1. Use pytest for all new tests
2. Use fixtures from `conftest.py` for common setup
3. Follow naming convention: `test_<component>_<scenario>.py`
4. Keep tests focused and isolated
5. Use mocks for external dependencies in unit tests