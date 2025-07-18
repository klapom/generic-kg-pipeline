# Test Reorganisation Plan

## Aktuelle Situation
- 31 Test-Dateien im Root-Verzeichnis
- 5 Tests bereits in `tests/` Verzeichnis organisiert
- Viele Tests entstanden während vLLM-Debugging

## Vorgeschlagene Struktur

```
tests/
├── archive/                    # vLLM Debugging Tests (nicht mehr benötigt)
│   ├── test_vllm_debug.py
│   ├── test_vllm_loading.py
│   ├── test_vllm_inference.py
│   ├── test_simple_vllm.py
│   ├── test_vllm_direct.py
│   ├── test_transformers_direct.py
│   ├── test_direct_client.py
│   └── test_smoldocling_pdf.py
│
├── unit/                       # Unit Tests
│   ├── parsers/
│   │   ├── test_basic_parse.py
│   │   ├── test_pdfplumber_tables.py
│   │   └── test_simple_text.py
│   ├── clients/
│   │   └── test_vllm_smoldocling.py    # (bereits in tests/)
│   └── core/
│       ├── test_content_chunker.py      # (bereits in tests/)
│       └── test_document_parsers.py     # (bereits in tests/)
│
├── integration/                # Integration Tests
│   ├── test_runner.py
│   ├── test_processing_with_logging.py
│   ├── test_pdf_vllm.py
│   ├── test_complete_bmw_processing.py
│   ├── test_hybrid_parser.py
│   ├── test_advanced_hybrid_parser.py
│   ├── test_full_document_extraction.py
│   └── test_api_endpoints.py           # (bereits in tests/)
│
├── debugging/                  # Debugging/Spezielle Tests (noch nützlich)
│   ├── page_issues/
│   │   ├── test_page2_debug.py
│   │   ├── test_page2_complex.py
│   │   ├── test_pages_6_10.py
│   │   └── test_specific_pages_raw.py
│   ├── extraction/
│   │   ├── test_complex_detection.py
│   │   ├── test_fallback_extraction.py
│   │   ├── test_structured_table_extraction.py
│   │   ├── test_table_separation.py
│   │   ├── test_tables_direct.py
│   │   ├── test_tables_raw_output.py
│   │   └── test_debug_separation.py
│   └── processing/
│       ├── test_bmw_pdf_with_logging.py
│       ├── test_pdf_single_page.py
│       ├── test_raw_output_to_segments.py
│       ├── test_direct_raw_output.py
│       └── test_complete_segment_content.py
│
├── fixtures/                   # Test-Daten und Fixtures
│   ├── __init__.py
│   ├── pdf_fixtures.py
│   └── conftest.py
│
└── production/                 # Produktions-Tests
    └── process_all_pdfs_private.py
```

## Zusätzliche Dateien

Diese Dateien sind keine Tests, sondern Hilfsskripte:
- `create_test_pdf.py` → `scripts/create_test_pdf.py`
- `create_samples.py` → `scripts/create_samples.py`
- `process_documents.py` → `scripts/process_documents.py`
- `process_documents_vllm.py` → `scripts/process_documents_vllm.py`

## Bewertung der bestehenden Tests

### ✅ Gute Tests (behalten und richtig organisieren)
- Die Tests in `tests/` sind bereits gut strukturiert mit pytest
- `test_basic_parse.py` - Solider Unit-Test
- `test_hybrid_parser.py` - Wichtiger Integration-Test
- `test_complete_bmw_processing.py` - Vollständiger E2E-Test

### ⚠️ Debugging-Tests (behalten aber separat)
- Die Page-spezifischen Tests waren wichtig für die Problemlösung
- Können für zukünftige Debugging-Sessions nützlich sein
- Sollten aber klar als "Debugging" markiert sein

### ❌ Archivieren (vLLM-Debugging)
- 8 vLLM-spezifische Debugging-Tests
- Nicht mehr benötigt da vLLM-Probleme gelöst

## Vorteile dieser Struktur

1. **Klare Trennung**: Unit vs Integration vs Debugging
2. **Archive behalten**: vLLM-Tests bleiben verfügbar falls benötigt
3. **Debugging-Tests**: Bleiben für zukünftige Probleme verfügbar
4. **Saubere CI/CD**: Nur unit/ und integration/ in CI-Pipeline
5. **Flexibilität**: Debugging-Tests können bei Bedarf ausgeführt werden

## Migration-Schritte

1. Erstelle neue Verzeichnisstruktur
2. Verschiebe vLLM-Tests nach `archive/`
3. Organisiere Unit-Tests
4. Organisiere Integration-Tests
5. Gruppiere Debugging-Tests thematisch
6. Update `.gitignore` um `tests/archive/` optional zu machen
7. Update CI/CD um nur unit/ und integration/ auszuführen