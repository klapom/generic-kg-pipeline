# Parser-Hierarchie Refactoring Plan

## 🔍 Aktuelle Situation

### Probleme:
1. **Zwei Parser-Verzeichnisse**:
   - `plugins/parsers/` - Haupt-Parser (PDF, DOCX, XLSX, etc.)
   - `core/parsers/` - Spezialisierte PDF-Parser (Hybrid, Advanced)

2. **Verwirrende Abhängigkeiten**:
   - `core/parsers/hybrid_pdf_parser.py` importiert von `plugins/parsers/`
   - Zirkuläre/cross-directory Imports möglich

3. **Unklare Verantwortlichkeiten**:
   - Warum ist HybridPDFParser in core/ statt bei anderen PDF-Parsern?
   - Wo sollten neue Parser hinzugefügt werden?

### Aktuelle Struktur:
```
plugins/parsers/
├── base_parser.py          # BaseParser ABC + Datenmodelle
├── parser_factory.py       # Factory für Parser-Erstellung
├── pdf_parser.py          # Standard PDF Parser
├── docx_parser.py         # DOCX Parser
├── xlsx_parser.py         # Excel Parser
├── pptx_parser.py         # PowerPoint Parser
├── txt_parser.py          # Text Parser
├── smoldocling_context_parser.py  # Context-aware Parser
└── context_mapping.py     # Context utilities

core/parsers/
├── hybrid_pdf_parser.py    # Hybrid PDF mit SmolDocling
├── advanced_pdf_extractor.py # Advanced PDF features
├── fallback_extractors.py  # PyPDF2/pdfplumber fallbacks
├── table_text_separator.py # Tabellen-Trennung
└── table_to_triples.py    # Tabellen zu RDF Triples
```

## 🎯 Ziel-Struktur

```
core/parsers/
├── interfaces/
│   ├── __init__.py
│   ├── base_parser.py      # BaseParser ABC
│   ├── parser_protocol.py  # Parser Protocol für typing
│   └── data_models.py      # Document, Segment, etc.
│
├── implementations/
│   ├── __init__.py
│   ├── pdf/
│   │   ├── __init__.py
│   │   ├── standard_pdf_parser.py    # Ehemals pdf_parser.py
│   │   ├── hybrid_pdf_parser.py      # Mit SmolDocling
│   │   ├── advanced_pdf_parser.py    # Advanced features
│   │   └── extractors/
│   │       ├── __init__.py
│   │       ├── pypdf2_extractor.py
│   │       ├── pdfplumber_extractor.py
│   │       └── table_extractor.py
│   │
│   ├── office/
│   │   ├── __init__.py
│   │   ├── docx_parser.py
│   │   ├── xlsx_parser.py
│   │   └── pptx_parser.py
│   │
│   └── text/
│       ├── __init__.py
│       └── txt_parser.py
│
├── strategies/
│   ├── __init__.py
│   ├── table_text_separator.py
│   ├── table_to_triples.py
│   ├── context_mapping.py
│   └── smoldocling_context.py
│
├── factory.py              # Parser Factory
└── __init__.py            # Exports für einfachen Import
```

## 📋 Migrations-Plan

### Phase 1: Vorbereitung (Ohne Breaking Changes)
1. **Neue Struktur erstellen**
   ```bash
   core/parsers/interfaces/
   core/parsers/implementations/pdf/
   core/parsers/implementations/office/
   core/parsers/implementations/text/
   core/parsers/strategies/
   ```

2. **Basis-Module kopieren**
   - `plugins/parsers/base_parser.py` → `core/parsers/interfaces/`
   - Aufteilen in:
     - `base_parser.py` (nur ABC)
     - `data_models.py` (Document, Segment, etc.)

### Phase 2: Parser Migration (Mit Kompatibilität)

#### 2.1 PDF Parser
```python
# core/parsers/implementations/pdf/__init__.py
# Re-export für Kompatibilität
from .standard_pdf_parser import PDFParser
from .hybrid_pdf_parser import HybridPDFParser
from .advanced_pdf_parser import AdvancedPDFParser

__all__ = ['PDFParser', 'HybridPDFParser', 'AdvancedPDFParser']
```

```python
# plugins/parsers/pdf_parser.py (Kompatibilitäts-Stub)
# Temporär für backward compatibility
import warnings
warnings.warn(
    "Import from plugins.parsers.pdf_parser is deprecated. "
    "Use core.parsers.implementations.pdf instead.",
    DeprecationWarning,
    stacklevel=2
)
from core.parsers.implementations.pdf import PDFParser
```

#### 2.2 Office Parser
- Gleiche Strategie für DOCX, XLSX, PPTX

#### 2.3 Strategies
- `table_text_separator.py` → `core/parsers/strategies/`
- `context_mapping.py` → `core/parsers/strategies/`

### Phase 3: Factory Update
```python
# core/parsers/factory.py
from typing import Dict, Type
from .interfaces import BaseParser
from .implementations.pdf import PDFParser, HybridPDFParser
from .implementations.office import DOCXParser, XLSXParser, PPTXParser
from .implementations.text import TXTParser

PARSER_REGISTRY: Dict[str, Type[BaseParser]] = {
    'pdf': PDFParser,
    'pdf_hybrid': HybridPDFParser,
    'docx': DOCXParser,
    'xlsx': XLSXParser,
    'pptx': PPTXParser,
    'txt': TXTParser,
}

def get_parser(file_type: str, use_hybrid: bool = False) -> BaseParser:
    """Get appropriate parser for file type"""
    if file_type == 'pdf' and use_hybrid:
        return PARSER_REGISTRY['pdf_hybrid']()
    return PARSER_REGISTRY[file_type]()
```

### Phase 4: Import Updates
```python
# Alte Imports:
from plugins.parsers.pdf_parser import PDFParser
from plugins.parsers.base_parser import Document, Segment

# Neue Imports:
from core.parsers.implementations.pdf import PDFParser
from core.parsers.interfaces import Document, Segment

# Oder einfacher:
from core.parsers import PDFParser, Document, Segment
```

### Phase 5: Cleanup
1. Tests aktualisieren
2. Deprecation warnings entfernen
3. Alte `plugins/parsers/` Struktur entfernen
4. Dokumentation aktualisieren

## 🔧 Implementierungs-Details

### Kompatibilitäts-Layer
```python
# core/parsers/__init__.py
# Convenience exports für einfache Imports
from .interfaces import BaseParser, Document, Segment, DocumentMetadata
from .interfaces import DocumentType, VisualElementType, ParseError

from .implementations.pdf import PDFParser, HybridPDFParser
from .implementations.office import DOCXParser, XLSXParser, PPTXParser
from .implementations.text import TXTParser

from .factory import get_parser, register_parser

__all__ = [
    # Interfaces
    'BaseParser', 'Document', 'Segment', 'DocumentMetadata',
    'DocumentType', 'VisualElementType', 'ParseError',
    
    # Implementations
    'PDFParser', 'HybridPDFParser',
    'DOCXParser', 'XLSXParser', 'PPTXParser',
    'TXTParser',
    
    # Factory
    'get_parser', 'register_parser',
]
```

### Parser Protocol (für Type Hints)
```python
# core/parsers/interfaces/parser_protocol.py
from typing import Protocol, runtime_checkable
from pathlib import Path
from .data_models import Document

@runtime_checkable
class ParserProtocol(Protocol):
    """Protocol for type checking parsers"""
    def parse(self, file_path: Path) -> Document: ...
    def can_parse(self, file_path: Path) -> bool: ...
```

## ⏰ Zeitplan

- **Tag 1**: Neue Struktur erstellen, Interfaces migrieren
- **Tag 2**: PDF Parser migrieren (inkl. Tests)
- **Tag 3**: Office + Text Parser migrieren
- **Tag 4**: Factory + Import Updates
- **Tag 5**: Tests + Dokumentation

## ✅ Vorteile

1. **Klare Struktur**: Sofort ersichtlich wo neue Parser hingehören
2. **Bessere Organisation**: Verwandte Parser zusammen (alle PDF-Parser in einem Ordner)
3. **Einfachere Imports**: `from core.parsers import PDFParser`
4. **Erweiterbarkeit**: Neue Parser-Typen einfach hinzufügen
5. **Testbarkeit**: Parser-spezifische Tests in entsprechenden Unterordnern

## 🚨 Risiken & Mitigation

1. **Breaking Changes**
   - Mitigation: Kompatibilitäts-Stubs in `plugins/parsers/`
   - Deprecation warnings vor Entfernung

2. **Import-Zyklen**
   - Mitigation: Klare Hierarchie, keine cross-imports zwischen implementations

3. **Test-Failures**
   - Mitigation: Schrittweise Migration, Tests nach jedem Schritt

## 📝 Migrations-Checkliste

- [ ] Neue Verzeichnisstruktur erstellen
- [ ] Interfaces extrahieren und migrieren
- [ ] PDF Parser migrieren
  - [ ] standard_pdf_parser.py
  - [ ] hybrid_pdf_parser.py
  - [ ] advanced_pdf_parser.py
  - [ ] Extractors
- [ ] Office Parser migrieren
  - [ ] docx_parser.py
  - [ ] xlsx_parser.py
  - [ ] pptx_parser.py
- [ ] Text Parser migrieren
- [ ] Strategies migrieren
- [ ] Factory aktualisieren
- [ ] Kompatibilitäts-Layer erstellen
- [ ] Import-Statements projekt-weit aktualisieren
- [ ] Tests anpassen
- [ ] Dokumentation aktualisieren
- [ ] Alte Struktur entfernen