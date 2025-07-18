# Refactoring Plan - Generic Knowledge Graph Pipeline

## Übersicht

Nach gründlicher Analyse der Codebase wurden mehrere kritische Bereiche identifiziert, die von einem Refactoring profitieren würden. Dieser Plan priorisiert die Änderungen nach Dringlichkeit und Auswirkung.

## 🚨 Kritische Probleme (Sofort angehen)

### 1. Test-Organisation (Highest Priority)
**Problem**: 31 Test-Dateien im Root-Verzeichnis ohne Struktur
**Auswirkung**: Schlechte Wartbarkeit, duplizierter Code, unklare Test-Coverage

**Lösung**:
```
tests/
├── unit/
│   ├── parsers/
│   │   ├── test_pdf_parser.py
│   │   ├── test_hybrid_parser.py
│   │   ├── test_table_extraction.py
│   │   └── test_smoldocling_parser.py
│   ├── clients/
│   │   ├── test_vllm_smoldocling.py
│   │   ├── test_vllm_qwen.py
│   │   └── test_hochschul_llm.py
│   └── core/
│       ├── test_chunking.py
│       ├── test_config.py
│       └── test_batch_processor.py
├── integration/
│   ├── test_document_pipeline.py
│   ├── test_model_loading.py
│   └── test_end_to_end.py
├── fixtures/
│   ├── __init__.py
│   ├── pdf_fixtures.py
│   ├── model_fixtures.py
│   └── sample_data.py
└── conftest.py  # Shared pytest configuration
```

**Zeitaufwand**: 2-3 Tage

### 2. Code-Duplikation in Tests
**Problem**: Model-Loading und Logger-Setup in jedem Test wiederholt
**Auswirkung**: 20+ Duplikationen, schwer zu warten

**Lösung** - Shared Fixtures erstellen:
```python
# tests/conftest.py
import pytest
from core.vllm.model_manager import VLLMModelManager

@pytest.fixture(scope="session")
def model_manager():
    """Shared model manager for all tests"""
    manager = VLLMModelManager()
    yield manager
    manager.cleanup()

@pytest.fixture
def configured_logger():
    """Standard logger configuration"""
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    return logging.getLogger(__name__)

@pytest.fixture
def sample_pdf_path():
    """Standard test PDF"""
    return Path("data/input/test_simple.pdf")
```

**Zeitaufwand**: 1-2 Tage

## ⚠️ Wichtige Verbesserungen (Kurzfristig: 1-2 Wochen)

### 3. Konfigurationsmanagement vereinheitlichen
**Problem**: Verschiedene Config-Patterns, keine Validierung
**Auswirkung**: Fehleranfällig, inkonsistent

**Lösung**:
```python
# core/config/manager.py
from pydantic import BaseModel
from typing import Dict, Type, Any

class ConfigurationManager:
    """Zentralisiertes Konfigurationsmanagement"""
    
    def __init__(self):
        self._configs: Dict[str, BaseModel] = {}
        self._profiles = {
            'development': self._load_dev_config,
            'testing': self._load_test_config,
            'production': self._load_prod_config
        }
    
    def load_profile(self, profile: str) -> None:
        """Lade Konfigurations-Profil"""
        if profile in self._profiles:
            self._profiles[profile]()
    
    def get_config(self, name: str, config_type: Type[BaseModel]) -> BaseModel:
        """Hole validierte Konfiguration"""
        if name not in self._configs:
            self._configs[name] = self._load_config(name, config_type)
        return self._configs[name]
```

**Zeitaufwand**: 3-4 Tage

### 4. Client-Architektur standardisieren
**Problem**: Inkonsistente Client-Interfaces
**Auswirkung**: Schwer erweiterbar, keine einheitliche Fehlerbehandlung

**Lösung**:
```python
# core/clients/base.py
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

RequestType = TypeVar('RequestType')
ResponseType = TypeVar('ResponseType')

class BaseModelClient(ABC, Generic[RequestType, ResponseType]):
    """Basis für alle Model-Clients"""
    
    def __init__(self, config: BaseModel):
        self.config = config
        self._validate_config()
    
    @abstractmethod
    async def process(self, request: RequestType) -> ResponseType:
        """Verarbeite Anfrage"""
        pass
    
    @abstractmethod
    def _validate_config(self) -> None:
        """Validiere Client-Konfiguration"""
        pass
    
    async def health_check(self) -> bool:
        """Prüfe Client-Verfügbarkeit"""
        return True
```

**Zeitaufwand**: 2-3 Tage

### 5. Parser-Hierarchie klären
**Problem**: Verwirrende Vererbung zwischen plugins/ und core/
**Auswirkung**: Unklar, wo neue Parser hingehören

**Lösung**:
```
core/parsers/
├── interfaces/
│   └── parser_protocol.py      # Protocol/Interface definitions
├── implementations/
│   ├── pdf/
│   │   ├── base.py
│   │   ├── hybrid.py
│   │   ├── native.py
│   │   └── smoldocling.py
│   ├── office/
│   │   ├── docx.py
│   │   └── xlsx.py
│   └── text/
│       └── txt.py
├── strategies/
│   ├── table_extraction.py
│   ├── layout_detection.py
│   └── text_separation.py
└── factory.py                  # Parser factory
```

**Zeitaufwand**: 3-4 Tage

## 📈 Mittelfristige Verbesserungen (1 Monat)

### 6. Domain-Driven Design einführen
**Problem**: Business-Logik mit Infrastruktur vermischt
**Lösung**: Klare Schichten-Architektur

```
src/
├── domain/              # Reine Business-Logik
│   ├── entities/       # Document, Segment, etc.
│   ├── value_objects/  # DocumentType, SegmentType
│   ├── services/       # DocumentProcessor, TripleGenerator
│   └── repositories/   # Interfaces only
├── application/        # Use Cases
│   ├── commands/      # ProcessDocumentCommand
│   └── queries/       # GetDocumentStatusQuery
├── infrastructure/    # Externe Abhängigkeiten
│   ├── parsers/      # Konkrete Parser-Implementierungen
│   ├── clients/      # HTTP/Model Clients
│   └── persistence/  # File/DB Zugriff
└── presentation/     # API Layer
    ├── api/         # FastAPI Routers
    └── cli/         # Command Line Interface
```

**Zeitaufwand**: 2-3 Wochen

### 7. Batch-Processing Pipeline
**Problem**: Keine einheitliche Pipeline-Abstraktion
**Lösung**: Pipeline-Pattern implementieren

```python
# core/pipeline/base.py
class Pipeline:
    def __init__(self, stages: List[Stage]):
        self.stages = stages
    
    async def execute(self, input_data: Any) -> Any:
        result = input_data
        for stage in self.stages:
            result = await stage.process(result)
        return result

# Verwendung:
pipeline = Pipeline([
    LoadDocumentStage(),
    ParseDocumentStage(),
    ChunkDocumentStage(),
    GenerateTriplesStage(),
    SaveResultsStage()
])
```

**Zeitaufwand**: 1 Woche

## 🎯 Priorisierte Umsetzung

### Phase 1: Sofort (Diese Woche)
1. **Test-Reorganisation beginnen**
   - Erstelle tests/ Struktur
   - Verschiebe Tests schrittweise
   - Erstelle conftest.py mit shared fixtures

2. **Code-Duplikation in Tests eliminieren**
   - Implementiere pytest fixtures
   - Refactore erste 5 Tests als Beispiel

### Phase 2: Nächste Woche
3. **Konfigurationsmanagement**
   - Implementiere ConfigurationManager
   - Migriere erste Module

4. **Client-Standardisierung**
   - Erstelle BaseModelClient
   - Refactore einen Client als Beispiel

### Phase 3: Kommende Wochen
5. **Parser-Hierarchie**
6. **Domain-Driven Design** (schrittweise)
7. **Pipeline-Abstraktion**

## 💡 Empfehlungen

1. **Inkrementelles Vorgehen**: Nicht alles auf einmal ändern
2. **Test-First**: Erst Tests schreiben, dann refactoren
3. **Feature-Freeze**: Während kritischer Refactorings keine neuen Features
4. **Code-Reviews**: Jede Änderung reviewen lassen
5. **Dokumentation**: Architektur-Entscheidungen dokumentieren

## 🚫 Was wir NICHT tun sollten

1. **Keine Big-Bang Refactorings** - Schritt für Schritt
2. **Keine Breaking Changes** ohne Migrations-Plan
3. **Kein Over-Engineering** - YAGNI Prinzip beachten
4. **Keine Perfektionismus** - 80/20 Regel anwenden

## 📊 Erfolgs-Metriken

- Test-Ausführungszeit reduziert um 50%
- Code-Duplikation reduziert um 70%
- Neue Features 2x schneller implementierbar
- Onboarding neuer Entwickler von 2 Wochen auf 3 Tage

## Nächste Schritte

1. **Diskussion** dieses Plans
2. **Priorisierung** anpassen nach Projekt-Bedürfnissen
3. **Zeitplan** festlegen
4. **Team-Zuweisung** für verschiedene Bereiche
5. **Start** mit Phase 1

---

Dieser Plan ist ein Vorschlag und sollte mit dem Team diskutiert und angepasst werden.