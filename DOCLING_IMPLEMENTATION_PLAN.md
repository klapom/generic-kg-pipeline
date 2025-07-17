# Docling Integration - Detaillierter Implementierungsplan

## 📅 Zeitplan: 2-3 Wochen

## Phase 1: Vorbereitung und Setup (Tag 1-2)

### 1.1 Dependencies installieren
```bash
# requirements.txt erweitern
docling>=2.26.0
docling-core>=2.0.0

# Installation
uv pip install docling docling-core
```

### 1.2 Projekt-Setup
- [ ] Branch erstellen: `feature/docling-integration`
- [ ] Backup der aktuellen Implementierung
- [ ] Test-PDFs sammeln (verschiedene Typen)
- [ ] Performance-Baseline messen

### 1.3 Konfiguration vorbereiten
```python
# config/docling_config.py
DOCLING_CONFIG = {
    "use_docling": False,  # Feature Flag
    "extract_images_directly": True,
    "fallback_to_legacy": True,
    "log_performance": True
}
```

## Phase 2: Core Implementation (Tag 3-7)

### 2.1 Neue SmolDocling Client Klasse
```
Datei: core/clients/vllm_smoldocling_final.py
```

**Hauptfunktionen:**
1. **Docling Check**
   ```python
   def _check_docling_available() -> bool
   ```

2. **Direkte Bildextraktion**
   ```python
   def _parse_with_docling_direct(pdf_path) -> SmolDoclingResult
   ```

3. **Fallback Mechanismus**
   ```python
   def parse_pdf(pdf_path) -> SmolDoclingResult:
       if use_docling and available:
           return _parse_with_docling_direct()
       else:
           return _parse_legacy()
   ```

### 2.2 Erweiterte Datenstrukturen
```python
@dataclass
class SmolDoclingPage:
    # Bestehende Felder
    page_number: int
    text: str
    tables: List[Dict]
    images: List[Dict]
    
    # NEU: Direkte Visual Elements
    visual_elements: List[VisualElement]
    _docling_doc: Optional[DoclingDocument]
```

### 2.3 Bildextraktion während Parsing
```python
def _extract_visuals_direct():
    # 1. Docling Element durchgehen
    # 2. bbox extrahieren
    # 3. Bild direkt aus PDF extrahieren
    # 4. VisualElement mit raw_data erstellen
```

## Phase 3: Integration in Pipeline (Tag 8-10)

### 3.1 HybridPDFParser anpassen
```python
# core/parsers/implementations/pdf/hybrid_pdf_parser.py

def parse(self, file_path: Path) -> Document:
    # NEU: Check ob visual_elements bereits raw_data haben
    if self.use_docling:
        result = self.docling_client.parse_pdf(file_path)
        # visual_elements haben bereits Bilder!
    else:
        # Legacy: 2-Phasen Ansatz
        result = self.smoldocling_client.parse_pdf(file_path)
        self._extract_image_bytes(file_path, visual_elements)
```

### 3.2 Konfiguration erweitern
```yaml
# config/pipeline_config.yaml
pdf_parser:
  use_docling: true
  docling_config:
    extract_images_directly: true
    max_image_size: 2048
    image_quality: 95
```

### 3.3 Factory Pattern für Client-Auswahl
```python
class SmolDoclingClientFactory:
    @staticmethod
    def create_client(config: dict) -> BaseSmolDoclingClient:
        if config.get("use_docling"):
            return VLLMSmolDoclingFinalClient(**config)
        else:
            return VLLMSmolDoclingClient(**config)
```

## Phase 4: Testing (Tag 11-12)

### 4.1 Unit Tests
```python
# tests/test_docling_final.py

def test_bbox_preservation():
    # Vergleiche bbox aus legacy vs docling

def test_image_extraction_quality():
    # Vergleiche extrahierte Bilder

def test_fallback_mechanism():
    # Test wenn docling fehlt

def test_performance():
    # Zeitmessung legacy vs docling
```

### 4.2 Integration Tests
```python
def test_full_pipeline_with_docling():
    # PDF → Docling → VLM → Knowledge Graph
    
def test_backward_compatibility():
    # Alte PDFs müssen weiterhin funktionieren
```

### 4.3 Test-Matrix
| Test Case | Legacy | Docling | Expected |
|-----------|---------|---------|----------|
| BMW PDF | ✅ | ✅ | Identische Ausgabe |
| Complex Layout | ✅ | ✅ | Bessere Struktur |
| Performance | Baseline | -15% | Schneller |
| Memory Usage | Baseline | +10% | Akzeptabel |

## Phase 5: Validierung und Benchmarking (Tag 13-14)

### 5.1 Performance Tests
```python
# benchmarks/docling_performance.py

def benchmark_extraction_methods():
    pdfs = ["bmw.pdf", "complex.pdf", "large.pdf"]
    
    for pdf in pdfs:
        # Legacy timing
        legacy_time = measure_legacy(pdf)
        
        # Docling timing
        docling_time = measure_docling(pdf)
        
        # Report
        print(f"{pdf}: Legacy={legacy_time}s, Docling={docling_time}s")
```

### 5.2 Qualitäts-Validierung
- [ ] Visuelle Inspektion extrahierter Bilder
- [ ] VLM-Ausgabe vergleichen
- [ ] Chunking-Ergebnisse prüfen
- [ ] Knowledge Graph Integrität

### 5.3 Memory Profiling
```python
import tracemalloc

tracemalloc.start()
# Parse with docling
current, peak = tracemalloc.get_traced_memory()
```

## Phase 6: Rollout (Tag 15+)

### 6.1 Schrittweise Aktivierung
```python
# Woche 1: 10% Traffic
DOCLING_CONFIG["rollout_percentage"] = 10

# Woche 2: 50% Traffic
DOCLING_CONFIG["rollout_percentage"] = 50

# Woche 3: 100% Traffic
DOCLING_CONFIG["rollout_percentage"] = 100
```

### 6.2 Monitoring
- [ ] Error Rate Dashboard
- [ ] Performance Metriken
- [ ] Memory Usage Graphs
- [ ] User Feedback

### 6.3 Rollback Plan
```python
# Bei Problemen:
DOCLING_CONFIG["use_docling"] = False
# Automatischer Fallback auf Legacy
```

## 📊 Erfolgs-Kriterien

1. **Performance**: ≥10% schneller als Legacy
2. **Qualität**: Identische oder bessere Extraktion
3. **Stabilität**: <0.1% Fehlerrate
4. **Memory**: <20% mehr als Legacy
5. **Kompatibilität**: 100% bestehende Tests grün

## 🚨 Risiken und Mitigationen

| Risiko | Wahrscheinlichkeit | Impact | Mitigation |
|--------|-------------------|---------|------------|
| Docling Bugs | Mittel | Hoch | Fallback + Tests |
| Performance Regression | Niedrig | Mittel | Benchmarks |
| Breaking Changes | Niedrig | Hoch | Feature Flag |
| Memory Issues | Mittel | Mittel | Monitoring |

## 📝 Dokumentation

### Zu erstellen:
1. **Migration Guide** für Entwickler
2. **Performance Report** mit Benchmarks
3. **API Dokumentation** für neue Features
4. **Troubleshooting Guide**

## 🎯 Nächste Schritte

1. **Heute**: Dependencies installieren
2. **Morgen**: Erste Implementierung starten
3. **Diese Woche**: Core Features fertig
4. **Nächste Woche**: Testing und Validierung
5. **Übernächste Woche**: Rollout beginnen