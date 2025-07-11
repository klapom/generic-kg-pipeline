# Implementation Status - Generic Knowledge Graph Pipeline

## Stand: 11. Juli 2025

### Übersicht
Diese Dokumentation beschreibt den aktuellen Implementierungsstand der Generic Knowledge Graph Pipeline mit vLLM-Integration für fortgeschrittene PDF-Verarbeitung.

## Implementierte Features

### 1. vLLM Integration
- **SmolDocling-256M-preview Integration**: Vision-Language Model für Dokumentenanalyse
  - Implementiert in: `core/clients/vllm_smoldocling_local.py`
  - DocTags Format Parsing für strukturierte Dokumentenerkennung
  - Multimodales Input-Format für vLLM angepasst

### 2. Hybrid PDF Parser
- **Intelligente Fallback-Strategie**: 
  - Primär: SmolDocling für Layout-Analyse
  - Fallback: pdfplumber für komplexe Layouts
  - Implementiert in: `core/parsers/hybrid_pdf_parser.py`
  
- **Konfigurierbare Modi**:
  - Mode 0: Niemals pdfplumber verwenden
  - Mode 1: pdfplumber nur als Fallback
  - Mode 2: Immer parallel ausführen

### 3. Advanced PDF Extractor
- **Bounding Box Filtering**: Verhindert Duplikate zwischen Text und Tabellen
  - Implementiert in: `core/parsers/advanced_pdf_extractor.py`
  - Layout-Erhaltung mit konfigurierbaren Toleranzen
  - Strukturierte Tabellendaten-Extraktion

### 4. Table-Text Separator
- **Intelligente Inhaltstrennung**:
  - Implementiert in: `core/parsers/table_text_separator.py`
  - Trennt Tabelleninhalte von regulärem Text
  - Erhält Tabellenstruktur für Triple-Generierung

### 5. Complex Layout Detection
- **Automatische Erkennung problematischer Layouts**:
  - Erkennt wenn SmolDocling Seiten als einzelnes Bild interpretiert
  - Triggert automatisch Fallback-Mechanismen
  - Konfigurierbare Schwellwerte

## Getestete Komponenten

### Erfolgreich verarbeitete Dokumente:
1. **BMW 3er G20 Preview** (15 Seiten)
   - 21 Segmente extrahiert (15 Text, 6 Tabellen)
   - 28.061 Zeichen erfolgreich verarbeitet
   
2. **BMW 1er Sedan CN** (15 Seiten)
   - 21 Segmente extrahiert (15 Text, 6 Tabellen)
   - 20.073 Zeichen verarbeitet

3. **BMW 8er G14/G15** (13 Seiten)
   - 18 Segmente extrahiert (13 Text, 5 Tabellen)
   - 29.489 Zeichen verarbeitet

### Verarbeitungs-Pipeline:
```python
# Pipeline-Ablauf
1. SmolDocling Layout-Analyse
2. Complex Layout Detection
3. Bei Bedarf: pdfplumber Fallback
4. Bounding Box Filtering
5. Table-Text Separation
6. Segment-Erstellung
```

## Konfiguration

### Layout-Einstellungen:
```yaml
layout_settings:
  use_layout: true
  table_x_tolerance: 3
  table_y_tolerance: 3
  text_x_tolerance: 5
  text_y_tolerance: 5
```

### vLLM-Einstellungen:
```yaml
vllm:
  model_id: "numinamath/SmolDocling-256M-Preview"
  gpu_memory_utilization: 0.2
  max_pages: 15
```

## Bekannte Einschränkungen

1. **Speichernutzung**: GPU-Memory auf 20% limitiert für Entwicklung
2. **Seitenlimit**: Maximal 15 Seiten pro Dokument
3. **SmolDocling Limitierungen**: 
   - Interpretiert komplexe Layouts manchmal als einzelnes Bild
   - Benötigt Fallback für bestimmte Tabellenstrukturen

## Nächste Schritte

1. Integration in Haupt-Pipeline
2. Erweiterung auf weitere Dokumenttypen
3. Performance-Optimierung für größere Dokumente
4. Integration der Qwen2.5-VL für Bildanalyse

## Technische Details

### Segment-Format:
```python
Segment(
    content: str,
    segment_type: str,  # 'text' oder 'table'
    metadata: {
        'page_number': int,
        'content_type': str,
        'char_count': int,
        # Für Tabellen zusätzlich:
        'table_id': str,
        'row_count': int,
        'col_count': int,
        'bbox': tuple
    }
)
```

### Erfolgsmetriken:
- Keine Duplikate zwischen Text und Tabellen ✓
- Tabellenstruktur erhalten ✓
- Layout-Formatierung beibehalten ✓
- 1:1 Verifizierbarkeit mit Originaldokument ✓

## Installation und Verwendung

```bash
# Environment aktivieren
source .venv/bin/activate

# Alle PDFs verarbeiten
python process_all_pdfs_private.py

# Einzelne PDF testen
python test_complete_segment_content.py
```

---
Stand: 11.07.2025 - Implementierung erfolgreich getestet und bereit für Integration