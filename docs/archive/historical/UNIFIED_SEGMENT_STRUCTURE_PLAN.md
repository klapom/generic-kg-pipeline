# Plan: Vereinheitlichung der Segment-Strukturen

## 🎯 Ziel
Alle Parser sollen eine einheitliche, konsistente Segment-Struktur erzeugen, die die Weiterverarbeitung in der Pipeline vereinfacht.

## 📊 Aktuelle Situation

### Probleme:
1. **Inkonsistente segment_type Werte** zwischen Parsern
2. **Unterschiedliche Metadata-Felder** je nach Parser
3. **Visuelle Elemente** werden separat behandelt, nicht als Segmente
4. **Keine einheitliche Taxonomie** für Inhaltstypen

### Auswirkungen:
- Downstream-Prozesse müssen jeden Parser-Typ anders behandeln
- Chunking-Strategien sind kompliziert
- Knowledge Graph Erstellung ist inkonsistent
- Kontext zwischen Text und Bildern geht verloren

## 🏗️ Vorgeschlagene Lösung

### 1. Einheitliche Segment-Type Taxonomie

```python
class UnifiedSegmentType(str, Enum):
    # Text-basierte Typen
    TEXT = "text"              # Normaler Fließtext
    HEADING_1 = "heading_1"    # Hauptüberschrift
    HEADING_2 = "heading_2"    # Unterüberschrift
    HEADING_3 = "heading_3"    # Unter-Unterüberschrift
    TITLE = "title"            # Dokumenttitel
    SUBTITLE = "subtitle"      # Untertitel
    
    # Strukturierte Inhalte
    TABLE = "table"            # Tabellen
    LIST = "list"              # Listen (bullet/numbered)
    CODE = "code"              # Code-Blöcke
    QUOTE = "quote"            # Zitate
    
    # Visuelle Inhalte (NEU als Segmente)
    IMAGE = "image"            # Bilder/Fotos
    CHART = "chart"            # Diagramme/Charts
    DIAGRAM = "diagram"        # Technische Zeichnungen
    FORMULA = "formula"        # Mathematische Formeln
    
    # Metadaten-Segmente
    METADATA = "metadata"      # Sheet-Header, Slide-Info etc.
    CAPTION = "caption"        # Bildunterschriften
    FOOTNOTE = "footnote"      # Fußnoten
    
    # Spezielle Typen
    PAGE_BREAK = "page_break"  # Seitenumbruch-Marker
    SECTION_BREAK = "section_break"  # Abschnitts-Marker
```

### 2. Einheitliche Metadata-Struktur

```python
# Basis-Metadata (PFLICHT für alle Segmente)
base_metadata = {
    "source_type": "pdf|docx|xlsx|pptx|txt",  # Dokumenttyp
    "source_location": {                       # Position im Originaldokument
        "page": 1,              # Seite/Slide/Sheet
        "index": 0,             # Sequenzielle Position
        "paragraph": 1,         # Falls zutreffend
        "row": 1,              # Falls Excel
        "column": "A"          # Falls Excel
    },
    "confidence": 0.95,        # Extraktions-Konfidenz (0-1)
    "language": "de",          # Sprache des Segments
}

# Optionale Metadata (je nach Segment-Typ)
optional_metadata = {
    # Für Überschriften
    "heading_level": 1,        # 1-6
    "outline_number": "1.2.3", # Gliederungsnummer
    
    # Für Tabellen
    "table_dimensions": {"rows": 5, "columns": 3},
    "has_header": True,
    
    # Für visuelle Elemente
    "visual_hash": "abc123...",     # Verweis auf VisualElement
    "visual_type": "photo|diagram|chart|screenshot",
    "has_caption": True,
    "caption_text": "Abbildung 1: ...",
    "alt_text": "Beschreibung für Accessibility",
    
    # Für Code
    "code_language": "python",
    
    # Style-Informationen
    "style": "Normal|Heading1|Quote",
    "formatting": ["bold", "italic", "underline"],
    
    # Relationen
    "parent_segment": "uuid",       # Verweis auf übergeordnetes Segment
    "child_segments": ["uuid1"],    # Verweise auf untergeordnete Segmente
}
```

### 3. Visuelle Elemente als Segmente

Visuelle Elemente werden ZUSÄTZLICH als Segmente eingefügt:

```python
# Beispiel: Bild-Segment
{
    "segment_type": "image",
    "content": "[IMAGE: Technisches Diagramm des BMW 3er Motors]",
    "page_number": 5,
    "metadata": {
        "source_type": "pdf",
        "source_location": {"page": 5, "index": 12},
        "confidence": 1.0,
        "visual_hash": "abc123...",
        "visual_type": "diagram",
        "has_caption": True,
        "caption_text": "Abb. 3: Motoraufbau BMW B58",
        "alt_text": "Schnittzeichnung eines 6-Zylinder Motors"
    },
    "visual_references": ["abc123..."]  # Hash des VisualElement
}
```

### 4. Implementierungs-Schritte

#### Phase 1: Vorbereitung
1. **Segment-Type Enum erweitern** in `core/parsers/interfaces/data_models.py`
2. **Metadata-Validierung** erstellen für einheitliche Struktur
3. **Mapping-Funktionen** schreiben für alte → neue Typen

#### Phase 2: Parser-Updates (Reihenfolge nach Komplexität)
1. **TXT Parser** anpassen (einfachste Struktur)
2. **PDF Parser** anpassen 
3. **DOCX Parser** anpassen
4. **XLSX Parser** anpassen
5. **PPTX Parser** anpassen

#### Phase 3: Visual Elements Integration
1. **Visual-to-Segment Converter** implementieren
2. **Segment-Reihenfolge** beibehalten (Text → Bild → Caption)
3. **Cross-Referenzen** zwischen Segmenten und VisualElements

#### Phase 4: Testing & Migration
1. **Unit Tests** für jeden Parser aktualisieren
2. **Migrations-Skript** für bestehende Daten
3. **Vergleichstest** alt vs. neu
4. **Performance-Impact** messen

## 📈 Vorteile

1. **Einheitliche Verarbeitung**: Alle Downstream-Prozesse können Segmente gleich behandeln
2. **Besserer Kontext**: Bilder und Text bleiben in richtiger Reihenfolge
3. **Einfacheres Chunking**: Klare Hierarchie und Typen
4. **Verbesserte KG-Erstellung**: Konsistente Entitäten und Relationen
5. **Zukunftssicher**: Neue Segment-Typen können einfach hinzugefügt werden

## ⚠️ Risiken & Herausforderungen

1. **Breaking Change**: Bestehende Pipelines müssen angepasst werden
2. **Performance**: Mehr Segmente durch Visual Elements
3. **Komplexität**: Mapping zwischen alten und neuen Typen
4. **Speicherbedarf**: Redundanz zwischen Segments und VisualElements

## 🔄 Alternativen

### Alternative 1: Minimale Änderung
- Nur segment_type vereinheitlichen
- Metadata bleibt parser-spezifisch
- Visual Elements bleiben separat

### Alternative 2: Wrapper-Lösung
- Post-Processing Layer, der Segmente vereinheitlicht
- Parser bleiben unverändert
- Transformation erst bei Bedarf

### Alternative 3: Konfigurierbares System
- Parser können zwischen "legacy" und "unified" Mode wählen
- Schrittweise Migration möglich
- Rückwärtskompatibilität

## 📝 Offene Fragen

1. **Sollen Visual Elements komplett in Segmente integriert werden oder dual existieren?**
2. **Wie detailliert soll die Segment-Type Taxonomie sein?**
3. **Sollen wir Rückwärtskompatibilität gewährleisten?**
4. **Wie gehen wir mit parser-spezifischen Features um?**
5. **Brauchen wir eine Versions-Kennzeichnung für das Segment-Format?**

## 🚀 Empfohlenes Vorgehen

1. **Diskussion & Entscheidung** über diesen Plan
2. **Proof of Concept** mit TXT Parser
3. **Schrittweise Migration** der anderen Parser
4. **Parallelbetrieb** alt/neu für Übergangsphase
5. **Vollständige Migration** nach Validierung

---

**Geschätzter Aufwand**: 
- Design & Spezifikation: 2-3 Tage
- Implementierung: 5-7 Tage
- Testing & Migration: 3-4 Tage
- **Gesamt: ~2 Wochen**