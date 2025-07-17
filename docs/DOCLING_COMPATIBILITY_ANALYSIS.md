# Docling Integration - Kompatibilitätsanalyse und Implementierungsplan

## 1. Aktueller Datenpfad

### 1.1 PDF-Verarbeitung
```
PDF → SmolDocling (vLLM) → parse_model_output() → HybridPDFParser → Document
                              ↓
                         Extrahiert:
                         - text (mit bbox)
                         - images (mit bbox [x1,y1,x2,y2] in 0-500 Skala)
                         - tables
                         - formulas
                              ↓
                         HybridPDFParser:
                         - Erstellt VisualElement mit bbox
                         - _extract_image_bytes() nutzt bbox:
                           - Skaliert von 0-500 zu Seitenkoordinaten
                           - Extrahiert Bildausschnitt aus PDF
                           - Speichert in visual.raw_data
                              ↓
                         VLM-Analyse:
                         - Nutzt visual.raw_data (PNG bytes)
                         - Fügt vlm_description hinzu
```

### 1.2 Office-Dokumente (DOCX, XLSX, PPTX)
```
Office → python-docx/openpyxl/python-pptx → Parser → Document
              ↓
         Extrahiert direkt:
         - Text/Tabellen als Segments
         - Bilder als raw bytes → VisualElement.raw_data
         - Charts als Text-Repräsentation
              ↓
         VLM-Analyse:
         - Nutzt visual.raw_data direkt
         - Keine bbox-basierte Extraktion nötig
```

## 2. Kritische Datenstrukturen

### 2.1 SmolDocling Output (aktuell)
```python
# Von parse_model_output():
{
    "text": "...",
    "images": [
        {
            "content": "<loc_0><loc_0><loc_500><loc_375><other>",
            "caption": "",
            "bbox": [0, 0, 500, 375]  # KRITISCH für PDF-Bildextraktion!
        }
    ],
    "tables": [...],
    "formulas": [...]
}
```

### 2.2 VisualElement
```python
@dataclass
class VisualElement:
    element_type: VisualElementType
    source_format: DocumentType
    content_hash: str
    bounding_box: Optional[List[float]] = None  # Für PDFs: [x1,y1,x2,y2]
    raw_data: Optional[bytes] = None  # PNG/JPEG bytes für VLM
    # ... weitere Felder
```

## 3. Docling Integration - Kompatibilitätsmapping

### 3.1 DoclingDocument zu unserem Format
```python
def convert_docling_to_legacy_format(doc: DoclingDocument, page_image: PIL.Image = None) -> dict:
    """
    Konvertiert DoclingDocument zu unserem aktuellen Format
    WICHTIG: Behält bbox-Informationen bei!
    """
    result = {
        "text": "",
        "images": [],
        "tables": [],
        "formulas": [],
        "text_blocks": []
    }
    
    # Iteriere über DoclingDocument Elemente
    for element in doc.elements:
        if element.type == "text":
            # Text mit Location tags
            if hasattr(element, 'bbox'):
                result["text_blocks"].append({
                    "text": element.content,
                    "bbox": [element.bbox.x0, element.bbox.y0, 
                            element.bbox.x1, element.bbox.y1]
                })
            result["text"] += element.content + "\n"
            
        elif element.type == "picture":
            # KRITISCH: bbox beibehalten für Bildextraktion!
            image_data = {
                "content": str(element),  # Raw DocTags
                "caption": element.caption if hasattr(element, 'caption') else "",
            }
            
            # Extrahiere bbox aus DocTags oder element.bbox
            if hasattr(element, 'bbox'):
                image_data["bbox"] = [
                    element.bbox.x0, 
                    element.bbox.y0,
                    element.bbox.x1, 
                    element.bbox.y1
                ]
            else:
                # Fallback: Parse aus DocTags
                import re
                loc_match = re.search(r'<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>', str(element))
                if loc_match:
                    image_data["bbox"] = [int(x) for x in loc_match.groups()]
            
            result["images"].append(image_data)
            
        elif element.type == "table":
            # Tabellen konvertieren
            result["tables"].append({
                "content": element.to_text(),  # oder element.export_to_markdown()
                "format": "text"
            })
    
    return result
```

### 3.2 Angepasster parse_model_output
```python
def parse_model_output(self, output: Any, page_image: PIL.Image = None) -> Dict[str, Any]:
    """Parse SmolDocling output mit docling_core"""
    
    # Option A: Direkte docling Nutzung
    if self.use_docling and page_image is not None:
        from docling_core.types.doc.document import DocTagsDocument
        from docling_core.types.doc import DoclingDocument
        
        # Extrahiere DocTags text
        doctags = self._extract_doctags_text(output)
        
        # Parse mit docling
        doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(
            [doctags], 
            [page_image]
        )
        
        # Konvertiere zu DoclingDocument
        doc = DoclingDocument(name="Page")
        doc.load_from_doctags(doctags_doc)
        
        # WICHTIG: Konvertiere zurück zu unserem Format mit bbox!
        return convert_docling_to_legacy_format(doc, page_image)
    
    # Option B: Fallback auf aktuelle Implementierung
    else:
        return self._parse_model_output_legacy(output)
```

### 3.3 HybridPDFParser Anpassungen
```python
# In HybridPDFParser._process_smoldocling_page():

# Stelle sicher, dass page_image an parse_model_output übergeben wird
parsed_data = self.smoldocling_client.parse_model_output(
    raw_output,
    page_image=page_images[page_num-1]  # NEU: Bild für docling
)

# Rest bleibt gleich - bbox wird weiterhin für _extract_image_bytes verwendet
```

## 4. Schrittweise Migration

### Phase 1: Parallelbetrieb (Feature Flag)
```python
class VLLMSmolDoclingClient:
    def __init__(self, use_docling: bool = False):
        self.use_docling = use_docling
        
    def parse_model_output(self, output, page_image=None):
        if self.use_docling and page_image:
            return self._parse_with_docling(output, page_image)
        else:
            return self._parse_legacy(output)
```

### Phase 2: Validierung
- Vergleiche Ausgaben beider Parser
- Stelle sicher, dass bbox-Werte identisch sind
- Teste Bildextraktion mit beiden Methoden

### Phase 3: Erweiterte Features
```python
class SmolDoclingPage:
    # Neue Felder
    _docling_doc: Optional[DoclingDocument] = None
    
    def to_markdown(self) -> str:
        """Nutze docling Export"""
        if self._docling_doc:
            return self._docling_doc.export_to_markdown()
        return self.text
    
    def to_html(self) -> str:
        """Nutze docling Export"""
        if self._docling_doc:
            return self._docling_doc.export_to_html()
        # Fallback
        return f"<div>{self.text}</div>"
```

## 5. Kritische Punkte für Kompatibilität

### ✅ MUSS erhalten bleiben:
1. **bbox-Koordinaten** in images[] - KRITISCH für PDF-Bildextraktion
2. **0-500 Skalierung** - Wird in _extract_image_bytes verwendet
3. **Datenstruktur** der parse_model_output Rückgabe
4. **VisualElement.raw_data** - Wird von VLM verwendet

### ⚠️ Potenzielle Probleme:
1. **Performance**: Docling könnte langsamer sein
2. **Speicher**: DoclingDocument hält mehr Metadaten
3. **Dependencies**: Zusätzliche Libraries erforderlich

### 🎯 Lösungsansätze:
1. **Lazy Loading**: DoclingDocument nur bei Bedarf erstellen
2. **Caching**: Konvertierte Formate zwischenspeichern
3. **Streaming**: Bei großen Dokumenten seitenweise verarbeiten

## 6. Test-Strategie

```python
def test_bbox_compatibility():
    """Test dass bbox-Werte erhalten bleiben"""
    
    # Parse mit alter Methode
    legacy_result = parser._parse_legacy(doctags)
    
    # Parse mit docling
    docling_result = parser._parse_with_docling(doctags, image)
    
    # Vergleiche bbox-Werte
    for i, (legacy_img, docling_img) in enumerate(
        zip(legacy_result["images"], docling_result["images"])
    ):
        assert legacy_img["bbox"] == docling_img["bbox"], \
            f"Image {i}: bbox mismatch"
    
    # Teste Bildextraktion
    visual = VisualElement(
        bounding_box=docling_img["bbox"],
        page_or_slide=1
    )
    
    # Sollte funktionieren
    _extract_image_bytes(pdf_path, [visual])
    assert visual.raw_data is not None
```

## 7. Empfehlung

1. **Schrittweise Integration** mit Feature Flag
2. **Behalte aktuelle Datenstrukturen** für Kompatibilität
3. **Wrapper-Funktionen** für Konvertierung
4. **Umfangreiche Tests** besonders für bbox-Erhaltung
5. **Performance-Monitoring** während Migration

Die Integration ist machbar ohne Breaking Changes, erfordert aber sorgfältige Implementierung der Konvertierungsfunktionen.