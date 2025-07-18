# Integration Plan: Fehlende Extraktionsmethoden

## Phase 1: Sofortige Integration (Höchste Priorität)

### 1.1 TableToTripleConverter Integration

**Ziel**: Aktivierung der bereits existierenden Triple-Generierung in der Hauptpipeline

**Änderungen in `HybridPDFParser`**:
```python
# In __init__:
from core.parsers.table_to_triples import TableToTripleConverter

self.triple_converter = TableToTripleConverter()
self.generate_triples = config.get('generate_triples', False)

# Nach Tabellenextraktion:
if self.generate_triples and extracted_tables:
    for table in extracted_tables:
        motor_data = self.triple_converter.extract_motorisierung_table(pdf_path, page_num)
        triples = self.triple_converter.generate_triples(motor_data)
        turtle_output = self.triple_converter.generate_turtle(triples)
        
        # Speichern in Segment-Metadata
        segment.metadata['rdf_triples'] = triples
        segment.metadata['turtle_output'] = turtle_output
```

**Konfiguration**:
```python
config = {
    'generate_triples': True,
    'triple_namespace': 'bmw:',
    'export_turtle': True,
    'output_dir': 'data/output/triples/'
}
```

### 1.2 Camelot Fallback Integration

**Ziel**: Camelot als Fallback für komplexe Tabellen

**Neue Datei: `core/parsers/implementations/pdf/extractors/camelot_extractor.py`**:
```python
import camelot
from typing import List, Dict, Any
import logging

class CamelotExtractor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_tables(self, pdf_path: Path, page_num: int) -> List[Dict[str, Any]]:
        """Extract tables using Camelot with both flavors"""
        tables = []
        
        # Try stream flavor first (for most tables)
        try:
            stream_tables = camelot.read_pdf(
                str(pdf_path), 
                pages=str(page_num),
                flavor='stream'
            )
            tables.extend(self._convert_camelot_tables(stream_tables, 'stream'))
        except Exception as e:
            self.logger.warning(f"Stream flavor failed: {e}")
        
        # Try lattice flavor for complex tables
        try:
            lattice_tables = camelot.read_pdf(
                str(pdf_path), 
                pages=str(page_num),
                flavor='lattice'
            )
            tables.extend(self._convert_camelot_tables(lattice_tables, 'lattice'))
        except Exception as e:
            self.logger.warning(f"Lattice flavor failed: {e}")
        
        return tables
    
    def _convert_camelot_tables(self, camelot_tables, flavor: str) -> List[Dict[str, Any]]:
        """Convert Camelot tables to our format"""
        tables = []
        for i, table in enumerate(camelot_tables):
            df = table.df
            formatted_table = {
                'table_id': f'{flavor}_{i}',
                'extractor': 'camelot',
                'flavor': flavor,
                'accuracy': table.accuracy,
                'headers': df.iloc[0].tolist() if len(df) > 0 else [],
                'data': df.iloc[1:].values.tolist() if len(df) > 1 else [],
                'row_count': len(df) - 1,
                'col_count': len(df.columns)
            }
            tables.append(formatted_table)
        return tables
```

**Integration in HybridPDFParser**:
```python
# In __init__:
from core.parsers.implementations.pdf.extractors.camelot_extractor import CamelotExtractor

self.camelot_extractor = CamelotExtractor()
self.use_camelot = config.get('use_camelot', True)

# In Fallback-Logik:
if self.use_camelot and page_needs_fallback:
    camelot_tables = self.camelot_extractor.extract_tables(pdf_path, page_num)
    if camelot_tables:
        # Wähle beste Tabelle basierend auf Accuracy
        best_table = max(camelot_tables, key=lambda t: t['accuracy'])
        extracted_tables.append(best_table)
```

### 1.3 Regex-Fallback für bekannte Formate

**Neue Datei: `core/parsers/implementations/pdf/extractors/regex_extractor.py`**:
```python
import re
from typing import List, Dict, Any
from pathlib import Path

class RegexExtractor:
    def __init__(self):
        self.patterns = {
            'bmw_motorisierung': {
                'section_start': r'Motorisierungen',
                'section_end': r'(?:Verfügbar|›|Ausstattung)',
                'models': ['320i', '330i', 'M340i', '330e', '318d', '320d', '330d'],
                'pattern': r'({model})\s+([R]\d+\s+[\d,]+l\s+\w+)\s+([\w-]+)\s+([\d/\s]+)\s+([\d\s–-]+)',
                'fields': ['Modell', 'Motor', 'Getriebe', 'Leistung_Drehmoment', 'CO2_Emission']
            }
        }
    
    def extract_structured_data(self, text: str, pattern_name: str) -> List[Dict[str, Any]]:
        """Extract structured data using regex patterns"""
        if pattern_name not in self.patterns:
            return []
        
        pattern_config = self.patterns[pattern_name]
        
        # Extract section
        section_match = re.search(
            f"{pattern_config['section_start']}(.*?){pattern_config['section_end']}", 
            text, 
            re.DOTALL
        )
        
        if not section_match:
            return []
        
        section_text = section_match.group(1)
        data = []
        
        # Extract for each model
        for model in pattern_config['models']:
            model_pattern = pattern_config['pattern'].format(model=model)
            matches = re.findall(model_pattern, section_text)
            
            for match in matches:
                entry = dict(zip(pattern_config['fields'], match))
                data.append(entry)
        
        return data
```

## Phase 2: Erweiterte Integration (Mittlere Priorität)

### 2.1 Multi-Parser-Fallback-Kaskade

**Erweiterung des HybridPDFParser**:
```python
class ExtractorChain:
    def __init__(self):
        self.extractors = [
            ('smoldocling', self.smoldocling_client),
            ('camelot', self.camelot_extractor),
            ('pdfplumber', self.pdfplumber_extractor),
            ('regex', self.regex_extractor)
        ]
        self.confidence_threshold = 0.8
    
    def extract_with_fallback(self, pdf_path: Path, page_num: int) -> Dict[str, Any]:
        """Try extractors in order until success"""
        for extractor_name, extractor in self.extractors:
            try:
                result = extractor.extract_page(pdf_path, page_num)
                confidence = self._calculate_confidence(result)
                
                if confidence > self.confidence_threshold:
                    result['extractor_used'] = extractor_name
                    result['confidence'] = confidence
                    return result
                
            except Exception as e:
                self.logger.warning(f"{extractor_name} failed: {e}")
                continue
        
        return {'error': 'All extractors failed'}
    
    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
        """Calculate extraction confidence"""
        score = 0.0
        
        # Text length score
        text_length = len(result.get('text', ''))
        if text_length > 100:
            score += 0.3
        
        # Table structure score
        tables = result.get('tables', [])
        if tables:
            score += 0.4
        
        # Metadata completeness
        metadata = result.get('metadata', {})
        if metadata:
            score += 0.3
        
        return min(score, 1.0)
```

### 2.2 Office-Parser Integration

**Neue Datei: `core/parsers/unified_document_parser.py`**:
```python
from core.parsers.implementations.office import DOCXParser, XLSXParser, PPTXParser
from core.parsers.implementations.pdf import HybridPDFParser

class UnifiedDocumentParser:
    def __init__(self, config: Dict[str, Any]):
        self.parsers = {
            '.pdf': HybridPDFParser(config),
            '.docx': DOCXParser(config),
            '.xlsx': XLSXParser(config),
            '.pptx': PPTXParser(config)
        }
    
    async def parse(self, file_path: Path) -> Document:
        """Parse any supported document type"""
        suffix = file_path.suffix.lower()
        
        if suffix not in self.parsers:
            raise ParseError(f"Unsupported file type: {suffix}")
        
        parser = self.parsers[suffix]
        return await parser.parse(file_path)
```

### 2.3 Erweiterte Debugging-Ausgaben

**Neue Datei: `core/parsers/detailed_analyzer.py`**:
```python
class DetailedAnalyzer:
    def __init__(self):
        self.results = {}
    
    def analyze_extraction(self, document: Document, extractor_name: str) -> Dict[str, Any]:
        """Detailed analysis of extraction results"""
        analysis = {
            'extractor': extractor_name,
            'timestamp': datetime.now().isoformat(),
            'document_stats': {
                'total_segments': len(document.segments),
                'total_visual_elements': len(document.visual_elements),
                'pages_processed': document.metadata.page_count
            },
            'content_analysis': self._analyze_content(document),
            'quality_metrics': self._calculate_quality_metrics(document)
        }
        
        return analysis
    
    def compare_extractors(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare results from multiple extractors"""
        comparison = {
            'extractor_performance': {},
            'content_overlap': {},
            'quality_ranking': []
        }
        
        # Performance comparison
        for result in results:
            extractor = result['extractor']
            comparison['extractor_performance'][extractor] = {
                'segments': result['document_stats']['total_segments'],
                'visual_elements': result['document_stats']['total_visual_elements'],
                'quality_score': result['quality_metrics']['overall_score']
            }
        
        # Rank by quality
        ranking = sorted(results, key=lambda x: x['quality_metrics']['overall_score'], reverse=True)
        comparison['quality_ranking'] = [r['extractor'] for r in ranking]
        
        return comparison
```

## Phase 3: Intelligente Optimierungen (Niedrige Priorität)

### 3.1 Adaptive Parsing

**Machine Learning-basierte Extractor-Auswahl**:
```python
class AdaptiveParser:
    def __init__(self):
        self.model = self._load_classification_model()
        self.feature_extractor = DocumentFeatureExtractor()
    
    def predict_best_extractor(self, document_path: Path) -> str:
        """Predict best extractor based on document features"""
        features = self.feature_extractor.extract_features(document_path)
        prediction = self.model.predict([features])
        return prediction[0]
    
    def learn_from_results(self, document_path: Path, results: Dict[str, Any]):
        """Learn from extraction results for future predictions"""
        features = self.feature_extractor.extract_features(document_path)
        best_extractor = max(results, key=lambda x: results[x]['quality_score'])
        
        # Update model with new training data
        self.model.partial_fit([features], [best_extractor])
```

### 3.2 Qualitätskontrolle

**Automatische Validierung und Korrektur**:
```python
class QualityController:
    def __init__(self):
        self.validators = [
            TableStructureValidator(),
            ContentCoherenceValidator(),
            MetadataValidator()
        ]
    
    def validate_extraction(self, document: Document) -> Dict[str, Any]:
        """Validate extraction quality"""
        validation_results = {}
        
        for validator in self.validators:
            result = validator.validate(document)
            validation_results[validator.__class__.__name__] = result
        
        return validation_results
    
    def auto_correct(self, document: Document, validation_results: Dict[str, Any]) -> Document:
        """Automatically correct common extraction errors"""
        corrected_document = document
        
        for validator_name, result in validation_results.items():
            if not result['valid']:
                corrector = self._get_corrector(validator_name)
                corrected_document = corrector.correct(corrected_document, result)
        
        return corrected_document
```

## Implementierungsschritte

1. **Woche 1**: TableToTripleConverter Integration + Camelot Fallback
2. **Woche 2**: Regex-Fallback + Multi-Parser-Kaskade
3. **Woche 3**: Office-Parser Integration + Erweiterte Debugging
4. **Woche 4**: Testing, Optimierung, Dokumentation

## Abhängigkeiten

```bash
# Neue Dependencies
pip install camelot-py[cv]
pip install tabula-py
pip install python-docx
pip install openpyxl
pip install python-pptx

# Systemabhängigkeiten für Camelot
apt-get install ghostscript
apt-get install python3-tk
```

## Konfiguration

```python
enhanced_config = {
    'extraction': {
        'generate_triples': True,
        'use_camelot': True,
        'use_regex_fallback': True,
        'multi_parser_fallback': True,
        'confidence_threshold': 0.8,
        'detailed_debugging': True
    },
    'office_formats': {
        'enabled': True,
        'extract_images': True,
        'extract_tables': True,
        'preserve_formatting': True
    },
    'quality_control': {
        'auto_validation': True,
        'auto_correction': True,
        'quality_threshold': 0.7
    }
}
```