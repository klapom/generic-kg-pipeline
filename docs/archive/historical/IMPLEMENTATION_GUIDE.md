# Implementierungsleitfaden: Generisches Knowledge Graph Pipeline System

## Übersicht

Dieser Leitfaden beschreibt die schrittweise Implementierung des neuen, schlanken Knowledge Graph Pipeline Systems basierend auf der Architektur-Dokumentation. Das System wird in 3 Phasen entwickelt: MVP, RAG-Integration und Advanced Features.

## 🎯 Aktueller Implementierungsstand

### ✅ VOLLSTÄNDIG IMPLEMENTIERT (MVP-Phase)
- **Multi-Modal Document Parser System (100%)**
  - PDF Parser mit vLLM SmolDocling Integration
  - DOCX Parser mit Image Extraction
  - XLSX Parser mit Chart Analysis
  - PPTX Parser mit Slide Visuals
  - Parser Factory für automatische Format-Erkennung
  - Context Mapping für präzise Text-Bild-Zuordnung

- **Content Chunking mit Context Inheritance (100%)**
  - Structure-Aware Chunking basierend auf Dokumentstruktur
  - Context Group Formation (PDF Sections, DOCX Headings, XLSX Sheets, PPTX Topics)
  - LLM-basierte Context Summary Generation
  - Dual-Task Prompting für optimale Kontext-Vererbung
  - Async Processing für Performance

- **LLM Client Infrastructure (100%)**
  - vLLM SmolDocling Client für PDF-Parsing
  - Hochschul-LLM Client für Triple Extraction
  - Qwen2.5-VL Client für Visual Analysis
  - OpenAI-kompatible API Integration

- **FastAPI Application (80%)**
  - Health, Documents, Pipeline, Query Endpoints
  - Multi-Modal Upload Support
  - Batch Processing Integration

- **Batch Processing System (100%)**
  - Concurrent Document Processing
  - Progress Tracking und Error Handling
  - Filesystem-basierte Verarbeitung

### 🔄 IN ENTWICKLUNG
- Triple Store Integration (Fuseki)
- Vector Store Integration (ChromaDB)
- End-to-End Pipeline Integration

## Voraussetzungen

### Entwicklungsumgebung

```bash
# Python 3.11+
python --version  # >= 3.11

# UV Package Manager (empfohlen)
pip install uv

# Virtuelle Umgebung
python -m venv .venv
source .venv/bin/activate

# Abhängigkeiten
uv pip install -r requirements.txt
```

### Systemanforderungen

```bash
# Fuseki Triple Store
docker run -d -p 3030:3030 \
  -v fuseki_data:/fuseki/databases \
  stain/jena-fuseki:latest

# ChromaDB (optional für RAG)
docker run -d -p 8001:8000 \
  -v chromadb_data:/chroma/data \
  chromadb/chroma:latest
```

## Phase 1: MVP Implementation (1-2 Wochen)

### 1.1 Projekt-Setup

```bash
# Projekt-Struktur erstellen
mkdir generic-kg-pipeline
cd generic-kg-pipeline

# Grundstruktur
mkdir -p {core,plugins/{parsers,ontologies,templates},config,data,tests}

# Git-Repository
git init
git add .
git commit -m "Initial project structure"
```

### 1.2 Core-Module implementieren

#### 1.2.1 Konfigurationssystem

```python
# core/config.py
from pydantic import BaseModel
from typing import List, Dict, Optional
import yaml

class ChunkingConfig(BaseModel):
    max_tokens: int = 2000
    overlap_ratio: float = 0.2
    preserve_context: bool = True

class LLMConfig(BaseModel):
    provider: str = "ollama"
    model: str = "qwen:7b"
    temperature: float = 0.1
    max_tokens: int = 4000

class StorageConfig(BaseModel):
    triple_store_url: str = "http://localhost:3030"
    vector_store_path: str = "data/vectors"

class DomainConfig(BaseModel):
    name: str = "general"
    ontology_path: str = "plugins/ontologies/general.ttl"
    enabled_formats: List[str] = ["pdf", "docx", "txt"]

class Config(BaseModel):
    domain: DomainConfig = DomainConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    llm: LLMConfig = LLMConfig()
    storage: StorageConfig = StorageConfig()
    
    @classmethod
    def from_file(cls, path: str) -> 'Config':
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
```

#### 1.2.2 Dokumenten-Parser

```python
# core/document_parser.py
from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass
import os

@dataclass
class Document:
    content: str
    format: str
    metadata: Dict[str, Any]
    source: Optional[str] = None

@dataclass
class Segment:
    content: str
    metadata: Dict[str, Any]
    position: int
    context: Optional[str] = None

class BaseParser(ABC):
    @abstractmethod
    def parse(self, document: Document) -> List[Segment]:
        pass

class DocumentParser:
    def __init__(self, config: Config):
        self.config = config
        self.parsers = self._load_parsers()
    
    def _load_parsers(self) -> Dict[str, BaseParser]:
        """Lädt Format-spezifische Parser"""
        parsers = {}
        for format_name in self.config.domain.enabled_formats:
            parser_class = self._get_parser_class(format_name)
            parsers[format_name] = parser_class(self.config)
        return parsers
    
    def _get_parser_class(self, format_name: str) -> BaseParser:
        """Dynamisches Laden der Parser-Klassen"""
        if format_name == "pdf":
            from plugins.parsers.pdf_parser import PDFParser
            return PDFParser
        elif format_name == "docx":
            from plugins.parsers.office_parser import DOCXParser
            return DOCXParser
        elif format_name == "txt":
            from plugins.parsers.text_parser import TextParser
            return TextParser
        else:
            raise ValueError(f"Unsupported format: {format_name}")
    
    def parse(self, document: Document) -> List[Segment]:
        """Parst Dokument und gibt Segmente zurück"""
        parser = self.parsers.get(document.format)
        if not parser:
            raise ValueError(f"No parser for format: {document.format}")
        
        return parser.parse(document)
```

#### 1.2.3 Content Chunker

```python
# core/content_chunker.py
from typing import List
import tiktoken

@dataclass
class Chunk:
    content: str
    metadata: Dict[str, Any]
    tokens: int
    overlap_with_previous: bool = False

class AdaptiveChunker:
    def __init__(self, config: ChunkingConfig):
        self.max_tokens = config.max_tokens
        self.overlap_ratio = config.overlap_ratio
        self.preserve_context = config.preserve_context
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def chunk(self, segments: List[Segment]) -> List[Chunk]:
        """Intelligente Chunking-Strategie"""
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for segment in segments:
            segment_tokens = len(self.tokenizer.encode(segment.content))
            
            # Segment passt in aktuellen Chunk
            if current_tokens + segment_tokens <= self.max_tokens:
                current_chunk += "\n" + segment.content
                current_tokens += segment_tokens
            else:
                # Aktuellen Chunk abschließen
                if current_chunk:
                    chunks.append(self._create_chunk(current_chunk, current_tokens))
                
                # Neuen Chunk beginnen
                if segment_tokens > self.max_tokens:
                    # Zu großes Segment splitten
                    sub_chunks = self._split_large_segment(segment)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                    current_tokens = 0
                else:
                    current_chunk = segment.content
                    current_tokens = segment_tokens
        
        # Letzten Chunk hinzufügen
        if current_chunk:
            chunks.append(self._create_chunk(current_chunk, current_tokens))
        
        return chunks
    
    def _create_chunk(self, content: str, tokens: int) -> Chunk:
        """Erstellt Chunk-Objekt"""
        return Chunk(
            content=content,
            tokens=tokens,
            metadata={"chunk_size": tokens}
        )
    
    def _split_large_segment(self, segment: Segment) -> List[Chunk]:
        """Splittet zu große Segmente"""
        words = segment.content.split()
        chunks = []
        current_words = []
        
        for word in words:
            current_words.append(word)
            current_text = " ".join(current_words)
            tokens = len(self.tokenizer.encode(current_text))
            
            if tokens >= self.max_tokens:
                # Chunk abschließen
                if len(current_words) > 1:
                    chunk_text = " ".join(current_words[:-1])
                    chunks.append(self._create_chunk(chunk_text, tokens-1))
                    current_words = [word]
                else:
                    # Einzelnes Wort zu groß - trotzdem hinzufügen
                    chunks.append(self._create_chunk(current_text, tokens))
                    current_words = []
        
        # Letzte Wörter
        if current_words:
            chunk_text = " ".join(current_words)
            tokens = len(self.tokenizer.encode(chunk_text))
            chunks.append(self._create_chunk(chunk_text, tokens))
        
        return chunks
```

#### 1.2.4 LLM Client

```python
# core/llm_client.py
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import json
import requests
from dataclasses import dataclass

@dataclass
class Triple:
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source: Optional[str] = None

class LLMProvider(ABC):
    @abstractmethod
    def generate_triples(self, content: str, ontology_context: str) -> List[Triple]:
        pass

class OllamaProvider(LLMProvider):
    def __init__(self, config: LLMConfig):
        self.base_url = "http://localhost:11434"
        self.model = config.model
        self.temperature = config.temperature
    
    def generate_triples(self, content: str, ontology_context: str) -> List[Triple]:
        """Generiert Triples via Ollama"""
        prompt = self._build_prompt(content, ontology_context)
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "temperature": self.temperature,
                "stream": False
            }
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"LLM request failed: {response.text}")
        
        result = response.json()
        return self._parse_triples(result["response"])
    
    def _build_prompt(self, content: str, ontology_context: str) -> str:
        """Erstellt Prompt für Triple-Extraktion"""
        return f"""Extract structured knowledge triples from the following text.
        
Context/Ontology: {ontology_context}

Text: {content}

Please extract triples in the following JSON format:
{{
    "triples": [
        {{
            "subject": "entity1",
            "predicate": "relationship",
            "object": "entity2",
            "confidence": 0.95
        }}
    ]
}}

Focus on factual relationships and use the provided ontology context.
Response:"""
    
    def _parse_triples(self, response: str) -> List[Triple]:
        """Parst LLM-Response zu Triple-Objekten"""
        try:
            data = json.loads(response)
            triples = []
            
            for triple_data in data.get("triples", []):
                triple = Triple(
                    subject=triple_data["subject"],
                    predicate=triple_data["predicate"],
                    object=triple_data["object"],
                    confidence=triple_data.get("confidence", 1.0)
                )
                triples.append(triple)
            
            return triples
        except json.JSONDecodeError:
            # Fallback für nicht-JSON Responses
            return self._parse_text_triples(response)
    
    def _parse_text_triples(self, text: str) -> List[Triple]:
        """Fallback-Parser für Text-basierte Responses"""
        # Einfache Heuristik - kann erweitert werden
        triples = []
        lines = text.strip().split('\n')
        
        for line in lines:
            if ' -> ' in line:
                parts = line.split(' -> ')
                if len(parts) == 3:
                    triple = Triple(
                        subject=parts[0].strip(),
                        predicate=parts[1].strip(),
                        object=parts[2].strip(),
                        confidence=0.8
                    )
                    triples.append(triple)
        
        return triples

class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.provider = self._get_provider()
    
    def _get_provider(self) -> LLMProvider:
        """Lädt Provider basierend auf Konfiguration"""
        if self.config.provider == "ollama":
            return OllamaProvider(self.config)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    def extract_triples(self, content: str, ontology_context: str = "") -> List[Triple]:
        """Extrahiert Triples aus Content"""
        return self.provider.generate_triples(content, ontology_context)
```

#### 1.2.5 Knowledge Graph Store

```python
# core/kg_store.py
from typing import List, Dict, Optional
import requests
from SPARQLWrapper import SPARQLWrapper, JSON

class KGStore:
    def __init__(self, config: StorageConfig):
        self.fuseki_url = config.triple_store_url
        self.dataset_name = "kg_dataset"
        self.sparql = SPARQLWrapper(f"{self.fuseki_url}/{self.dataset_name}/sparql")
        self.update_endpoint = f"{self.fuseki_url}/{self.dataset_name}/update"
    
    def store_triples(self, triples: List[Triple]) -> bool:
        """Speichert Triples in Fuseki"""
        try:
            sparql_update = self._build_insert_query(triples)
            
            response = requests.post(
                self.update_endpoint,
                data=sparql_update,
                headers={"Content-Type": "application/sparql-update"}
            )
            
            return response.status_code == 204
        except Exception as e:
            print(f"Error storing triples: {e}")
            return False
    
    def _build_insert_query(self, triples: List[Triple]) -> str:
        """Erstellt SPARQL INSERT Query"""
        prefixes = """
        PREFIX kg: <http://example.org/kg/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        """
        
        insert_data = "INSERT DATA {\n"
        
        for triple in triples:
            subject = self._format_uri(triple.subject)
            predicate = self._format_uri(triple.predicate)
            obj = self._format_value(triple.object)
            
            insert_data += f"  {subject} {predicate} {obj} .\n"
        
        insert_data += "}"
        
        return prefixes + insert_data
    
    def _format_uri(self, value: str) -> str:
        """Formatiert Wert als URI"""
        if value.startswith("http"):
            return f"<{value}>"
        else:
            # Erstelle URI mit Namespace
            safe_value = value.replace(" ", "_").replace("-", "_")
            return f"kg:{safe_value}"
    
    def _format_value(self, value: str) -> str:
        """Formatiert Objekt-Wert"""
        if value.startswith("http"):
            return f"<{value}>"
        else:
            # Literal-Wert
            return f'"{value}"'
    
    def query(self, sparql_query: str) -> List[Dict]:
        """Führt SPARQL Query aus"""
        try:
            self.sparql.setQuery(sparql_query)
            self.sparql.setReturnFormat(JSON)
            
            results = self.sparql.query().convert()
            return results["results"]["bindings"]
        except Exception as e:
            print(f"Query error: {e}")
            return []
```

#### 1.2.6 Hauptpipeline

```python
# core/pipeline.py
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class ProcessingResult:
    triples: List[Triple]
    metadata: Dict[str, Any]
    success: bool
    error: Optional[str] = None

class Pipeline:
    def __init__(self, config: Config):
        self.config = config
        self.document_parser = DocumentParser(config)
        self.chunker = AdaptiveChunker(config.chunking)
        self.llm_client = LLMClient(config.llm)
        self.kg_store = KGStore(config.storage)
        self.ontology_context = self._load_ontology_context()
    
    def _load_ontology_context(self) -> str:
        """Lädt Ontologie-Kontext für LLM"""
        try:
            with open(self.config.domain.ontology_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Ontology file not found: {self.config.domain.ontology_path}")
            return ""
    
    def process_document(self, document: Document) -> ProcessingResult:
        """Verarbeitet einzelnes Dokument"""
        try:
            # 1. Dokument parsen
            segments = self.document_parser.parse(document)
            
            # 2. Segmente chunken
            chunks = self.chunker.chunk(segments)
            
            # 3. Triples aus Chunks extrahieren
            all_triples = []
            for chunk in chunks:
                triples = self.llm_client.extract_triples(
                    chunk.content, 
                    self.ontology_context
                )
                all_triples.extend(triples)
            
            # 4. Triples speichern
            success = self.kg_store.store_triples(all_triples)
            
            return ProcessingResult(
                triples=all_triples,
                metadata={
                    "segments_count": len(segments),
                    "chunks_count": len(chunks),
                    "triples_count": len(all_triples)
                },
                success=success
            )
        
        except Exception as e:
            return ProcessingResult(
                triples=[],
                metadata={},
                success=False,
                error=str(e)
            )
    
    def process_batch(self, documents: List[Document]) -> List[ProcessingResult]:
        """Verarbeitet Batch von Dokumenten"""
        results = []
        
        for document in documents:
            result = self.process_document(document)
            results.append(result)
            
            # Logging
            if result.success:
                print(f"✅ Processed {document.source}: {result.metadata['triples_count']} triples")
            else:
                print(f"❌ Failed {document.source}: {result.error}")
        
        return results
```

### 1.3 Parser-Plugins implementieren

#### 1.3.1 Text Parser

```python
# plugins/parsers/text_parser.py
from core.document_parser import BaseParser, Document, Segment
from typing import List

class TextParser(BaseParser):
    def __init__(self, config):
        self.config = config
    
    def parse(self, document: Document) -> List[Segment]:
        """Einfacher Text-Parser"""
        paragraphs = document.content.split('\n\n')
        segments = []
        
        for i, paragraph in enumerate(paragraphs):
            if paragraph.strip():
                segment = Segment(
                    content=paragraph.strip(),
                    metadata={
                        "type": "paragraph",
                        "position": i,
                        "format": "text"
                    },
                    position=i
                )
                segments.append(segment)
        
        return segments
```

#### 1.3.2 PDF Parser (Vereinfacht)

```python
# plugins/parsers/pdf_parser.py
from core.document_parser import BaseParser, Document, Segment
from typing import List
import PyPDF2

class PDFParser(BaseParser):
    def __init__(self, config):
        self.config = config
    
    def parse(self, document: Document) -> List[Segment]:
        """Einfacher PDF-Parser (ohne SmolDocling)"""
        # Für MVP: Einfache Textextraktion
        # In späteren Phasen: SmolDocling Integration
        
        segments = []
        
        # Simuliere Seiten-basierte Segmentierung
        pages = document.content.split('\n\n\n')  # Vereinfachte Seiten-Trennung
        
        for page_num, page_content in enumerate(pages):
            if page_content.strip():
                segment = Segment(
                    content=page_content.strip(),
                    metadata={
                        "type": "page",
                        "page_number": page_num + 1,
                        "format": "pdf"
                    },
                    position=page_num
                )
                segments.append(segment)
        
        return segments
```

### 1.4 Konfiguration

```yaml
# config/default.yaml
domain:
  name: "general"
  ontology_path: "plugins/ontologies/general.ttl"
  enabled_formats: ["pdf", "docx", "txt"]

chunking:
  max_tokens: 2000
  overlap_ratio: 0.2
  preserve_context: true

llm:
  provider: "ollama"
  model: "qwen:7b"
  temperature: 0.1
  max_tokens: 4000

storage:
  triple_store_url: "http://localhost:3030"
  vector_store_path: "data/vectors"
```

### 1.5 CLI-Interface

```python
# main.py
import argparse
from pathlib import Path
from core.config import Config
from core.pipeline import Pipeline
from core.document_parser import Document

def main():
    parser = argparse.ArgumentParser(description="Generic Knowledge Graph Pipeline")
    parser.add_argument("--config", default="config/default.yaml", help="Config file path")
    parser.add_argument("--input", required=True, help="Input document or directory")
    parser.add_argument("--format", help="Document format (auto-detect if not specified)")
    parser.add_argument("--domain", help="Domain configuration")
    
    args = parser.parse_args()
    
    # Konfiguration laden
    config = Config.from_file(args.config)
    if args.domain:
        config.domain.name = args.domain
    
    # Pipeline initialisieren
    pipeline = Pipeline(config)
    
    # Input verarbeiten
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Einzelnes Dokument
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        document = Document(
            content=content,
            format=args.format or input_path.suffix[1:],
            metadata={"source": str(input_path)},
            source=str(input_path)
        )
        
        result = pipeline.process_document(document)
        
        if result.success:
            print(f"✅ Successfully processed {input_path}")
            print(f"📊 Extracted {len(result.triples)} triples")
        else:
            print(f"❌ Failed to process {input_path}: {result.error}")
    
    elif input_path.is_dir():
        # Batch-Verarbeitung
        documents = []
        for file_path in input_path.iterdir():
            if file_path.is_file() and file_path.suffix[1:] in config.domain.enabled_formats:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                document = Document(
                    content=content,
                    format=file_path.suffix[1:],
                    metadata={"source": str(file_path)},
                    source=str(file_path)
                )
                documents.append(document)
        
        results = pipeline.process_batch(documents)
        
        # Statistiken
        successful = sum(1 for r in results if r.success)
        total_triples = sum(len(r.triples) for r in results)
        
        print(f"📊 Batch processing completed:")
        print(f"   Successfully processed: {successful}/{len(results)} documents")
        print(f"   Total triples extracted: {total_triples}")

if __name__ == "__main__":
    main()
```

### 1.6 Requirements

```txt
# requirements.txt
pydantic==2.5.0
pyyaml==6.0.1
requests==2.31.0
SPARQLWrapper==2.0.0
tiktoken==0.5.2
PyPDF2==3.0.1
python-docx==0.8.11
openpyxl==3.1.2
fastapi==0.104.1
uvicorn==0.24.0
```

## Phase 2: RAG-Integration (1 Woche)

### 2.1 Vector Store Integration

```python
# core/vector_store.py
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional

class VectorStore:
    def __init__(self, config: StorageConfig):
        self.client = chromadb.PersistentClient(path=config.vector_store_path)
        self.collection = self.client.get_or_create_collection(
            name="document_segments",
            metadata={"hnsw:space": "cosine"}
        )
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def add_chunks(self, chunks: List[Chunk]) -> bool:
        """Fügt Chunks mit Embeddings hinzu"""
        try:
            contents = [chunk.content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            
            # Embeddings erstellen
            embeddings = self.embedding_model.encode(contents).tolist()
            
            # IDs generieren
            ids = [f"chunk_{i}" for i in range(len(chunks))]
            
            # Zu ChromaDB hinzufügen
            self.collection.add(
                documents=contents,
                metadatas=metadatas,
                embeddings=embeddings,
                ids=ids
            )
            
            return True
        except Exception as e:
            print(f"Error adding chunks: {e}")
            return False
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Semantische Ähnlichkeitssuche"""
        try:
            # Query-Embedding
            query_embedding = self.embedding_model.encode([query]).tolist()
            
            # Suche in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Ergebnisse formatieren
            formatted_results = []
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i]
                })
            
            return formatted_results
        except Exception as e:
            print(f"Similarity search error: {e}")
            return []
```

### 2.2 RAG Processor

```python
# core/rag_processor.py
from typing import List, Dict, Optional
from core.vector_store import VectorStore
from core.kg_store import KGStore
from core.llm_client import LLMClient

class RAGProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.vector_store = VectorStore(config.storage)
        self.kg_store = KGStore(config.storage)
        self.llm_client = LLMClient(config.llm)
        self.similarity_threshold = 0.7
    
    def process_with_rag(self, chunks: List[Chunk], ontology_context: str) -> List[Triple]:
        """Verarbeitet Chunks mit RAG-Enhancement"""
        all_triples = []
        
        for chunk in chunks:
            # 1. Ähnliche Chunks finden
            similar_chunks = self.vector_store.similarity_search(
                chunk.content, 
                k=3
            )
            
            # 2. Kontext anreichern
            enriched_content = self._build_enriched_content(chunk, similar_chunks)
            
            # 3. Triples extrahieren
            triples = self.llm_client.extract_triples(
                enriched_content, 
                ontology_context
            )
            
            # 4. Triples validieren
            validated_triples = self._validate_triples(triples)
            
            all_triples.extend(validated_triples)
        
        return all_triples
    
    def _build_enriched_content(self, chunk: Chunk, similar_chunks: List[Dict]) -> str:
        """Erstellt angereicherten Inhalt mit Kontext"""
        enriched = f"Current Content:\n{chunk.content}\n\n"
        
        if similar_chunks:
            enriched += "Related Context:\n"
            for i, similar in enumerate(similar_chunks[:2]):  # Top 2 ähnliche
                if similar['distance'] < self.similarity_threshold:
                    enriched += f"Context {i+1}: {similar['content'][:200]}...\n"
        
        return enriched
    
    def _validate_triples(self, triples: List[Triple]) -> List[Triple]:
        """Validiert Triples gegen bestehenden Knowledge Graph"""
        validated = []
        
        for triple in triples:
            # Prüfe auf Duplikate
            if not self._is_duplicate(triple):
                validated.append(triple)
        
        return validated
    
    def _is_duplicate(self, triple: Triple) -> bool:
        """Prüft ob Triple bereits existiert"""
        query = f"""
        SELECT ?s ?p ?o WHERE {{
            ?s ?p ?o .
            FILTER(CONTAINS(STR(?s), "{triple.subject}"))
            FILTER(CONTAINS(STR(?p), "{triple.predicate}"))
            FILTER(CONTAINS(STR(?o), "{triple.object}"))
        }}
        """
        
        results = self.kg_store.query(query)
        return len(results) > 0
```

### 2.3 Erweiterte Pipeline

```python
# core/pipeline.py (erweitert)
class EnhancedPipeline(Pipeline):
    def __init__(self, config: Config):
        super().__init__(config)
        self.rag_processor = RAGProcessor(config)
    
    def process_document_with_rag(self, document: Document) -> ProcessingResult:
        """Verarbeitet Dokument mit RAG-Enhancement"""
        try:
            # 1. Standard-Verarbeitung
            segments = self.document_parser.parse(document)
            chunks = self.chunker.chunk(segments)
            
            # 2. Chunks zu Vector Store hinzufügen
            self.rag_processor.vector_store.add_chunks(chunks)
            
            # 3. RAG-basierte Triple-Extraktion
            triples = self.rag_processor.process_with_rag(chunks, self.ontology_context)
            
            # 4. Triples speichern
            success = self.kg_store.store_triples(triples)
            
            return ProcessingResult(
                triples=triples,
                metadata={
                    "segments_count": len(segments),
                    "chunks_count": len(chunks),
                    "triples_count": len(triples),
                    "rag_enhanced": True
                },
                success=success
            )
        
        except Exception as e:
            return ProcessingResult(
                triples=[],
                metadata={},
                success=False,
                error=str(e)
            )
```

## Phase 3: Advanced Features (2 Wochen)

### 3.1 API-Server

```python
# api/server.py
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from core.config import Config
from core.pipeline import EnhancedPipeline
from core.document_parser import Document

app = FastAPI(title="Generic Knowledge Graph Pipeline API")

# Globale Pipeline-Instanz
pipeline = None

class DocumentRequest(BaseModel):
    content: str
    format: str
    domain: str = "general"

class QueryRequest(BaseModel):
    question: str
    query_type: str = "hybrid"  # semantic, sparql, hybrid
    k: int = 5

class QueryResponse(BaseModel):
    results: List[Dict]
    query_type: str
    response_time: float

@app.on_event("startup")
async def startup_event():
    global pipeline
    config = Config.from_file("config/default.yaml")
    pipeline = EnhancedPipeline(config)

@app.post("/documents/process")
async def process_document(request: DocumentRequest):
    """Verarbeitet Dokument und erstellt Knowledge Graph"""
    try:
        document = Document(
            content=request.content,
            format=request.format,
            metadata={"domain": request.domain},
            source="api_request"
        )
        
        result = pipeline.process_document_with_rag(document)
        
        return {
            "success": result.success,
            "triples_count": len(result.triples),
            "metadata": result.metadata,
            "error": result.error
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload und Verarbeitung von Dokumenten"""
    try:
        content = await file.read()
        
        document = Document(
            content=content.decode('utf-8'),
            format=file.filename.split('.')[-1],
            metadata={"filename": file.filename},
            source=file.filename
        )
        
        result = pipeline.process_document_with_rag(document)
        
        return {
            "filename": file.filename,
            "success": result.success,
            "triples_count": len(result.triples),
            "metadata": result.metadata
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_knowledge(request: QueryRequest):
    """Hybrid-Query über Vector Store und Knowledge Graph"""
    import time
    start_time = time.time()
    
    try:
        if request.query_type == "semantic":
            # Nur Vector-basierte Suche
            results = pipeline.rag_processor.vector_store.similarity_search(
                request.question, 
                k=request.k
            )
        elif request.query_type == "sparql":
            # SPARQL-Query generieren und ausführen
            sparql_query = f"""
            SELECT ?s ?p ?o WHERE {{
                ?s ?p ?o .
                FILTER(CONTAINS(LCASE(STR(?s)), LCASE("{request.question}")) || 
                       CONTAINS(LCASE(STR(?o)), LCASE("{request.question}")))
            }}
            LIMIT {request.k}
            """
            results = pipeline.kg_store.query(sparql_query)
        else:  # hybrid
            # Kombinierte Suche
            vector_results = pipeline.rag_processor.vector_store.similarity_search(
                request.question, k=request.k//2
            )
            
            sparql_query = f"""
            SELECT ?s ?p ?o WHERE {{
                ?s ?p ?o .
                FILTER(CONTAINS(LCASE(STR(?s)), LCASE("{request.question}")) || 
                       CONTAINS(LCASE(STR(?o)), LCASE("{request.question}")))
            }}
            LIMIT {request.k//2}
            """
            sparql_results = pipeline.kg_store.query(sparql_query)
            
            results = {
                "vector_results": vector_results,
                "sparql_results": sparql_results
            }
        
        response_time = time.time() - start_time
        
        return QueryResponse(
            results=results,
            query_type=request.query_type,
            response_time=response_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health Check Endpoint"""
    return {"status": "healthy", "pipeline": "ready"}

@app.get("/stats")
async def get_statistics():
    """Statistiken über Knowledge Graph"""
    # Triple-Statistiken
    triple_count_query = "SELECT (COUNT(*) as ?count) WHERE { ?s ?p ?o }"
    triple_count = pipeline.kg_store.query(triple_count_query)
    
    # Vector-Statistiken
    vector_count = pipeline.rag_processor.vector_store.collection.count()
    
    return {
        "triples_count": triple_count[0]["count"]["value"] if triple_count else 0,
        "vectors_count": vector_count,
        "domains": [pipeline.config.domain.name]
    }
```

### 3.2 Erweiterte Konfiguration

```yaml
# config/advanced.yaml
domain:
  name: "general"
  ontology_path: "plugins/ontologies/general.ttl"
  enabled_formats: ["pdf", "docx", "xlsx", "pptx", "txt", "html"]

chunking:
  max_tokens: 2000
  overlap_ratio: 0.2
  preserve_context: true
  adaptive_chunking: true

llm:
  provider: "ollama"
  model: "qwen:72b"
  temperature: 0.1
  max_tokens: 4000
  timeout: 60

storage:
  triple_store_url: "http://localhost:3030"
  vector_store_path: "data/vectors"
  backup_enabled: true
  backup_interval: 3600

rag:
  similarity_threshold: 0.7
  max_context_chunks: 3
  enable_validation: true
  enable_deduplication: true

api:
  host: "0.0.0.0"
  port: 8000
  cors_enabled: true
  rate_limiting: true
  max_requests_per_minute: 100

monitoring:
  metrics_enabled: true
  logging_level: "INFO"
  export_metrics: true
```

### 3.3 Monitoring & Metrics

```python
# core/monitoring.py
from typing import Dict, List
import time
import json
from datetime import datetime

class MetricsCollector:
    def __init__(self):
        self.metrics = {
            "documents_processed": 0,
            "triples_extracted": 0,
            "vectors_created": 0,
            "processing_times": [],
            "error_count": 0,
            "start_time": time.time()
        }
    
    def record_processing(self, duration: float, triples_count: int, vectors_count: int):
        """Zeichnet Processing-Metriken auf"""
        self.metrics["documents_processed"] += 1
        self.metrics["triples_extracted"] += triples_count
        self.metrics["vectors_created"] += vectors_count
        self.metrics["processing_times"].append(duration)
    
    def record_error(self, error: str):
        """Zeichnet Fehler auf"""
        self.metrics["error_count"] += 1
        
        # Fehler-Log (könnte in DB/File gespeichert werden)
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "error": error
        }
        print(f"ERROR: {error_entry}")
    
    def get_metrics(self) -> Dict:
        """Gibt aktuelle Metriken zurück"""
        uptime = time.time() - self.metrics["start_time"]
        avg_processing_time = (
            sum(self.metrics["processing_times"]) / len(self.metrics["processing_times"])
            if self.metrics["processing_times"] else 0
        )
        
        return {
            "uptime_seconds": uptime,
            "documents_processed": self.metrics["documents_processed"],
            "triples_extracted": self.metrics["triples_extracted"],
            "vectors_created": self.metrics["vectors_created"],
            "error_count": self.metrics["error_count"],
            "average_processing_time": avg_processing_time,
            "throughput_docs_per_second": self.metrics["documents_processed"] / uptime if uptime > 0 else 0
        }
```

### 3.4 Tests

```python
# tests/test_pipeline.py
import pytest
from core.config import Config
from core.pipeline import EnhancedPipeline
from core.document_parser import Document

def test_text_processing():
    """Test basic text processing"""
    config = Config.from_file("config/test.yaml")
    pipeline = EnhancedPipeline(config)
    
    document = Document(
        content="Apple Inc. is a technology company founded by Steve Jobs.",
        format="txt",
        metadata={"test": True},
        source="test"
    )
    
    result = pipeline.process_document_with_rag(document)
    
    assert result.success
    assert len(result.triples) > 0
    assert any("Apple" in str(triple.subject) for triple in result.triples)

def test_rag_enhancement():
    """Test RAG enhancement functionality"""
    config = Config.from_file("config/test.yaml")
    pipeline = EnhancedPipeline(config)
    
    # First document
    doc1 = Document(
        content="Tesla produces electric vehicles.",
        format="txt",
        metadata={},
        source="test1"
    )
    
    # Second document (should find context from first)
    doc2 = Document(
        content="Elon Musk is the CEO of Tesla.",
        format="txt",
        metadata={},
        source="test2"
    )
    
    result1 = pipeline.process_document_with_rag(doc1)
    result2 = pipeline.process_document_with_rag(doc2)
    
    assert result1.success
    assert result2.success
    assert len(result2.triples) > 0

def test_similarity_search():
    """Test vector similarity search"""
    config = Config.from_file("config/test.yaml")
    pipeline = EnhancedPipeline(config)
    
    # Add test content
    doc = Document(
        content="Artificial intelligence is transforming industries.",
        format="txt",
        metadata={},
        source="test"
    )
    
    pipeline.process_document_with_rag(doc)
    
    # Search for similar content
    results = pipeline.rag_processor.vector_store.similarity_search(
        "AI and machine learning", k=5
    )
    
    assert len(results) > 0
    assert any("artificial intelligence" in result['content'].lower() for result in results)
```

## Deployment

### 3.4 Horizontale Skalierung (GPU-Model-bewusst)

```python
# core/gpu_scaling.py
from typing import List, Dict
import redis
from dataclasses import dataclass

@dataclass
class NodeConfig:
    node_id: str
    gpu_available: bool
    max_memory: int
    endpoint: str

class GPUAwareScaler:
    """GPU-bewusste Skalierung über separate Nodes statt Worker-Threads"""
    
    def __init__(self, config: ScalingConfig):
        self.nodes = [NodeConfig(**n) for n in config.nodes]
        self.redis_client = redis.Redis(host=config.redis_host)
        self.queue_name = "document_processing_queue"
    
    def distribute_batch(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """Verteilt Dokumente auf GPU-Nodes (NICHT Threads!)"""
        
        # Nodes nach GPU-Verfügbarkeit sortieren
        gpu_nodes = [n for n in self.nodes if n.gpu_available]
        cpu_nodes = [n for n in self.nodes if not n.gpu_available]
        
        distributions = {}
        
        # PDF-Dokumente zu GPU-Nodes (SmolDocling)
        pdf_docs = [d for d in documents if d.format == 'pdf']
        for i, doc in enumerate(pdf_docs):
            node = gpu_nodes[i % len(gpu_nodes)]
            if node.node_id not in distributions:
                distributions[node.node_id] = []
            distributions[node.node_id].append(doc)
        
        # Andere Formate können auf CPU-Nodes
        other_docs = [d for d in documents if d.format != 'pdf']
        all_nodes = gpu_nodes + cpu_nodes
        for i, doc in enumerate(other_docs):
            node = all_nodes[i % len(all_nodes)]
            if node.node_id not in distributions:
                distributions[node.node_id] = []
            distributions[node.node_id].append(doc)
        
        return distributions
    
    def submit_to_node(self, node_id: str, documents: List[Document]):
        """Sendet Dokumente an spezifischen Node zur Verarbeitung"""
        task = {
            'node_id': node_id,
            'documents': [d.dict() for d in documents],
            'timestamp': datetime.now().isoformat()
        }
        
        self.redis_client.lpush(f"{self.queue_name}:{node_id}", json.dumps(task))

# Node Worker (läuft in separatem Container/Prozess)
class NodeWorker:
    """Worker für einzelnen GPU/CPU-Node"""
    
    def __init__(self, node_id: str, has_gpu: bool = True):
        self.node_id = node_id
        self.has_gpu = has_gpu
        self.pipeline = None
        self.model_loaded = False
        
    def start(self):
        """Startet Node-Worker mit einmaligem Model-Loading"""
        
        # Pipeline einmal initialisieren
        config = Config.from_file("config/node.yaml")
        self.pipeline = EnhancedPipeline(config)
        
        # Model einmal laden (wenn GPU vorhanden)
        if self.has_gpu:
            logger.info(f"Node {self.node_id}: Loading GPU models once...")
            self.pipeline.initialize_gpu_models()
            self.model_loaded = True
        
        # Message-Loop für sequenzielle Verarbeitung
        while True:
            task = self._get_next_task()
            if task:
                results = self._process_task_sequential(task)
                self._store_results(results)
    
    def _process_task_sequential(self, task: Dict) -> List[ProcessingResult]:
        """Sequenzielle Verarbeitung auf diesem Node"""
        results = []
        
        for doc_data in task['documents']:
            doc = Document(**doc_data)
            
            # Sequenzielle Verarbeitung (KEIN Multi-Threading!)
            result = self.pipeline.process_document_with_rag(doc)
            results.append(result)
            
            logger.info(f"Node {self.node_id}: Processed {doc.source}")
        
        return results
```

**Deployment mit Docker Compose für Skalierung:**

```yaml
# docker-compose.scaling.yml
version: '3.8'

services:
  # GPU Node 1
  gpu-node-1:
    build: .
    environment:
      - NODE_ID=gpu_node_1
      - HAS_GPU=true
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: python -m core.node_worker --node-id gpu_node_1
    
  # GPU Node 2
  gpu-node-2:
    build: .
    environment:
      - NODE_ID=gpu_node_2
      - HAS_GPU=true
      - CUDA_VISIBLE_DEVICES=1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: python -m core.node_worker --node-id gpu_node_2
    
  # CPU Node für Non-PDF
  cpu-node-1:
    build: .
    environment:
      - NODE_ID=cpu_node_1
      - HAS_GPU=false
    command: python -m core.node_worker --node-id cpu_node_1
    
  # Message Queue
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
```

### 3.5 Docker Setup

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download sentence transformer model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Application code
COPY . .

# Create data directories
RUN mkdir -p data/vectors data/cache

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 3.6 Production docker-compose.yml

```yaml
# docker-compose.yml
version: '3.8'

services:
  kg-pipeline:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./config:/app/config
    environment:
      - DOMAIN=general
      - LLM_PROVIDER=ollama
      - FUSEKI_URL=http://fuseki:3030
    depends_on:
      - fuseki
      - ollama
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
  
  fuseki:
    image: stain/jena-fuseki:latest
    ports:
      - "3030:3030"
    volumes:
      - fuseki_data:/fuseki/databases
    environment:
      - ADMIN_PASSWORD=admin123
      - JVM_ARGS=-Xmx2g
    restart: unless-stopped
  
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    restart: unless-stopped
  
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - kg-pipeline
    restart: unless-stopped

volumes:
  fuseki_data:
  ollama_data:
```

## Fazit

Dieser Implementierungsleitfaden bietet eine strukturierte Herangehensweise für die Entwicklung eines schlanken, generischen Knowledge Graph Pipeline Systems. Das System wird in 3 Phasen entwickelt:

1. **MVP (1-2 Wochen)**: Grundfunktionalität mit 15 Kernkomponenten
2. **RAG-Integration (1 Woche)**: Hybrid Vector+Graph Retrieval
3. **Advanced Features (2 Wochen)**: API, Monitoring, Production-Setup

**Vorteile gegenüber dem alten System:**
- 70% weniger Code
- Domänen-agnostisch
- Bessere Performance durch sequenzielle Verarbeitung
- Einfachere Wartung durch Configuration-First Ansatz
- Moderne RAG-Funktionalität

**Nächste Schritte:**
1. MVP-Implementation starten
2. Erste Tests mit einfachen Dokumenten
3. Domänen-spezifische Ontologien entwickeln
4. RAG-Integration für verbesserte Extraktion
5. Production-Deployment mit Monitoring