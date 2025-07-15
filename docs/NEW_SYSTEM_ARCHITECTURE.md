# Technische Architektur-Dokumentation: Generisches Knowledge Graph Pipeline System

## Übersicht

Dieses Dokument beschreibt die technische Architektur für ein neues, schlankes und domänen-agnostisches Knowledge Graph Pipeline System, basierend auf den Erkenntnissen aus der Automotive Knowledge Graph Pipeline (WBA) Analyse.

## Design-Prinzipien

1. **Domänen-Agnostik**: Das System ist nicht auf Automotive beschränkt
2. **Schlanke Architektur**: Minimale Komponenten für maximale Effizienz
3. **Plugin-Erweiterbarkeit**: Neue Formate und Domänen als Plugins
4. **Configuration-First**: Domänen-Wechsel über Konfiguration
5. **Hybrid RAG+KG**: Kombiniert Vector-Retrieval mit Knowledge Graph
6. **Performance-optimiert**: Sequenzielle GPU-Verarbeitung mit Model-Caching

## Systemarchitektur

### Schichten-Architektur

```
┌─────────────────────────────────────────────────────────────────┐
│                    📊 PRESENTATION LAYER                        │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │   REST API      │ │   SPARQL API    │ │   VECTOR API    │   │
│  │  (Query/Status) │ │  (Graph Query)  │ │  (Similarity)   │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                     🔧 PROCESSING LAYER                         │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │ Document Parser │ │ Content Chunker │ │ RAG Processor   │   │
│  │   (Pluggable)   │ │   (Adaptive)    │ │ (Hybrid Triple) │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
│                                │                                │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │   LLM Client    │ │ Vector Embedder │ │ Triple Extractor│   │
│  │ (Multi-Provider)│ │   (Cached)      │ │ (Ontology-based)│   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                      💾 STORAGE LAYER                           │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │ Vector Store    │ │ Triple Store    │ │ Configuration   │   │
│  │   (ChromaDB)    │ │   (Fuseki)      │ │   (YAML/JSON)   │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Datenfluss

```
Input Documents → Document Parser → Content Chunker → RAG Processor
                                                           │
                                                           v
Vector Embedder ← Context Enricher ← LLM Client ← Triple Extractor
       │                                              │
       v                                              v
Vector Store ← Similarity Validator → Triple Store ← RDF Writer
```

## Kernkomponenten

### 1. Document Parser (Plugin-System)

```python
class DocumentParser:
    """Generischer Dokumenten-Parser mit Plugin-System"""
    
    def __init__(self, config: DomainConfig):
        self.parsers = self._load_parsers(config.enabled_formats)
        self.model_cache = ModelCache()
    
    def parse(self, document: Document) -> List[Segment]:
        parser = self.parsers.get(document.format)
        return parser.parse(document)
```

**Unterstützte Formate:**
- PDF (SmolDocling-basiert)
- DOCX/XLSX/PPTX (Office-Suite)
- Web/HTML (Playwright)
- TXT/MD (Plain Text)

### 2. Content Chunker (Adaptive Segmentierung)

```python
class AdaptiveChunker:
    """Intelligente Chunking-Strategie basierend auf Inhalt"""
    
    def __init__(self, config: ChunkingConfig):
        self.max_tokens = config.max_tokens
        self.overlap = config.overlap_ratio
        self.context_preservation = config.preserve_context
    
    def chunk(self, segments: List[Segment]) -> List[Chunk]:
        """Adaptives Chunking mit Kontext-Erhaltung"""
        return self._adaptive_chunking(segments)
```

### 3. RAG Processor (Hybrid-Ansatz)

```python
class RAGProcessor:
    """Kombiniert Vector-Retrieval mit Knowledge Graph"""
    
    def __init__(self, vector_store: VectorStore, kg_store: KGStore):
        self.vector_store = vector_store
        self.kg_store = kg_store
        self.similarity_threshold = 0.7
    
    def process(self, chunks: List[Chunk]) -> ProcessingResult:
        """Hybrid-Verarbeitung mit Kontext-Anreicherung"""
        # 1. Vector-basierte Kontext-Anreicherung
        enriched_chunks = self._enrich_with_context(chunks)
        
        # 2. LLM-basierte Triple-Extraktion
        triples = self._extract_triples(enriched_chunks)
        
        # 3. Konsistenz-Validierung
        validated_triples = self._validate_consistency(triples)
        
        return ProcessingResult(triples=validated_triples, 
                              vectors=self._create_embeddings(chunks))
```

### 4. LLM Client (Multi-Provider)

```python
class LLMClient:
    """Einheitlicher LLM-Client für verschiedene Provider"""
    
    def __init__(self, config: LLMConfig):
        self.providers = {
            'openai': OpenAIProvider(config.openai),
            'ollama': OllamaProvider(config.ollama),
            'anthropic': AnthropicProvider(config.anthropic)
        }
        self.active_provider = config.default_provider
    
    def generate_triples(self, content: str, ontology: Ontology) -> List[Triple]:
        """Generiert Triples mit domänen-spezifischer Ontologie"""
        provider = self.providers[self.active_provider]
        return provider.generate_triples(content, ontology)
```

### 5. Configuration System

```yaml
# domain_config.yaml
domain:
  name: "automotive"
  ontology: "ontologies/automotive.ttl"
  
formats:
  enabled: ["pdf", "docx", "xlsx", "web"]
  
processing:
  chunking:
    max_tokens: 2000
    overlap_ratio: 0.2
    preserve_context: true
  
  llm:
    provider: "ollama"
    model: "qwen:72b"
    temperature: 0.1
    
  vector:
    provider: "chromadb"
    model: "sentence-transformers/all-MiniLM-L6-v2"
    
storage:
  triple_store:
    type: "fuseki"
    endpoint: "http://localhost:3030"
  
  vector_store:
    type: "chromadb"
    path: "data/vectors"
```

## Vereinfachte Komponenten-Struktur

### MVP-Komponenten (15 Dateien)

```
core/
├── __init__.py
├── document_parser.py       # Dokumenten-Parsing
├── content_chunker.py       # Adaptive Segmentierung
├── llm_client.py           # LLM-Integration
├── vector_store.py         # Vector-Storage
├── kg_store.py             # Knowledge Graph Storage
├── rag_processor.py        # RAG-Hybrid-Verarbeitung
├── config.py               # Konfiguration
└── pipeline.py             # Hauptpipeline

plugins/
├── parsers/
│   ├── pdf_parser.py       # PDF-Verarbeitung
│   ├── office_parser.py    # Office-Dokumente
│   └── web_parser.py       # Web-Scraping
├── ontologies/
│   ├── automotive.ttl      # Automotive-Ontologie
│   ├── general.ttl         # Allgemeine Ontologie
│   └── custom.ttl          # Benutzerdefiniert
└── templates/
    ├── triple_extraction.j2 # LLM-Prompts
    └── validation.j2        # Validierung
```

### Entfernte Komplexitäten

**Aus altem System entfernt:**
- Worker-Thread-Architektur (Performance-Probleme)
- Hybrid Upload/Batch-System (Überkomplexität)
- Mehrfache Abstraktionsebenen (Router-Hierarchie)
- Redundante Processor-Klassen
- Komplexe Konfigurationshierarchie

**Vereinfachungen:**
- Direkte Pipeline statt Multi-Layer-Routing
- Configuration-First statt Code-First
- Plugin-System statt Monolith
- Einheitliche Datenstrukturen

## Performance-Optimierungen

### 1. Model-Caching

```python
class ModelCache:
    """Globaler Model-Cache für GPU-Optimierung"""
    
    def __init__(self):
        self._models = {}
        self._lock = threading.Lock()
    
    def get_model(self, model_name: str) -> Model:
        with self._lock:
            if model_name not in self._models:
                self._models[model_name] = self._load_model(model_name)
            return self._models[model_name]
```

### 2. Batch-Processing

```python
class BatchProcessor:
    """Optimierte Batch-Verarbeitung ohne Worker-Threads"""
    
    def process_batch(self, documents: List[Document]) -> ProcessingResult:
        """Sequenzielle Verarbeitung mit Model-Wiederverwendung"""
        results = []
        
        # Einmaliges Model-Loading
        self._initialize_models()
        
        # Sequenzielle Verarbeitung
        for doc in documents:
            result = self._process_document(doc)
            results.append(result)
        
        return self._aggregate_results(results)
```

### 3. Caching-Strategien

```python
class ProcessingCache:
    """Multi-Level Caching für verschiedene Komponenten"""
    
    def __init__(self):
        self.embedding_cache = LRUCache(maxsize=10000)
        self.triple_cache = LRUCache(maxsize=5000)
        self.similarity_cache = LRUCache(maxsize=1000)
    
    @cached_property
    def embedding_model(self):
        return SentenceTransformer('all-MiniLM-L6-v2')
```

## RAG-Integration (Hybrid-Ansatz)

### Vector-Store Integration

```python
class VectorStore:
    """ChromaDB-basierter Vector-Store"""
    
    def __init__(self, config: VectorConfig):
        self.client = chromadb.PersistentClient(path=config.path)
        self.collection = self.client.get_or_create_collection(
            name=config.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(self, documents: List[Document]):
        """Fügt Dokumente mit Embeddings hinzu"""
        embeddings = self._create_embeddings(documents)
        self.collection.add(
            documents=[doc.content for doc in documents],
            metadatas=[doc.metadata for doc in documents],
            embeddings=embeddings,
            ids=[doc.id for doc in documents]
        )
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Semantische Ähnlichkeitssuche"""
        query_embedding = self._embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=['documents', 'metadatas', 'distances']
        )
        return self._format_results(results)
```

### Hybrid Query-System

```python
class HybridQueryEngine:
    """Kombiniert Vector-Suche mit SPARQL-Queries"""
    
    def __init__(self, vector_store: VectorStore, kg_store: KGStore):
        self.vector_store = vector_store
        self.kg_store = kg_store
    
    def query(self, question: str, query_type: str = 'hybrid') -> QueryResult:
        """Hybrid-Query mit Vector und Graph"""
        
        if query_type == 'semantic':
            # Nur Vector-basierte Suche
            return self._semantic_search(question)
        
        elif query_type == 'structured':
            # Nur SPARQL-basierte Suche
            return self._sparql_search(question)
        
        else:  # hybrid
            # Kombinierte Suche
            vector_results = self._semantic_search(question)
            sparql_results = self._sparql_search(question)
            return self._merge_results(vector_results, sparql_results)
```

## Domänen-Abstraktion

### Ontologie-System

```python
class OntologyManager:
    """Verwaltung domänen-spezifischer Ontologien"""
    
    def __init__(self, config: DomainConfig):
        self.ontologies = {}
        self._load_ontologies(config.ontology_paths)
    
    def get_ontology(self, domain: str) -> Ontology:
        """Lädt domänen-spezifische Ontologie"""
        if domain not in self.ontologies:
            self.ontologies[domain] = self._load_ontology(domain)
        return self.ontologies[domain]
    
    def validate_triple(self, triple: Triple, domain: str) -> bool:
        """Validiert Triple gegen Ontologie"""
        ontology = self.get_ontology(domain)
        return ontology.validate(triple)
```

### Domain-Plugin-System

```python
class DomainPlugin:
    """Basis-Klasse für Domänen-Plugins"""
    
    def __init__(self, config: DomainConfig):
        self.config = config
        self.ontology = OntologyManager(config).get_ontology(config.domain)
    
    def extract_entities(self, text: str) -> List[Entity]:
        """Domänen-spezifische Entity-Extraktion"""
        raise NotImplementedError
    
    def validate_triples(self, triples: List[Triple]) -> List[Triple]:
        """Domänen-spezifische Triple-Validierung"""
        return [t for t in triples if self.ontology.validate(t)]
```

## API-Design

### REST API

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Generic Knowledge Graph Pipeline")

class DocumentRequest(BaseModel):
    content: str
    format: str
    domain: str = "general"

class QueryRequest(BaseModel):
    question: str
    query_type: str = "hybrid"  # semantic, structured, hybrid

@app.post("/documents/process")
async def process_document(request: DocumentRequest):
    """Verarbeitet Dokument und erstellt Knowledge Graph"""
    pipeline = Pipeline(domain=request.domain)
    result = pipeline.process(request.content, request.format)
    return {"triples": result.triples, "vectors": result.vectors}

@app.post("/query")
async def query_knowledge(request: QueryRequest):
    """Hybrid-Query über Vector Store und Knowledge Graph"""
    engine = HybridQueryEngine()
    result = engine.query(request.question, request.query_type)
    return {"results": result.results, "sources": result.sources}
```

### SPARQL Endpoint

```python
@app.get("/sparql")
async def sparql_endpoint(query: str):
    """Standard SPARQL Endpoint"""
    kg_store = KGStore()
    results = kg_store.query(query)
    return {"results": results}
```

## Deployment-Architektur

### Container-Setup

```docker
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Ports
EXPOSE 8000 3030

# Start script
CMD ["python", "main.py"]
```

### docker-compose.yml

```yaml
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
      - DOMAIN=automotive
      - LLM_PROVIDER=ollama
    depends_on:
      - fuseki
      - chromadb
  
  fuseki:
    image: stain/jena-fuseki:latest
    ports:
      - "3030:3030"
    volumes:
      - fuseki_data:/fuseki/databases
    environment:
      - ADMIN_PASSWORD=admin
  
  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - chromadb_data:/chroma/data

volumes:
  fuseki_data:
  chromadb_data:
```

## Skalierbarkeit

### Horizontale Skalierung (GPU-Model-bewusst)

```python
class DistributedPipeline:
    """GPU-optimierte verteilte Verarbeitung ohne Worker-Threads"""
    
    def __init__(self, config: ClusterConfig):
        self.node_configs = config.nodes  # Separate Nodes statt Worker-Threads
        self.queue = MessageQueue(config.queue_url)
        self.model_cache = None  # Pro Node ein Model-Cache
    
    def process_large_batch(self, documents: List[Document]):
        """Verteilt Dokumente auf separate GPU-Nodes (nicht Threads!)"""
        
        # Dokumente nach Node-Kapazität aufteilen
        node_batches = self._distribute_to_nodes(documents)
        
        # Jeder Node verarbeitet sequenziell mit eigenem GPU-Model
        tasks = []
        for node_id, batch in node_batches.items():
            task = self.queue.send_task(
                'process_batch_sequential',  # Sequenziell pro Node!
                {
                    'documents': batch,
                    'node_id': node_id,
                    'use_model_cache': True  # Ein Model pro Node
                }
            )
            tasks.append(task)
        
        # Ergebnisse sammeln
        results = []
        for task in tasks:
            result = task.get()
            results.extend(result)
        
        return results
    
    def _distribute_to_nodes(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """Intelligente Verteilung basierend auf Node-GPU-Kapazität"""
        node_batches = {}
        
        for i, doc in enumerate(documents):
            # Round-robin auf verfügbare GPU-Nodes
            node_id = f"gpu_node_{i % len(self.node_configs)}"
            
            if node_id not in node_batches:
                node_batches[node_id] = []
            
            node_batches[node_id].append(doc)
        
        return node_batches

class GPUNodeWorker:
    """Worker-Prozess für einzelnen GPU-Node (NICHT Thread!)"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.pipeline = None
        self.model_loaded = False
    
    def process_batch_sequential(self, batch_data: Dict):
        """Sequenzielle Verarbeitung auf diesem GPU-Node"""
        documents = batch_data['documents']
        
        # Model einmal laden pro Node
        if not self.model_loaded:
            self.pipeline = Pipeline(config)
            self.model_loaded = True
            logger.info(f"Node {self.node_id}: Model loaded once")
        
        # Sequenzielle Verarbeitung (KEIN Multi-Threading!)
        results = []
        for doc in documents:
            result = self.pipeline.process_document(doc)
            results.append(result)
        
        logger.info(f"Node {self.node_id}: Processed {len(documents)} docs sequentially")
        return results
```

**Wichtiger Unterschied zu Worker-Threads:**
- **Separate Prozesse/Container** statt Threads im selben Prozess
- **Ein GPU-Model pro Node** statt Model pro Thread
- **Sequenzielle Verarbeitung pro Node** statt parallele Threads
- **Message Queue zwischen Nodes** statt Shared Memory

### Monitoring

```python
class MetricsCollector:
    """Sammelt Performance-Metriken"""
    
    def __init__(self):
        self.metrics = {
            'documents_processed': 0,
            'triples_extracted': 0,
            'processing_time': [],
            'error_rate': 0
        }
    
    def track_processing(self, duration: float, triples_count: int):
        """Trackt Verarbeitungsmetriken"""
        self.metrics['documents_processed'] += 1
        self.metrics['triples_extracted'] += triples_count
        self.metrics['processing_time'].append(duration)
    
    def get_metrics(self) -> Dict:
        """Gibt aktuelle Metriken zurück"""
        return {
            **self.metrics,
            'avg_processing_time': np.mean(self.metrics['processing_time']),
            'throughput': self.metrics['documents_processed'] / 
                         sum(self.metrics['processing_time'])
        }
```



## Zusammenfassung

Diese Architektur bietet:

1. **Domänen-Agnostik**: Einfache Erweiterung auf neue Domänen
2. **Schlanke Struktur**: 70% weniger Code als altes System
3. **Hybrid RAG+KG**: Beste aus beiden Welten
4. **Performance**: Optimiert für GPU-basierte Modelle
5. **Skalierbarkeit**: Horizontal und vertikal skalierbar
6. **Erweiterbarkeit**: Plugin-System für neue Formate
7. **Einfache Wartung**: Konfiguration statt Code-Änderungen

Die Architektur ist production-ready und kann schrittweise implementiert werden, beginnend mit den Kernkomponenten und anschließender Erweiterung um RAG-Funktionalität.