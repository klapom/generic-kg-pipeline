# API-Dokumentation: Generisches Knowledge Graph Pipeline System

## Übersicht

Diese Dokumentation beschreibt die REST API und Schnittstellen des generischen Knowledge Graph Pipeline Systems. Das System bietet sowohl programmatische APIs als auch Web-Schnittstellen für die Dokumentenverarbeitung und Knowledge Graph-Abfragen.

## 🎯 Aktueller Implementierungsstatus

### ✅ IMPLEMENTIERTE API-ENDPUNKTE (80% vollständig)

**Document Processing APIs (100%):**
- ✅ `POST /documents/process` - Einzeldokument-Verarbeitung
- ✅ `POST /documents/upload` - Multi-Modal File Upload (PDF, DOCX, XLSX, PPTX)
- ✅ `POST /documents/batch` - Batch-Verarbeitung mit Context Inheritance
- ✅ Multi-Modal Content Support mit vLLM SmolDocling und Qwen2.5-VL

**System Management APIs (100%):**
- ✅ `GET /health` - Health Check mit Component Status
- ✅ `GET /stats` - Detaillierte Systemstatistiken
- ✅ FastAPI Application mit automatischer OpenAPI-Dokumentation

**Configuration APIs (80%):**
- ✅ YAML-basierte Konfiguration für Chunking und LLM-Clients
- ✅ Domain-agnostische Architektur
- 🔄 Runtime-Konfiguration via API (in Entwicklung)

### 🔄 IN ENTWICKLUNG
- Query APIs (Vector Search, SPARQL, Hybrid Queries)
- Triple Store Integration (Fuseki)
- Vector Store Integration (ChromaDB)
- WebSocket APIs für Real-time Updates
- Admin APIs für Node-Management

### 📋 AKTUELL VERFÜGBARE ENDPUNKTE

Basierend auf der implementierten FastAPI-Anwendung in `api/` sind folgende Endpunkte verfügbar:

**Health & Monitoring:**
- ✅ `GET /health` - Umfassender Health Check aller Services
- ✅ `GET /health/liveness` - Kubernetes Liveness Probe
- ✅ `GET /health/readiness` - Kubernetes Readiness Probe  
- ✅ `GET /health/services/{service_name}` - Detaillierte Service-Informationen

**Document Processing:**
- ✅ `POST /documents/upload` - Multi-Modal File Upload (PDF, DOCX, XLSX, PPTX)
- ✅ `POST /documents/batch` - Batch-Verarbeitung mit Context Inheritance
- ✅ `GET /documents/{document_id}` - Document Status und Metadaten

**Pipeline Management:**
- ✅ `POST /pipeline/process` - Document Pipeline Processing
- ✅ `GET /pipeline/status` - Pipeline Status und Statistiken

**System APIs:**
- ✅ `GET /` - Root Endpoint mit API-Informationen
- ✅ `GET /docs` - Automatische OpenAPI/Swagger-Dokumentation
- ✅ `GET /redoc` - ReDoc-Dokumentation

Die folgenden Endpunkte sind in der Dokumentation beschrieben, aber noch in Entwicklung:

## Base URL

```
Production: https://api.kg-pipeline.com
Development: http://localhost:8000
```

## Authentication

Aktuell verwendet das System API-Key-basierte Authentifizierung (optional für MVP):

```bash
# Header für authentifizierte Requests
Authorization: Bearer YOUR_API_KEY
```

## Allgemeine Response-Struktur

```json
{
    "success": true,
    "data": {...},
    "error": null,
    "timestamp": "2025-01-15T10:30:00Z",
    "processing_time": 1.234
}
```

Bei Fehlern:

```json
{
    "success": false,
    "data": null,
    "error": {
        "code": "PROCESSING_ERROR",
        "message": "Failed to parse document",
        "details": "..."
    },
    "timestamp": "2025-01-15T10:30:00Z"
}
```

## Endpunkte

### 1. Document Processing

#### POST /documents/process

Verarbeitet Dokument-Inhalte und erstellt Knowledge Graph.

**Request Body:**
```json
{
    "content": "Apple Inc. is a technology company founded by Steve Jobs in 1976.",
    "format": "txt",
    "domain": "general",
    "options": {
        "enable_rag": true,
        "chunk_size": 2000,
        "extract_entities": true
    }
}
```

**Parameters:**
- `content` (string, required): Dokumenteninhalt als Text
- `format` (string, required): Dokumentformat (`pdf`, `docx`, `txt`, `html`)
- `domain` (string, optional): Domänen-Kontext (default: `general`)
- `options` (object, optional): Verarbeitungsoptionen

**Response:**
```json
{
    "success": true,
    "data": {
        "triples_count": 15,
        "vectors_created": 8,
        "metadata": {
            "segments_count": 3,
            "chunks_count": 2,
            "processing_time": 2.456,
            "domain": "general",
            "rag_enhanced": true
        },
        "triples": [
            {
                "subject": "kg:Apple_Inc",
                "predicate": "kg:foundedBy",
                "object": "kg:Steve_Jobs",
                "confidence": 0.95
            }
        ]
    }
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/documents/process \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Tesla produces electric vehicles.",
    "format": "txt",
    "domain": "automotive"
  }'
```

#### POST /documents/upload

Upload und Verarbeitung von Dokumentdateien.

**Request:**
- `Content-Type: multipart/form-data`
- `file`: Dokumentdatei (PDF, DOCX, TXT, etc.)
- `domain` (optional): Domänen-Kontext
- `options` (optional): JSON-String mit Verarbeitungsoptionen

**Response:**
```json
{
    "success": true,
    "data": {
        "filename": "document.pdf",
        "file_size": 1024,
        "triples_count": 42,
        "vectors_created": 15,
        "processing_summary": {
            "pages_processed": 5,
            "entities_extracted": 28,
            "processing_time": 8.234
        }
    }
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@/path/to/document.pdf" \
  -F "domain=automotive" \
  -F 'options={"enable_rag": true}'
```

#### POST /documents/batch

Verarbeitung mehrerer Dokumente als Batch.

**Request Body:**
```json
{
    "documents": [
        {
            "content": "Document 1 content...",
            "format": "txt",
            "source": "doc1.txt"
        },
        {
            "content": "Document 2 content...",
            "format": "pdf",
            "source": "doc2.pdf"
        }
    ],
    "domain": "general",
    "options": {
        "parallel_processing": false,
        "enable_rag": true
    }
}
```

**Response:**
```json
{
    "success": true,
    "data": {
        "total_documents": 2,
        "successfully_processed": 2,
        "failed": 0,
        "total_triples": 67,
        "total_vectors": 23,
        "processing_time": 15.678,
        "results": [
            {
                "source": "doc1.txt",
                "success": true,
                "triples_count": 35,
                "vectors_count": 12
            }
        ]
    }
}
```

### 2. Knowledge Graph Queries

#### POST /query

Hybrid-Query über Vector Store und Knowledge Graph.

**Request Body:**
```json
{
    "question": "What companies produce electric vehicles?",
    "query_type": "hybrid",
    "k": 10,
    "filters": {
        "domain": "automotive",
        "confidence_threshold": 0.8
    }
}
```

**Parameters:**
- `question` (string, required): Suchanfrage in natürlicher Sprache
- `query_type` (string, optional): Art der Suche
  - `semantic`: Nur Vector-basierte Suche
  - `sparql`: Nur Knowledge Graph SPARQL
  - `hybrid`: Kombinierte Suche (default)
- `k` (integer, optional): Anzahl Ergebnisse (default: 10)
- `filters` (object, optional): Zusätzliche Filter

**Response:**
```json
{
    "success": true,
    "data": {
        "query_type": "hybrid",
        "results_count": 8,
        "response_time": 0.456,
        "semantic_results": [
            {
                "content": "Tesla is an electric vehicle manufacturer...",
                "similarity_score": 0.92,
                "metadata": {
                    "source": "tesla_doc.pdf",
                    "domain": "automotive"
                }
            }
        ],
        "sparql_results": [
            {
                "subject": "kg:Tesla",
                "predicate": "kg:produces",
                "object": "kg:ElectricVehicle",
                "confidence": 0.95
            }
        ],
        "combined_score": 0.89
    }
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Who founded Apple Inc?",
    "query_type": "hybrid",
    "k": 5
  }'
```

#### GET /sparql

Standard SPARQL Endpoint für direkte Graph-Queries.

**Parameters:**
- `query` (string, required): SPARQL Query
- `format` (string, optional): Response-Format (`json`, `xml`, `turtle`)

**Example SPARQL Query:**
```sparql
SELECT ?subject ?predicate ?object WHERE {
    ?subject ?predicate ?object .
    FILTER(CONTAINS(STR(?subject), "Apple"))
}
LIMIT 10
```

**cURL Example:**
```bash
curl -G http://localhost:8000/sparql \
  --data-urlencode 'query=SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10' \
  --data-urlencode 'format=json'
```

**Response:**
```json
{
    "success": true,
    "data": {
        "head": {
            "vars": ["s", "p", "o"]
        },
        "results": {
            "bindings": [
                {
                    "s": {"type": "uri", "value": "http://example.org/kg/Apple_Inc"},
                    "p": {"type": "uri", "value": "http://example.org/kg/foundedBy"},
                    "o": {"type": "uri", "value": "http://example.org/kg/Steve_Jobs"}
                }
            ]
        }
    }
}
```

### 3. Vector Search

#### POST /vectors/search

Direkte semantische Suche über Vector Store.

**Request Body:**
```json
{
    "query": "artificial intelligence and machine learning",
    "k": 5,
    "filters": {
        "domain": "technology",
        "min_confidence": 0.7
    }
}
```

**Response:**
```json
{
    "success": true,
    "data": {
        "results": [
            {
                "content": "AI and ML are transforming industries...",
                "similarity_score": 0.94,
                "metadata": {
                    "source": "ai_report.pdf",
                    "chunk_id": "chunk_42",
                    "domain": "technology"
                }
            }
        ],
        "query_vector_length": 384,
        "search_time": 0.023
    }
}
```

#### POST /vectors/add

Direktes Hinzufügen von Inhalten zum Vector Store (für erweiterte Anwendungen).

**Request Body:**
```json
{
    "documents": [
        {
            "content": "Content to be vectorized...",
            "metadata": {
                "source": "manual_input",
                "domain": "general"
            }
        }
    ]
}
```

### 4. System Management

#### GET /health

Health Check für System-Status.

**Response:**
```json
{
    "success": true,
    "data": {
        "status": "healthy",
        "components": {
            "pipeline": "ready",
            "vector_store": "connected",
            "knowledge_graph": "connected",
            "llm_provider": "available"
        },
        "uptime": 3600,
        "version": "1.0.0"
    }
}
```

#### GET /stats

Detaillierte Systemstatistiken.

**Response:**
```json
{
    "success": true,
    "data": {
        "documents_processed": 1250,
        "triples_count": 45670,
        "vectors_count": 12340,
        "domains": ["general", "automotive", "technology"],
        "processing_stats": {
            "average_processing_time": 2.45,
            "throughput_docs_per_hour": 1200,
            "error_rate": 0.02
        },
        "storage_stats": {
            "knowledge_graph_size_mb": 45.2,
            "vector_store_size_mb": 123.5,
            "total_storage_mb": 168.7
        }
    }
}
```

#### GET /metrics

Prometheus-kompatible Metriken (für Monitoring).

**Response:**
```
# HELP kg_documents_processed_total Total number of documents processed
# TYPE kg_documents_processed_total counter
kg_documents_processed_total 1250

# HELP kg_triples_extracted_total Total number of triples extracted
# TYPE kg_triples_extracted_total counter
kg_triples_extracted_total 45670

# HELP kg_processing_duration_seconds Time spent processing documents
# TYPE kg_processing_duration_seconds histogram
kg_processing_duration_seconds_bucket{le="1"} 450
kg_processing_duration_seconds_bucket{le="5"} 1100
kg_processing_duration_seconds_bucket{le="10"} 1200
```

### 5. Configuration Management

#### GET /config

Aktuelle Systemkonfiguration abrufen.

**Response:**
```json
{
    "success": true,
    "data": {
        "domain": {
            "name": "general",
            "ontology_path": "plugins/ontologies/general.ttl",
            "enabled_formats": ["pdf", "docx", "txt"]
        },
        "llm": {
            "provider": "ollama",
            "model": "qwen:7b",
            "temperature": 0.1
        },
        "chunking": {
            "max_tokens": 2000,
            "overlap_ratio": 0.2
        }
    }
}
```

#### POST /config/domain

Domänen-Konfiguration ändern (zur Laufzeit).

**Request Body:**
```json
{
    "domain": "automotive",
    "ontology_path": "plugins/ontologies/automotive.ttl",
    "enabled_formats": ["pdf", "docx"]
}
```

**Response:**
```json
{
    "success": true,
    "data": {
        "message": "Domain configuration updated successfully",
        "new_domain": "automotive",
        "reload_required": false
    }
}
```

## Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `INVALID_FORMAT` | Unsupported document format | 400 |
| `PROCESSING_ERROR` | Document processing failed | 422 |
| `LLM_UNAVAILABLE` | LLM service not available | 503 |
| `STORAGE_ERROR` | Knowledge Graph storage failed | 500 |
| `VECTOR_ERROR` | Vector store operation failed | 500 |
| `INVALID_QUERY` | Malformed SPARQL query | 400 |
| `RATE_LIMIT_EXCEEDED` | Too many requests | 429 |
| `UNAUTHORIZED` | Invalid API key | 401 |
| `DOMAIN_NOT_FOUND` | Unknown domain configuration | 404 |

## Rate Limiting

Das System implementiert Rate Limiting für API-Stabilität:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642518000
```

**Standard Limits:**
- Document Processing: 100 requests/hour
- Queries: 1000 requests/hour
- Vector Search: 500 requests/hour

## WebSocket API (Advanced)

Für Real-time-Updates bei länger dauernden Verarbeitungsprozessen:

### WS /stream/processing

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/stream/processing');

ws.onmessage = function(event) {
    const update = JSON.parse(event.data);
    console.log('Processing update:', update);
};
```

**Message Types:**
```json
{
    "type": "progress",
    "data": {
        "document_id": "doc_123",
        "stage": "chunking",
        "progress": 0.45,
        "estimated_remaining": 120
    }
}
```

```json
{
    "type": "completed",
    "data": {
        "document_id": "doc_123",
        "triples_count": 42,
        "processing_time": 8.5
    }
}
```

## SDK und Client Libraries

### Python SDK

```python
from kg_pipeline_client import KGPipelineClient

client = KGPipelineClient(
    base_url="http://localhost:8000",
    api_key="your_api_key"
)

# Document processing
result = client.process_document(
    content="Apple Inc. is a technology company.",
    format="txt",
    domain="general"
)

# Query knowledge graph
results = client.query(
    question="What companies produce smartphones?",
    query_type="hybrid"
)

# Upload file
with open("document.pdf", "rb") as f:
    result = client.upload_document(f, domain="automotive")
```

### JavaScript SDK

```javascript
import { KGPipelineClient } from 'kg-pipeline-js';

const client = new KGPipelineClient({
    baseUrl: 'http://localhost:8000',
    apiKey: 'your_api_key'
});

// Process document
const result = await client.processDocument({
    content: 'Tesla produces electric vehicles.',
    format: 'txt',
    domain: 'automotive'
});

// Query
const queryResults = await client.query({
    question: 'Who founded Tesla?',
    queryType: 'hybrid'
});
```

## OpenAPI Specification

Die vollständige OpenAPI/Swagger-Spezifikation ist verfügbar unter:

```
GET /docs         # Swagger UI
GET /redoc        # ReDoc
GET /openapi.json # OpenAPI JSON Schema
```

## Beispiel-Workflows

### Workflow 1: Einzeldokument verarbeiten

```bash
# 1. Upload document
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@company_report.pdf" \
  -F "domain=business"

# 2. Query processed information
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main financial metrics?", "query_type": "hybrid"}'

# 3. Get detailed triples
curl -G http://localhost:8000/sparql \
  --data-urlencode 'query=SELECT ?metric ?value WHERE { ?company ?metric ?value }'
```

### Workflow 2: Batch-Verarbeitung

```bash
# 1. Batch upload
curl -X POST http://localhost:8000/documents/batch \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {"content": "Doc 1...", "format": "txt"},
      {"content": "Doc 2...", "format": "txt"}
    ],
    "domain": "research"
  }'

# 2. Monitor processing
curl http://localhost:8000/stats

# 3. Query aggregated knowledge
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are common themes across documents?"}'
```

### Workflow 3: Domain-Switch

```bash
# 1. Check current domain
curl http://localhost:8000/config

# 2. Switch to automotive domain
curl -X POST http://localhost:8000/config/domain \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "automotive",
    "ontology_path": "plugins/ontologies/automotive.ttl"
  }'

# 3. Process automotive documents
curl -X POST http://localhost:8000/documents/process \
  -H "Content-Type: application/json" \
  -d '{
    "content": "BMW produces luxury vehicles in Germany.",
    "format": "txt"
  }'
```

## Skalierung und Load Balancing

### GPU-bewusste Skalierungsstrategie

Das System verwendet eine GPU-bewusste Skalierungsstrategie, die Worker-Thread-Probleme bei GPU-Modellen vermeidet:

#### POST /admin/nodes/register

Registriert neuen Processing-Node im Cluster.

**Request Body:**
```json
{
    "node_id": "gpu_node_1",
    "node_type": "gpu",
    "capabilities": {
        "gpu_memory": "24GB",
        "cuda_version": "11.8",
        "supported_formats": ["pdf", "docx", "txt"]
    },
    "endpoint": "http://gpu-node-1:8001"
}
```

#### GET /admin/nodes/status

Gibt Status aller registrierten Nodes zurück.

**Response:**
```json
{
    "success": true,
    "data": {
        "nodes": [
            {
                "node_id": "gpu_node_1",
                "status": "active",
                "current_load": 3,
                "max_capacity": 10,
                "model_loaded": true,
                "processing_time_avg": 2.34
            },
            {
                "node_id": "cpu_node_1",
                "status": "active",
                "current_load": 8,
                "max_capacity": 20,
                "model_loaded": false,
                "processing_time_avg": 0.89
            }
        ],
        "total_capacity": 30,
        "current_utilization": 0.37
    }
}
```

### Batch-Verarbeitung mit Node-Distribution

Für große Batch-Verarbeitungen nutzt das System intelligente Node-Verteilung:

```python
# Beispiel Client-Code für skalierte Verarbeitung
from kg_pipeline_client import ScaledKGPipelineClient

client = ScaledKGPipelineClient(
    base_url="http://loadbalancer:8000",
    scaling_strategy="gpu_aware"
)

# Batch mit 1000 Dokumenten
large_batch = [...]

# Client verteilt automatisch auf verfügbare Nodes
# PDFs → GPU-Nodes (für SmolDocling)
# Andere → CPU-Nodes oder GPU-Nodes mit freier Kapazität
result = client.process_batch_distributed(
    documents=large_batch,
    options={
        "prefer_gpu_for": ["pdf"],
        "max_parallel_nodes": 5,
        "timeout_per_document": 60
    }
)

print(f"Processed on {result['nodes_used']} nodes")
print(f"Total time: {result['total_time']}s")
print(f"Documents/second: {result['throughput']}")
```

### Load Balancing Strategien

Das System unterstützt verschiedene Load-Balancing-Strategien:

1. **Round-Robin**: Gleichmäßige Verteilung
2. **Least-Loaded**: Bevorzugt Nodes mit weniger Last
3. **GPU-Aware**: PDFs zu GPU-Nodes, andere flexibel
4. **Capacity-Based**: Basierend auf Node-Kapazität

**Konfiguration:**
```yaml
# config/scaling.yaml
scaling:
  strategy: "gpu_aware"
  nodes:
    - id: "gpu_node_1"
      type: "gpu"
      weight: 2  # Kann doppelt so viele PDFs verarbeiten
    - id: "cpu_node_1"
      type: "cpu"
      weight: 1
  health_check_interval: 30
  node_timeout: 300
```

**Wichtige Hinweise zur GPU-Skalierung:**
- **Keine Worker-Threads**: Jeder GPU-Node läuft als separater Prozess/Container
- **Model-Caching**: Jeder Node lädt Modelle einmal und behält sie im Speicher
- **Sequenzielle Verarbeitung**: Innerhalb jedes Nodes werden Dokumente sequenziell verarbeitet
- **Message Queue**: Koordination zwischen Nodes über Redis/RabbitMQ

## Testing und Development

### Test-Endpunkte

```bash
# Health check
curl http://localhost:8000/health

# Test processing with sample data
curl -X POST http://localhost:8000/documents/process \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Test document content for processing.",
    "format": "txt",
    "domain": "general"
  }'

# Test query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "test query", "k": 1}'
```

Diese API-Dokumentation bietet eine vollständige Referenz für die Integration und Nutzung des generischen Knowledge Graph Pipeline Systems. Das System ist darauf ausgelegt, sowohl einfache Einzeldokument-Verarbeitungen als auch komplexe Batch-Operationen und erweiterte Query-Szenarien zu unterstützen.