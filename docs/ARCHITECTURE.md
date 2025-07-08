# Generic Knowledge Graph Pipeline Architecture

## Overview

The Generic Knowledge Graph Pipeline System is designed with a clear separation of concerns, particularly regarding GPU workloads. The system uses two distinct GPU-accelerated services:

1. **vLLM with SmolDocling** - Advanced PDF parsing and document understanding
2. **Hochschul-LLM (Qwen1.5-based)** - Knowledge graph triple extraction

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Client Applications                                │
│                        (Web UI, CLI, API Clients)                           │
└─────────────────────────────────────────┬───────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FastAPI Application Layer                           │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────────┐   │
│  │ Document Upload │  │ Pipeline Control  │  │ Query & Export API      │   │
│  │     Endpoint    │  │    Endpoints      │  │     Endpoints           │   │
│  └─────────────────┘  └──────────────────┘  └─────────────────────────┘   │
└─────────────────────────────────────────┬───────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Core Pipeline Engine                               │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────────┐   │
│  │ Plugin Manager  │  │ Workflow Engine   │  │  RAG Pipeline Manager   │   │
│  └─────────────────┘  └──────────────────┘  └─────────────────────────┘   │
└─────────────────────────────────────────┬───────────────────────────────────┘
                                         │
           ┌────────────────────┬────────┴────────┬────────────────────┐
           ▼                    ▼                  ▼                    ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌──────────┐
│ Document Parsers │  │  Text Chunking   │  │ Triple Extraction│  │  Storage │
│                  │  │    Service       │  │     Service      │  │ Services │
├──────────────────┤  └──────────────────┘  ├──────────────────┤  ├──────────┤
│ • PDF Parser     │                         │ • Prompt Engine  │  │ • Fuseki │
│ • DOCX Parser    │                         │ • Result Parser  │  │ • ChromaDB│
│ • XLSX Parser    │                         │ • Validation     │  └──────────┘
│ • Text Parser    │                         └──────────────────┘
└────────┬─────────┘                                  │
         │                                            │
         ▼                                            ▼
┌──────────────────┐                         ┌──────────────────┐
│ vLLM SmolDocling │                         │  Hochschul-LLM   │
│    (GPU 1)       │                         │    (GPU 2)       │
│                  │                         │                  │
│ • PDF Processing │                         │ • Qwen1.5-72B    │
│ • Table Extract. │                         │ • Triple Extract.│
│ • Image Analysis │                         │ • Semantic Under.│
└──────────────────┘                         └──────────────────┘
```

## Component Details

### 1. Document Processing Pipeline

The document processing pipeline handles various document formats with specialized parsers:

#### PDF Processing (GPU Workload 1)
- **Service**: vLLM with SmolDocling
- **Purpose**: Advanced PDF parsing including complex layouts, tables, and images
- **GPU Usage**: Dedicated GPU for high-throughput document understanding
- **Features**:
  - Layout analysis
  - Table extraction with structure preservation
  - Image caption generation
  - Multi-column text extraction

#### Office Document Processing
- **Service**: Native Python libraries
- **Purpose**: Extract content from DOCX, XLSX files
- **GPU Usage**: None (CPU-based)

### 2. Triple Extraction Pipeline

#### Knowledge Extraction (GPU Workload 2)
- **Service**: Hochschul-LLM (Qwen1.5-based)
- **Purpose**: Extract semantic triples from document chunks
- **GPU Usage**: External high-performance LLM infrastructure
- **Features**:
  - Domain-specific prompt engineering
  - Multi-hop reasoning
  - Relationship validation
  - Confidence scoring

### 3. Storage Layer

#### Triple Store (Fuseki)
- **Purpose**: Store and query RDF triples
- **Technology**: Apache Jena Fuseki
- **Features**:
  - SPARQL endpoint
  - Named graphs support
  - Reasoning capabilities

#### Vector Store (ChromaDB)
- **Purpose**: Store document embeddings for RAG
- **Technology**: ChromaDB
- **Features**:
  - Similarity search
  - Metadata filtering
  - Persistent storage

## Data Flow

1. **Document Upload**
   ```
   Client → API → Document Parser Selection → Format-specific Parser
   ```

2. **PDF Processing Flow**
   ```
   PDF File → vLLM SmolDocling → Structured Content → Chunking Service
   ```

3. **Triple Extraction Flow**
   ```
   Document Chunks → RAG Context → Hochschul-LLM → Triple Validation → Fuseki
   ```

4. **Query Flow**
   ```
   Query → ChromaDB Similarity Search → Context Retrieval → SPARQL Generation → Results
   ```

## Performance Considerations

### GPU Workload Separation

The architecture deliberately separates GPU workloads to optimize performance:

1. **vLLM SmolDocling (GPU 1)**
   - Optimized for batch processing
   - High memory usage for model loading
   - Continuous operation for document queue

2. **Hochschul-LLM (GPU 2)**
   - Request-based processing
   - Lower latency requirements
   - Higher computational requirements per request

### Scaling Strategies

1. **Horizontal Scaling**
   - Multiple API instances behind load balancer
   - Separate scaling for parsing and extraction services

2. **Vertical Scaling**
   - GPU memory allocation tuning
   - Batch size optimization

3. **Queue-based Processing**
   - Asynchronous document processing
   - Priority queues for different document types

## Plugin Architecture

The system supports extensibility through plugins:

### Parser Plugins
```python
class BaseParser(ABC):
    @abstractmethod
    def parse(self, file_path: Path) -> Document:
        pass
```

### Ontology Plugins
- Domain-specific ontologies
- Custom relationship types
- Validation rules

### Template Plugins
- Prompt templates for different domains
- Output format templates

## Security Considerations

1. **API Security**
   - API key authentication for Hochschul-LLM
   - Rate limiting per client
   - Input validation

2. **Data Security**
   - Encrypted storage for sensitive documents
   - Secure communication with external services
   - Audit logging

3. **Resource Protection**
   - GPU memory limits
   - Request timeouts
   - DOS protection

## Monitoring and Observability

1. **Metrics**
   - Document processing throughput
   - GPU utilization
   - API response times
   - Error rates

2. **Logging**
   - Structured logging with correlation IDs
   - Error tracking
   - Performance profiling

3. **Health Checks**
   - Service availability
   - GPU health
   - Storage connectivity

## Development and Deployment

### Local Development
```bash
# Start with docker-compose
docker-compose -f docker/docker-compose.yml up

# Or run services individually
python -m uvicorn api.main:app --reload
```

### Production Deployment
- Kubernetes manifests for orchestration
- Helm charts for configuration management
- CI/CD pipeline with automated testing

## Future Enhancements

1. **Additional LLM Providers**
   - Support for multiple LLM backends
   - Dynamic provider selection

2. **Enhanced Caching**
   - Result caching for common queries
   - Embedding cache optimization

3. **Real-time Processing**
   - WebSocket support for live updates
   - Streaming triple extraction