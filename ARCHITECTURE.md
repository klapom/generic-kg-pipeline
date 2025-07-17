# Generic Knowledge Graph Pipeline - Architecture Documentation

## 1. High-Level System Architecture

```mermaid
graph TB
    subgraph "External Services"
        FS[Apache Fuseki<br/>Triple Store]
        CD[ChromaDB<br/>Vector Store]
        HLL[Hochschul-LLM<br/>GPU 2]
        VLLM[vLLM Service<br/>GPU 1]
    end
    
    subgraph "API Layer"
        API[FastAPI<br/>Main Application]
        HR[Health Router]
        DR[Documents Router]
        PR[Pipeline Router]
        QR[Query Router]
        CR[Config Router]
        
        API --> HR
        API --> DR
        API --> PR
        API --> QR
        API --> CR
    end
    
    subgraph "Core Processing"
        PF[Parser Factory]
        BP[Batch Processor]
        CH[Chunking System]
        TE[Triple Extractor]
        VM[VLM Manager]
    end
    
    subgraph "Document Parsers"
        PDF[PDF Parser<br/>Hybrid]
        OFF[Office Parser<br/>DOCX/XLSX/PPTX]
        TXT[Text Parser]
    end
    
    subgraph "Storage"
        DOC[Document Store]
        VEC[Vector Store]
        TRI[Triple Store]
    end
    
    DR --> PF
    PF --> PDF
    PF --> OFF
    PF --> TXT
    
    PDF --> VLLM
    PDF --> VM
    
    BP --> PF
    BP --> CH
    CH --> TE
    TE --> HLL
    
    TE --> TRI
    TE --> FS
    CH --> VEC
    CH --> CD
    
    QR --> FS
    QR --> CD
```

## 2. Document Processing Pipeline

```mermaid
graph LR
    subgraph "Input"
        D[Document<br/>PDF/DOCX/TXT]
    end
    
    subgraph "Parser Selection"
        PF[Parser Factory]
        DT{Document<br/>Type?}
        
        PF --> DT
    end
    
    subgraph "Content Extraction"
        PDF[HybridPDFParser]
        SMOL[SmolDocling<br/>Client]
        PLU[PDFPlumber<br/>Fallback]
        OFF[Office Parser]
        TXT[Text Parser]
        
        PDF --> SMOL
        PDF --> PLU
    end
    
    subgraph "Visual Analysis"
        VLM[VLM Integration]
        IMG[Image Analysis]
        TAB[Table Analysis]
        
        VLM --> IMG
        VLM --> TAB
    end
    
    subgraph "Structuring"
        SEG[Segmentation]
        CHK[Chunking]
        CTX[Context<br/>Enrichment]
        
        SEG --> CHK
        CHK --> CTX
    end
    
    subgraph "Knowledge Extraction"
        TE[Triple<br/>Extraction]
        EMB[Embeddings]
        
        CTX --> TE
        CTX --> EMB
    end
    
    subgraph "Storage"
        FS[(Fuseki<br/>RDF Store)]
        CD[(ChromaDB<br/>Vectors)]
        
        TE --> FS
        EMB --> CD
    end
    
    D --> PF
    DT -->|PDF| PDF
    DT -->|Office| OFF
    DT -->|Text| TXT
    
    SMOL --> SEG
    PLU --> SEG
    OFF --> SEG
    TXT --> SEG
    
    PDF --> VLM
```

## 3. Client Architecture

```mermaid
classDiagram
    class BaseModelClient {
        <<abstract>>
        +endpoint: str
        +metrics: ClientMetrics
        +process(request)
        +health_check()
        +get_metrics()
        #_process_internal()
        #_health_check_internal()
    }
    
    class BaseVLLMClient {
        <<abstract>>
        +model_id: str
        +model_manager: VLLMModelManager
        +parse_model_output()
        +ensure_model_loaded()
    }
    
    class VLLMSmolDoclingFinalClient {
        +use_docling: bool
        +environment: str
        +parse_pdf(pdf_path)
        +_parse_with_docling()
        +_extract_visuals_direct()
    }
    
    class HochschulLLMClient {
        +extract_triples()
        +generate_context()
        +validate_response()
    }
    
    class TransformersClient {
        <<abstract>>
        +model_name: str
        +load_model()
        +process_image()
    }
    
    BaseModelClient <|-- BaseVLLMClient
    BaseModelClient <|-- HochschulLLMClient
    BaseModelClient <|-- TransformersClient
    BaseVLLMClient <|-- VLLMSmolDoclingFinalClient
    
    TransformersClient <|-- TransformersQwen25VLClient
    TransformersClient <|-- TransformersLLaVAClient
    TransformersClient <|-- TransformersPixtralClient
```

## 4. Configuration System

```mermaid
graph TB
    subgraph "Configuration Sources"
        YAML[config.yaml]
        ENV[Environment<br/>Variables]
        CLI[CLI Arguments]
    end
    
    subgraph "Configuration Manager"
        UM[UnifiedConfig<br/>Manager]
        HR[Hot Reload<br/>Watcher]
        VAL[Validation<br/>Pydantic]
        
        UM --> HR
        UM --> VAL
    end
    
    subgraph "Configuration Domains"
        SVC[Services Config]
        MDL[Models Config]
        PRS[Parsing Config]
        STG[Storage Config]
        BTH[Batch Config]
        CHK[Chunking Config]
    end
    
    subgraph "Consumers"
        API[API Layer]
        CLI[CLI Tools]
        CL[Clients]
        PS[Parsers]
        BP[Processors]
    end
    
    YAML --> UM
    ENV --> UM
    CLI --> UM
    
    UM --> SVC
    UM --> MDL
    UM --> PRS
    UM --> STG
    UM --> BTH
    UM --> CHK
    
    SVC --> API
    SVC --> CL
    MDL --> CL
    PRS --> PS
    BTH --> BP
```

## 5. Parser System Architecture

```mermaid
graph TD
    subgraph "Parser Factory"
        PF[ParserFactory]
        REG[Parser Registry]
        SEL[Parser Selector]
        
        PF --> REG
        PF --> SEL
    end
    
    subgraph "Base Components"
        BP[BaseParser]
        DM[Data Models]
        INT[Interfaces]
        
        BP --> DM
        BP --> INT
    end
    
    subgraph "PDF Processing"
        HPP[HybridPDFParser]
        SMOL[SmolDocling<br/>Integration]
        PLU[PDFPlumber<br/>Extractor]
        VLM[VLM Analysis]
        
        HPP --> SMOL
        HPP --> PLU
        HPP --> VLM
    end
    
    subgraph "Office Processing"
        DOCX[DOCXParser]
        XLSX[XLSXParser]
        PPTX[PPTXParser]
        
        DOCX --> python-docx
        XLSX --> openpyxl
        PPTX --> python-pptx
    end
    
    subgraph "Output"
        DOC[Document]
        SEG[Segments]
        VIS[Visual Elements]
        MET[Metadata]
        
        DOC --> SEG
        DOC --> VIS
        DOC --> MET
    end
    
    PF --> HPP
    PF --> DOCX
    PF --> XLSX
    PF --> PPTX
    
    BP --> HPP
    BP --> DOCX
    BP --> XLSX
    BP --> PPTX
    
    HPP --> DOC
    DOCX --> DOC
    XLSX --> DOC
    PPTX --> DOC
```

## 6. VLM Processing Pipeline

```mermaid
graph TB
    subgraph "Model Management"
        MM[Model Manager]
        ML[Model Loader]
        MC[Model Cache]
        MU[Model Unloader]
        
        MM --> ML
        MM --> MC
        MM --> MU
    end
    
    subgraph "Processing Stages"
        CLS[Document<br/>Classifier]
        TSP[Two-Stage<br/>Processor]
        CE[Confidence<br/>Evaluator]
        
        CLS --> TSP
        TSP --> CE
    end
    
    subgraph "Visual Analysis"
        IMG[Image<br/>Extraction]
        OCR[OCR<br/>Processing]
        TAB[Table<br/>Detection]
        FIG[Figure<br/>Analysis]
        
        TSP --> IMG
        TSP --> OCR
        TSP --> TAB
        TSP --> FIG
    end
    
    subgraph "Results"
        VE[Visual<br/>Elements]
        TXT[Extracted<br/>Text]
        STR[Structure<br/>Info]
        
        CE --> VE
        CE --> TXT
        CE --> STR
    end
    
    MM --> CLS
    IMG --> VE
    OCR --> TXT
    TAB --> STR
    FIG --> VE
```

## 7. Data Flow Sequence

```mermaid
sequenceDiagram
    participant U as User
    participant API as API Layer
    participant PF as Parser Factory
    participant P as Parser
    participant C as Chunker
    participant TE as Triple Extractor
    participant S as Storage
    
    U->>API: Upload Document
    API->>PF: Create Parser
    PF->>P: Parse Document
    P->>P: Extract Content
    P->>P: Extract Visuals
    P-->>API: Document Object
    
    API->>C: Chunk Document
    C->>C: Create Segments
    C->>C: Add Context
    C-->>API: Chunks
    
    API->>TE: Extract Triples
    TE->>TE: Process with LLM
    TE-->>API: Triples
    
    API->>S: Store Results
    S->>S: Save to Fuseki
    S->>S: Save to ChromaDB
    S-->>API: Success
    
    API-->>U: Processing Complete
```

## 8. Environment Variable Configuration

```mermaid
graph LR
    subgraph "Docling Features"
        USE[USE_DOCLING]
        EXT[EXTRACT_IMAGES_DIRECTLY]
        ROLL[DOCLING_ROLLOUT_PERCENTAGE]
    end
    
    subgraph "Image Settings"
        MAX[MAX_IMAGE_SIZE]
        QUAL[IMAGE_QUALITY]
        TAB[EXTRACT_TABLES]
        FORM[EXTRACT_FORMULAS]
    end
    
    subgraph "Performance"
        LOG[LOG_PERFORMANCE]
        THR[PERFORMANCE_THRESHOLD_SECONDS]
        PDF[MAX_PDF_SIZE_MB]
        BAT[MAX_PAGES_BATCH]
    end
    
    subgraph "Error Handling"
        CONT[CONTINUE_ON_ERROR]
    end
    
    subgraph "Application"
        APP[VLLMSmolDoclingFinalClient]
        CFG[docling_config.py]
        
        CFG --> APP
    end
    
    USE --> CFG
    EXT --> CFG
    ROLL --> CFG
    MAX --> CFG
    QUAL --> CFG
    TAB --> CFG
    FORM --> CFG
    LOG --> CFG
    THR --> CFG
    PDF --> CFG
    BAT --> CFG
    CONT --> CFG
```

## Key Architecture Principles

1. **Separation of Concerns**: Clear boundaries between parsing, processing, and storage
2. **Extensibility**: Plugin architecture for parsers and processors
3. **Scalability**: Async processing, separate GPU workloads
4. **Resilience**: Fallback mechanisms, retry logic, error handling
5. **Configurability**: Environment-based configuration with hot reload
6. **Observability**: Comprehensive logging, metrics, and health checks

## Technology Stack

- **Framework**: FastAPI (async Python)
- **Document Processing**: PyMuPDF, PDFPlumber, python-docx
- **VLM**: vLLM, Transformers, SmolDocling
- **Storage**: Apache Fuseki (RDF), ChromaDB (Vectors)
- **Configuration**: Pydantic, YAML
- **Monitoring**: Custom metrics, health endpoints