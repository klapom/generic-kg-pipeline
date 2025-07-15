Projektanweisung
Projektziel
Entwicklung eines domänen-agnostischen Document-to-Knowledge-Graph Systems mit 70% weniger Code als das bestehende Automotive-System, das RAG-Technologie integriert und horizontal skalierbar ist.
Kernvorgaben
Technische Anforderungen:

Sequenzielle GPU-Verarbeitung (keine Worker-Threads)
Plugin-basierte Architektur für Domänen/Formate
Hybrid RAG+KG mit ChromaDB und Fuseki
Configuration-First Ansatz (YAML)
15 Kernmodule statt 50+

Phasenplan (4-5 Wochen):

MVP (1-2 Wochen): Core-Pipeline, Basic Parser, LLM-Integration
RAG (1 Woche): Vector Store, Hybrid Queries, Context-Enrichment
Production (2 Wochen): API, Monitoring, Docker, Skalierung

Deliverables:

Funktionierende Pipeline mit REST API
Docker-basiertes Deployment
Plugin-System für neue Domänen
Performance-Monitoring
Dokumentation

Erfolgskriterien:

70%+ Performance-Steigerung bei Batch-Verarbeitung
Domänen-Wechsel ohne Code-Änderung
Horizontale Skalierung auf GPU-Nodes
<5s Query-Response für Hybrid-Suchen

Nicht im Scope:

Migration bestehender Automotive-Daten
Multi-Tenancy Features
Advanced Security/Compliance

Implementierungsrichtlinien
Arbeitsweise:

Schritt für Schritt denken
Komplexe Probleme in kleinere Probleme zerlegen
Immer im Internet nach Best Practices oder aktuellen Papers für eine Problemstellung suchen
Konzepte zunächst abstimmen, bevor Implementierung erfolgt

Technische Richtlinien:

Folge dem Implementierungsleitfaden und Best Practices Guide
Vermeide bekannte Anti-Patterns (Worker-Threads, Über-Abstraktion)
Configuration-First vor Code-First Ansatz

Aktueller Implementierungsstand
✅ VOLLSTÄNDIG IMPLEMENTIERT (MVP-Phase):

Multi-Modal Document Parser System (100%)
- PDF Parser mit vLLM SmolDocling Integration
- DOCX Parser mit Image Extraction
- XLSX Parser mit Chart Analysis
- PPTX Parser mit Slide Visuals
- Parser Factory für automatische Format-Erkennung
- Context Mapping für präzise Text-Bild-Zuordnung

Content Chunking mit Context Inheritance (100%)
- Structure-Aware Chunking basierend auf Dokumentstruktur
- Context Group Formation (PDF Sections, DOCX Headings, XLSX Sheets, PPTX Topics)
- LLM-basierte Context Summary Generation
- Dual-Task Prompting für optimale Kontext-Vererbung
- Async Processing für Performance

LLM Client Infrastructure (100%)
- vLLM SmolDocling Client für PDF-Parsing
- Hochschul-LLM Client für Triple Extraction
- Qwen2.5-VL Client für Visual Analysis
- OpenAI-kompatible API Integration

FastAPI Application (80%)
- Health, Documents, Pipeline, Query Endpoints
- Multi-Modal Upload Support
- Batch Processing Integration

Batch Processing System (100%)
- Concurrent Document Processing
- Progress Tracking und Error Handling
- Filesystem-basierte Verarbeitung

🔄 IN ENTWICKLUNG:
- Triple Store Integration (Fuseki)
- Vector Store Integration (ChromaDB)
- End-to-End Pipeline Integration

Architektur-Übersicht
Implementierte Komponenten-Struktur
core/                           # Kernmodule (implementiert)
├── content_chunker.py         # ✅ Chunking mit Context Inheritance
├── batch_processor.py         # ✅ Batch-Verarbeitung
├── config.py                  # ✅ Unified Configuration
├── clients/                   # ✅ LLM Client Infrastructure
│   ├── vllm_smoldocling.py   # ✅ PDF-Parsing
│   ├── hochschul_llm.py      # ✅ Triple Extraction
│   └── qwen25_vl.py          # ✅ Visual Analysis
└── chunking/                  # ✅ Chunking System
    ├── chunk_models.py        # ✅ Enhanced Data Models
    ├── context_grouper.py     # ✅ Context Group Formation
    ├── context_summarizer.py  # ✅ LLM-basierte Context Generation
    └── base_chunker.py        # ✅ Base Chunking Logic

plugins/                        # Plugin-Module (implementiert)
├── parsers/                   # ✅ Multi-Modal Parser
│   ├── pdf_parser.py         # ✅ vLLM SmolDocling Integration
│   ├── docx_parser.py        # ✅ Image Extraction
│   ├── xlsx_parser.py        # ✅ Chart Analysis
│   ├── pptx_parser.py        # ✅ Slide Visuals
│   ├── parser_factory.py     # ✅ Automatic Format Selection
│   └── context_mapping.py    # ✅ Enhanced Context Mapping
├── ontologies/                # Domain-Ontologien
└── templates/                 # ✅ LLM-Prompt-Templates
    ├── context_generation.txt # ✅ Context Inheritance Prompts
    └── task_with_context.txt  # ✅ Task Integration Prompts

api/                           # FastAPI Application (implementiert)
├── main.py                    # ✅ Hauptanwendung
└── routers/                   # ✅ API Endpoints
    ├── health.py             # ✅ Health Checks
    ├── documents.py          # ✅ Document Upload
    ├── pipeline.py           # ✅ Pipeline Processing
    └── query.py              # ✅ Query Interface

config/                        # Configuration (implementiert)
├── default.yaml              # ✅ System Configuration
└── chunking.yaml             # ✅ Chunking Configuration
Datenfluss
Input Documents → Document Parser → Content Chunker → RAG Processor
                                                           │
                                                           v
Vector Embedder ← Context Enricher ← LLM Client ← Triple Extractor
       │                                              │
       v                                              v
Vector Store ← Similarity Validator → Triple Store ← RDF Writer
Kritische Lessons Learned
✅ Best Practices (Übernehmen)

Sequential Processing mit Model-Caching statt Worker-Threads
Context-Preserving Chunking für bessere LLM-Performance
Multi-Level Quality Validation für Triples
Domain-adaptive Prompt Engineering
Graceful Degradation und Fallback-Strategien

❌ Anti-Patterns (Vermeiden)

Worker-Thread-Architektur bei GPU-Modellen
Hybrid Live/Batch System (unnötige Komplexität)
Über-abstrahierte Router-Hierarchie
Inkonsistente Datenstrukturen
Premature Optimization

Performance-Optimierungen
Model-Caching
python# ✅ Empfohlen: Einmaliges Model-Loading
def process_sequential_cached(documents):
    switch_to_optimized_smoldocling()  # Einmal laden
    results = []
    for doc in documents:
        result = process_file(doc)  # Model wird wiederverwendet
        results.append(result)
    cleanup_smoldocling_model()
    return results
GPU-bewusste Skalierung

Separate Nodes statt Worker-Threads
Ein GPU-Model pro Node statt Model pro Thread
Sequenzielle Verarbeitung pro Node
Message Queue zwischen Nodes statt Shared Memory

Token-Limit-Management
Optimale Chunking-Konfiguration
yamlpdf_documents:
  max_tokens: 2000
  overlap_ratio: 0.2
  preserve_structure: true

office_documents:
  max_tokens: 1500
  overlap_ratio: 0.15
  preserve_tables: true

web_content:
  max_tokens: 1000
  overlap_ratio: 0.3
  clean_html: true
RAG-Integration
Hybrid-Ansatz
Vector Store ←→ Knowledge Graph
     ↑               ↑
Semantic Search ←→ SPARQL Queries
     ↓               ↓
  Similarity ←→ Structured Facts
Implementierungsstrategie

ChromaDB für Vector Storage
Hybrid Queries (Vector + SPARQL)
Context-Enrichment für LLM
Advanced Similarity-basierte Validation

Entwicklungshinweise
Phase 1: MVP

Core Pipeline mit Document Parser, Content Chunker, LLM Client, KG Store
Basic Text/PDF Parsing (ohne SmolDocling zunächst)
Ollama LLM Integration mit Qwen:7b
Fuseki Triple Store Setup
CLI Interface

Phase 2: RAG-Integration

ChromaDB Vector Store Integration
Semantic Similarity Search
Hybrid Query Engine
Context-Enrichment für Triple-Extraktion
Quality Validation mit Similarity

Phase 3: Production Features

FastAPI REST API
Domain Plugin System
SmolDocling PDF Parser
Performance Monitoring
Docker Deployment
Horizontale Skalierung

Konfigurationsbeispiel
yamldomain:
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

storage:
  triple_store_url: "http://localhost:3030"
  vector_store_path: "data/vectors"