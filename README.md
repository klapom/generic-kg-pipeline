# Generic Knowledge Graph Pipeline System

![Build Status](https://img.shields.io/github/actions/workflow/status/klapom/generic-kg-pipeline/ci.yml?branch=main)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)

A flexible, plugin-based pipeline system for extracting knowledge graphs from various document formats. The system uses vLLM with SmolDocling for advanced PDF parsing, Qwen2.5-VL for visual analysis, and a Hochschul-LLM (Qwen1.5-based) for triple extraction with **Context Inheritance** chunking strategy.

## 🎯 Current Implementation Status

### ✅ COMPLETED (MVP Phase)
- **Multi-Modal Document Parser System (100%)**
  - PDF Parser with vLLM SmolDocling integration
  - DOCX Parser with image extraction
  - XLSX Parser with chart analysis
  - PPTX Parser with slide visuals
  - Parser Factory for automatic format detection
  - Context mapping for precise text-image relationships

- **Content Chunking with Context Inheritance (100%)**
  - Structure-aware chunking based on document structure
  - Context group formation (PDF sections, DOCX headings, XLSX sheets, PPTX topics)
  - LLM-based context summary generation
  - Dual-task prompting for optimal context inheritance
  - Async processing for performance

- **LLM Client Infrastructure (100%)**
  - vLLM SmolDocling client for PDF parsing
  - Hochschul-LLM client for triple extraction
  - Qwen2.5-VL client for visual analysis
  - OpenAI-compatible API integration

- **FastAPI Application (80%)**
  - Health, Documents, Pipeline, Query endpoints
  - Multi-modal upload support
  - Batch processing integration

- **Batch Processing System (100%)**
  - Concurrent document processing
  - Progress tracking and error handling
  - Filesystem-based processing

### 🔄 IN DEVELOPMENT
- Triple Store Integration (Fuseki)
- Vector Store Integration (ChromaDB)
- End-to-End Pipeline Integration

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/klapom/generic-kg-pipeline.git
cd generic-kg-pipeline

# Copy environment configuration
cp .env.example .env
# Edit .env with your Hochschul-LLM credentials

# Start all services with Docker
docker-compose -f docker/docker-compose.yml up -d

# Or install locally
pip install -r requirements.txt
python -m uvicorn api.main:app --reload
```

## 📋 Architecture Overview

The system separates GPU workloads for optimal performance:

### GPU Workload 1: Document Parsing
- **vLLM with SmolDocling**: Advanced PDF parsing and document understanding
- Handles complex document layouts, tables, and images
- Runs on dedicated GPU for maximum throughput

### External API: Triple Extraction
- **Hochschul-LLM (Qwen1.5-based)**: Knowledge graph triple extraction
- External high-performance LLM service via OpenAI-compatible API
- Optimized for semantic understanding and relationship extraction

### Visual Analysis
- **Qwen2.5-VL**: Multi-modal visual element analysis
- Processes embedded diagrams, charts, and images
- Integrated into document parsing pipeline

### Core Components
```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Document Input │────▶│  vLLM SmolDocling│────▶│  Context        │
│  (PDF, DOCX,    │     │  (Local GPU)     │     │  Inheritance    │
│   XLSX, PPTX)   │     │  + Qwen2.5-VL    │     │  Chunking       │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Triple Store   │◀────│  Hochschul-LLM   │◀────│  RAG Pipeline   │
│  (Fuseki)       │     │  (External API)  │     │  (ChromaDB)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- Docker and Docker Compose
- NVIDIA GPU with CUDA support (for vLLM SmolDocling)
- Access to Hochschul-LLM API (external service)
- Access to Qwen2.5-VL API (for visual analysis)

### Local Development
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Start development server
python -m uvicorn api.main:app --reload
```

### Docker Deployment
```bash
# Build and start all services
make docker-up

# View logs
docker-compose -f docker/docker-compose.yml logs -f

# Stop services
make docker-down
```

## 📚 Documentation

- [Architecture Details](docs/ARCHITECTURE.md)
- [API Documentation](docs/API.md)
- [Plugin Development](docs/PLUGINS.md)
- [Configuration Guide](docs/CONFIGURATION.md)

## 🧪 Testing

```bash
# Run all tests
make test

# Run with coverage
pytest --cov=core --cov=plugins tests/

# Linting
make lint

# Format code
make format
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- SmolDocling for advanced document parsing capabilities
- Hochschule team for providing the high-performance LLM infrastructure
- Open source community for the amazing tools and libraries