# Generic Knowledge Graph Pipeline System

![Build Status](https://img.shields.io/github/actions/workflow/status/klapom/generic-kg-pipeline/ci.yml?branch=main)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)

A flexible, plugin-based pipeline system for extracting knowledge graphs from various document formats. The system uses vLLM with SmolDocling for advanced PDF parsing and a Hochschul-LLM (Qwen1.5-based) for triple extraction.

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

### GPU Workload 2: Triple Extraction
- **Hochschul-LLM (Qwen1.5-based)**: Knowledge graph triple extraction
- External high-performance LLM service
- Optimized for semantic understanding and relationship extraction

### Core Components
```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Document Input │────▶│  vLLM SmolDocling│────▶│  Chunking       │
│  (PDF, DOCX,    │     │  (GPU 1)         │     │  & Processing   │
│   XLSX, TXT)    │     └──────────────────┘     └────────┬────────┘
└─────────────────┘                                        │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Triple Store   │◀────│  Hochschul-LLM   │◀────│  RAG Pipeline   │
│  (Fuseki)       │     │  (GPU 2)         │     │  (ChromaDB)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- Docker and Docker Compose
- NVIDIA GPU with CUDA support (for vLLM)
- Access to Hochschul-LLM API

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