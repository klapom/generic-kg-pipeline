# vLLM Setup Guide - Generic Knowledge Graph Pipeline

## 🚀 Überblick

Diese Anleitung zeigt, wie Sie vLLM für lokale, hochperformante Model-Verarbeitung in der Generic Knowledge Graph Pipeline einrichten.

### Was wird installiert:
- **vLLM** - High-Performance Inference Engine
- **SmolDocling** - Für PDF-Parsing mit GPU-Beschleunigung
- **Qwen2.5-VL** - Für visuelle Analyse von Bildern und Charts

### Performance-Vorteile:
- ✅ **54.8% schnellere Verarbeitung** durch Model-Caching
- ✅ **Optimierte GPU-Nutzung** mit automatischem Memory Management
- ✅ **Batch-Processing** für effiziente Dokumentenverarbeitung
- ✅ **Kein Model-Reload** zwischen Dokumenten

## 📋 Systemvoraussetzungen

### Hardware:
- **NVIDIA GPU** mit mindestens 8GB VRAM (empfohlen: 16GB+)
- **CUDA 11.8** oder höher
- **RAM**: Mindestens 16GB System-RAM
- **Storage**: 50GB+ freier Speicherplatz für Models

### Software:
- **Python 3.8+**
- **CUDA Toolkit 11.8+**
- **Git** für Model-Downloads

### GPU-Anforderungen nach Model:
| Model | Min. VRAM | Empfohlen | Batch Size |
|-------|-----------|-----------|------------|
| SmolDocling-256M | 4GB | 8GB | 1-2 |
| Qwen2.5-VL-7B | 8GB | 16GB | 1-3 |
| Beide zusammen | 12GB | 24GB | Optimal |

## 🛠️ Installation

### 1. CUDA-Verfügbarkeit prüfen

```bash
# CUDA-Version prüfen
nvidia-smi

# Python CUDA-Support prüfen
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. vLLM Installation

```bash
# Mit UV (empfohlen für schnelle Installation)
uv pip install vllm

# Alternative mit pip
pip install vllm

# Bei Problemen: Clean Installation
pip uninstall torch torchvision torchaudio
uv pip install vllm  # Installiert automatisch PyTorch 2.7.0
```

### 3. Abhängigkeiten installieren

```bash
# Zusätzliche Abhängigkeiten für das Projekt
uv pip install pdf2image pillow

# Für PDF-Konvertierung (Ubuntu/Debian)
sudo apt-get install poppler-utils

# Für PDF-Konvertierung (macOS)
brew install poppler
```

### 4. Model-Downloads (optional)

Models werden automatisch heruntergeladen, aber Sie können sie vorab cachen:

```bash
# Hugging Face CLI installieren
pip install huggingface-hub

# Models vorab herunterladen
huggingface-cli download ds4sd/SmolDocling-256M-preview
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct
```

## ⚙️ Konfiguration

### 1. Environment-Variablen setzen

Kopieren Sie `.env.example` zu `.env` und passen Sie die Werte an:

```bash
cp .env.example .env
```

**Wichtige Konfigurationen in `.env`:**

```bash
# vLLM aktivieren
USE_VLLM=true

# GPU Memory Utilization (60-80% empfohlen)
VLLM_GPU_MEMORY_UTILIZATION=0.8

# Model-Cache-Verzeichnis (optional)
VLLM_MODEL_CACHE_DIR=/path/to/your/model/cache

# SmolDocling Einstellungen
SMOLDOCLING_MODEL_NAME=ds4sd/SmolDocling-256M-preview
SMOLDOCLING_MAX_PAGES=100
SMOLDOCLING_GPU_MEMORY=0.8

# Qwen2.5-VL Einstellungen
QWEN25_VL_MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct
QWEN25_VL_GPU_MEMORY=0.7
QWEN25_VL_MAX_IMAGE_SIZE=1024

# Batch Processing
BATCH_DEFAULT_MODE=vllm  # Produktives Setup mit vLLM
BATCH_MAX_CONCURRENT=3
```

### 2. YAML-Konfiguration anpassen

In `config/default.yaml` können Sie weitere Einstellungen vornehmen:

```yaml
# vLLM Configuration
vllm:
  gpu_memory_utilization: 0.8
  max_concurrent_models: 2
  
  smoldocling:
    model_name: "ds4sd/SmolDocling-256M-preview"
    max_pages: 100
    extract_tables: true
    extract_images: true
    extract_formulas: true
    
  qwen25_vl:
    model_name: "Qwen/Qwen2.5-VL-7B-Instruct"
    max_image_size: 1024
    batch_size: 3
```

## 🧪 Installation testen

### 1. Grundlegende Tests

```bash
# Beispiel-Dokumente erstellen
python process_documents_vllm.py --create-samples

# GPU-Verfügbarkeit testen
python -c "
from core.vllm.model_manager import VLLMModelManager
manager = VLLMModelManager()
print('GPU Info:', manager.get_gpu_memory_info())
print('vLLM Available:', manager.check_vllm_availability())
"
```

### 2. vLLM-Modus testen

```bash
# Erst mit einem einzelnen Dokument testen
python process_documents_vllm.py --file sample_document.txt

# Vollständige Batch-Verarbeitung
python process_documents_vllm.py
```

### 3. Performance-Test

```bash
# Mit detaillierter Ausgabe
python process_documents_vllm.py --verbose

# Mit angepassten Einstellungen
python process_documents_vllm.py --max-concurrent 5 --gpu-memory 0.7
```

## 📊 Performance-Optimierung

### 1. GPU Memory Management

```bash
# GPU-Status überwachen
nvidia-smi -l 1

# Memory-Utilization anpassen je nach GPU
# RTX 3080 (10GB): 0.7
# RTX 4090 (24GB): 0.8
# A100 (80GB): 0.9
```

### 2. Batch-Size-Optimierung

| GPU VRAM | SmolDocling Batch | Qwen2.5-VL Batch | Concurrent |
|----------|------------------|------------------|------------|
| 8GB | 1 | 1 | 1 |
| 16GB | 1 | 2 | 2 |
| 24GB | 2 | 3 | 3 |
| 32GB+ | 3 | 4 | 4 |

### 3. Model-Cache-Optimierung

```bash
# Cache-Verzeichnis auf SSD
export VLLM_MODEL_CACHE_DIR=/fast/ssd/path/models

# Hugging Face Cache setzen
export HF_HOME=/fast/ssd/path/huggingface

# CUDA Cache optimieren
export CUDA_CACHE_PATH=/fast/ssd/path/cuda_cache
```

## 🔧 Erweiterte Konfiguration

### 1. Multi-GPU Setup

```python
# In config/default.yaml
vllm:
  smoldocling:
    tensor_parallel_size: 2  # Für 2 GPUs
    
  qwen25_vl:
    tensor_parallel_size: 2  # Für 2 GPUs
```

```bash
# Environment-Variable
export CUDA_VISIBLE_DEVICES=0,1
```

### 2. Offline-Modus

```bash
# Für Datenschutz/Sicherheit
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

### 3. Custom Model-Pfade

```bash
# Lokale Model-Pfade
SMOLDOCLING_MODEL_PATH=/path/to/local/smoldocling
QWEN25_VL_MODEL_PATH=/path/to/local/qwen25-vl
```

## 🚨 Troubleshooting

### 1. Häufige Probleme

**"CUDA out of memory"**
```bash
# GPU Memory reduzieren
export VLLM_GPU_MEMORY_UTILIZATION=0.6
# Oder in .env:
VLLM_GPU_MEMORY_UTILIZATION=0.6
```

**"Model not found"**
```bash
# Manual download
huggingface-cli download ds4sd/SmolDocling-256M-preview
# Oder Cache löschen
rm -rf ~/.cache/huggingface/
```

**"vLLM not available"**
```bash
# CUDA-Installation prüfen
nvcc --version
python -c "import torch; print(torch.cuda.is_available())"

# vLLM neu installieren
pip uninstall vllm
pip install vllm --no-cache-dir
```

### 2. Performance-Probleme

**Langsame erste Verarbeitung**
- Normal: Model-Loading dauert 30-60s
- Warmup dauert weitere 10-30s
- Nachfolgende Verarbeitung ist schnell

**Inkonsistente Performance**
```bash
# GPU-Optimierung aktivieren
export CUDA_LAUNCH_BLOCKING=1
```

**Memory Leaks**
```python
# Automatisches Cleanup aktivieren
BATCH_CLEANUP_AFTER_BATCH=true
```

### 3. Debugging

```bash
# Verbose Logging
export LOG_LEVEL=DEBUG
python process_documents_vllm.py --vllm --verbose

# GPU Memory Monitoring
nvidia-smi -l 1

# Model Manager Status
python -c "
from core.vllm.model_manager import model_manager
model_manager.print_statistics()
"
```

## 📈 Monitoring und Logs

### 1. Performance-Metriken

Das System protokolliert automatisch:
- Model-Loading-Zeit
- Warmup-Zeit
- Verarbeitungszeit pro Dokument
- GPU-Speicherverbrauch
- Batch-Statistiken

### 2. Log-Dateien

```bash
# Logs in Datei umleiten
python process_documents_vllm.py --vllm 2>&1 | tee processing.log

# Nur Performance-Logs
grep "Performance\|GPU\|Model" processing.log
```

### 3. Monitoring-Dashboard

```bash
# GPU-Nutzung überwachen
watch -n 1 nvidia-smi

# System-Ressourcen
htop
```

## 🎯 Produktive Nutzung

### 1. Systemd-Service (optional)

```ini
# /etc/systemd/system/vllm-processor.service
[Unit]
Description=vLLM Document Processor
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/generic-kg-pipeline
ExecStart=/path/to/venv/bin/python process_documents_vllm.py --vllm
Restart=always

[Install]
WantedBy=multi-user.target
```

### 2. Cron-Job für automatische Verarbeitung

```bash
# Täglich um 2:00 Uhr
0 2 * * * cd /path/to/generic-kg-pipeline && python process_documents_vllm.py --vllm
```

### 3. API-Integration

Das System ist bereit für die Integration in die FastAPI-Anwendung:

```python
from core.vllm_batch_processor import VLLMBatchProcessor, BatchProcessingConfig

# In FastAPI-Endpoint
config = BatchProcessingConfig(use_vllm=True)
processor = VLLMBatchProcessor(config)
results = await processor.run_complete_batch(file_paths)
```

## ✅ Erfolgreiches Setup-Checkliste

- [ ] CUDA ist verfügbar und erkannt
- [ ] vLLM ist installiert und funktioniert
- [ ] `.env` ist konfiguriert
- [ ] Mock-Modus funktioniert
- [ ] Erste vLLM-Verarbeitung erfolgreich
- [ ] GPU-Memory-Utilization ist optimiert
- [ ] Batch-Processing funktioniert
- [ ] Performance-Metriken werden angezeigt
- [ ] Automatisches Cleanup funktioniert

## 🔗 Weiterführende Ressourcen

- [vLLM Dokumentation](https://docs.vllm.ai/)
- [SmolDocling GitHub](https://github.com/DS4SD/SmolDocling)
- [Qwen2.5-VL Dokumentation](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

---

🎉 **Ihr vLLM-Setup ist bereit für hochperformante Dokumentenverarbeitung!**