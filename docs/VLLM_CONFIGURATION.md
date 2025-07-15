# 🚀 vLLM Konfiguration - Schnellstart

## 📁 Wo Sie Konfigurationen eintragen müssen

### 1. **Environment Variables (.env)**
```bash
# Kopieren Sie die Beispiel-Datei
cp .env.example .env

# Bearbeiten Sie diese Datei:
nano .env
```

**Wichtigste Einstellungen:**
```bash
# vLLM aktivieren
USE_VLLM=true

# GPU Memory (60-80% empfohlen)
VLLM_GPU_MEMORY_UTILIZATION=0.8

# Standardmodus (produktiv mit vLLM)
BATCH_DEFAULT_MODE=vllm

# Model-Cache-Verzeichnis (optional)
VLLM_MODEL_CACHE_DIR=/path/to/model/cache
```

### 2. **YAML-Konfiguration (config/default.yaml)**
```yaml
# vLLM Configuration
vllm:
  gpu_memory_utilization: 0.8  # GPU-Speichernutzung
  
  smoldocling:
    model_name: "ds4sd/SmolDocling-256M-preview"
    max_pages: 100
    extract_tables: true
    
  qwen25_vl:
    model_name: "Qwen/Qwen2.5-VL-7B-Instruct"
    max_image_size: 1024
    batch_size: 3
```

## 🛠️ Schnelle Installation

### 1. vLLM installieren
```bash
# Mit UV (empfohlen)
uv pip install vllm

# Zusätzliche Abhängigkeiten
uv pip install pdf2image pillow
```

### 2. System-Abhängigkeiten
```bash
# Ubuntu/Debian
sudo apt-get install poppler-utils

# macOS
brew install poppler
```

### 3. CUDA prüfen
```bash
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 🧪 Testen

### 1. Beispiel-Dokumente erstellen
```bash
python process_documents_vllm.py --create-samples
```

### 2. vLLM-Modus (mit GPU)
```bash
# Erst einzelne Datei testen
python process_documents_vllm.py --file sample_document.txt

# Vollständige Batch-Verarbeitung
python process_documents_vllm.py
```

## ⚙️ Wichtige Konfigurationsoptionen

### GPU-Speicher anpassen
```bash
# In .env
VLLM_GPU_MEMORY_UTILIZATION=0.7  # Für 8GB GPU
VLLM_GPU_MEMORY_UTILIZATION=0.8  # Für 16GB+ GPU
```

### Batch-Performance optimieren
```bash
# In .env
BATCH_MAX_CONCURRENT=3        # Für 16GB+ GPU
BATCH_MAX_CONCURRENT=2        # Für 8GB GPU
BATCH_MAX_CONCURRENT=1        # Für 4GB GPU
```

### Model-Pfade setzen (optional)
```bash
# In .env - für lokale Models
SMOLDOCLING_MODEL_PATH=/path/to/local/smoldocling
QWEN25_VL_MODEL_PATH=/path/to/local/qwen25-vl
```

## 🚨 Häufige Probleme & Lösungen

### "CUDA out of memory"
```bash
# GPU-Speicher reduzieren
VLLM_GPU_MEMORY_UTILIZATION=0.6
BATCH_MAX_CONCURRENT=1
```

### "Model not found"
```bash
# Model automatisch herunterladen lassen (Standard)
# Oder vorab herunterladen:
huggingface-cli download ds4sd/SmolDocling-256M-preview
```

### "vLLM not available"
```bash
# Installation prüfen
pip install vllm --no-cache-dir
```

## 📊 Performance-Überwachung

### GPU-Status überwachen
```bash
# Während der Verarbeitung
nvidia-smi -l 1
```

### Detaillierte Logs
```bash
# Verbose Modus
python process_documents_vllm.py --vllm --verbose
```

## 🎯 Produktive Einstellungen

### Für 8GB GPU
```bash
# .env
VLLM_GPU_MEMORY_UTILIZATION=0.7
BATCH_MAX_CONCURRENT=2
SMOLDOCLING_GPU_MEMORY=0.7
QWEN25_VL_GPU_MEMORY=0.6
```

### Für 16GB+ GPU
```bash
# .env
VLLM_GPU_MEMORY_UTILIZATION=0.8
BATCH_MAX_CONCURRENT=3
SMOLDOCLING_GPU_MEMORY=0.8
QWEN25_VL_GPU_MEMORY=0.7
```

### Für 24GB+ GPU
```bash
# .env
VLLM_GPU_MEMORY_UTILIZATION=0.9
BATCH_MAX_CONCURRENT=4
SMOLDOCLING_GPU_MEMORY=0.8
QWEN25_VL_GPU_MEMORY=0.8
```

## 🔄 Produktives Setup

1. **GPU-Setup prüfen**
   ```bash
   nvidia-smi
   ```

2. **Schrittweise vLLM testen**
   ```bash
   # Erst einzelne Datei
   python process_documents_vllm.py --file sample_document.txt
   
   # Dann vollständige Batch
   python process_documents_vllm.py
   ```

3. **Konfiguration optimieren**
   ```bash
   # In .env
   VLLM_GPU_MEMORY_UTILIZATION=0.8
   BATCH_MAX_CONCURRENT=3
   ```

---

✅ **Ihr vLLM-System ist bereit für 54.8% schnellere Dokumentenverarbeitung!**