# Configuration Management Refactoring Plan

## 🎯 Ziel
Eine einzige, einheitliche Konfigurationsdatei mit klarer Struktur, die später durch eine grafische Oberfläche verwaltet werden kann.

## 📊 Aktuelle Situation

### Probleme:
- **5+ Konfigurationsdateien** (default.yaml, chunking.yaml, .env, config.py)
- **300+ Zeilen Python-Konfigurationscode**
- **Massive Duplikation** zwischen YAML und Python-Klassen
- **Hardcoded Werte** überall im Code verstreut
- **Inkonsistente Patterns** (get_config(), lokale Configs, Environment-Variablen)

### Gefundene Konfigurationsquellen:
1. `config/default.yaml` (127 Zeilen)
2. `config/chunking.yaml` (173 Zeilen)
3. `core/config.py` (318 Zeilen mit 15+ Klassen)
4. `.env.example` (56 Zeilen)
5. Hardcoded Werte in 20+ Dateien

## 🏗️ Neue Einheitliche Struktur

### 1. Eine Konfigurationsdatei: `config.yaml`

```yaml
# Generic Knowledge Graph Pipeline Configuration
# Version: 1.0.0

# Umgebungs-Profile (dev, test, prod)
profile: ${PROFILE:dev}

# Allgemeine Einstellungen
general:
  name: "Generic KG Pipeline"
  version: "1.0.0"
  debug: ${DEBUG:false}
  log_level: ${LOG_LEVEL:INFO}
  
# Domain-Konfiguration
domain:
  name: ${DOMAIN_NAME:general}
  ontology: "plugins/ontologies/${domain.name}.ttl"
  enabled_formats: [pdf, docx, xlsx, pptx, txt]

# Service-Endpoints (für alle Services)
services:
  vllm:
    url: ${VLLM_URL:http://localhost:8002}
    timeout: 300
    health_check_path: /health
  hochschul_llm:
    url: ${HOCHSCHUL_LLM_URL:http://localhost:8001}
    api_key: ${HOCHSCHUL_LLM_API_KEY}
    timeout: 60
  fuseki:
    url: ${FUSEKI_URL:http://localhost:3030}
    dataset: ${FUSEKI_DATASET:kg_dataset}
  chromadb:
    url: ${CHROMADB_URL:http://localhost:8000}
    collection: ${CHROMADB_COLLECTION:documents}
  ollama:
    url: ${OLLAMA_URL:http://localhost:11434}
    model: ${OLLAMA_MODEL:llama2}

# Modell-Konfiguration
models:
  # Haupt-LLM
  llm:
    provider: ${LLM_PROVIDER:hochschul}  # hochschul, ollama, openai
    temperature: 0.1
    max_tokens: 4000
    top_p: 0.9
    
  # Vision-Language Models
  vision:
    smoldocling:
      enabled: true
      model_id: "numinamath/SmolDocling-256M-Preview"
      gpu_memory: 0.2
      max_pages: 15
      
    qwen_vl:
      enabled: true
      model_id: "Qwen/Qwen2-VL-7B-Instruct"
      gpu_memory: 0.8
      
# Parser-Konfiguration
parsing:
  # PDF-Spezifisch
  pdf:
    provider: ${PDF_PARSER:hybrid}  # native, hybrid, smoldocling
    pdfplumber_mode: 1  # 0=never, 1=fallback, 2=always
    layout:
      use_layout: true
      table_x_tolerance: 3
      table_y_tolerance: 3
      text_x_tolerance: 5
      text_y_tolerance: 5
    complex_detection:
      min_text_blocks: 2
      min_tables: 1
      coverage_threshold: 0.8
      
  # Allgemeine Parser-Einstellungen
  common:
    max_file_size: 100  # MB
    timeout: 120
    encoding: utf-8
    
# Chunking-Konfiguration
chunking:
  default_strategy: semantic
  chunk_size: 1000
  chunk_overlap: 200
  
  strategies:
    pdf:
      preserve_tables: true
      table_as_single_chunk: true
      respect_page_boundaries: true
      
    text:
      split_by: sentence
      min_chunk_size: 100
      
  context:
    inherit_metadata: true
    max_inheritance_depth: 3
    
# Triple-Generierung
triples:
  generation:
    batch_size: 10
    context_window: 2000
    include_metadata: true
    
  extraction:
    patterns:
      - "entity_relation_entity"
      - "subject_predicate_object"
    confidence_threshold: 0.7
    
# Storage-Konfiguration  
storage:
  output_dir: ${OUTPUT_DIR:data/output}
  processed_dir: ${PROCESSED_DIR:data/processed}
  temp_dir: ${TEMP_DIR:/tmp/kg_pipeline}
  
# Batch-Processing
batch:
  max_workers: ${MAX_WORKERS:4}
  queue_size: 100
  retry_attempts: 3
  retry_delay: 5
  
# Templates und Prompts
templates:
  base_path: "plugins/templates"
  custom_path: ${CUSTOM_TEMPLATES_PATH:null}
  
# Feature-Flags (für GUI später wichtig)
features:
  visual_analysis: true
  table_extraction: true
  context_enhancement: true
  rag_generation: false
  
# Monitoring
monitoring:
  metrics_enabled: ${METRICS_ENABLED:false}
  metrics_port: 9090
  health_check_interval: 30
```

### 2. Vereinfachte Python-Konfiguration

```python
# core/config/manager.py
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
import os
from pydantic import BaseModel, Field, validator
import re

class ConfigManager:
    """Zentraler Konfigurations-Manager"""
    
    def __init__(self, config_path: Path = Path("config.yaml")):
        self.config_path = config_path
        self._raw_config: Dict[str, Any] = {}
        self._config: Optional[UnifiedConfig] = None
        self.load()
    
    def load(self) -> None:
        """Lade und parse Konfiguration"""
        # 1. Lade YAML
        with open(self.config_path) as f:
            raw = yaml.safe_load(f)
        
        # 2. Ersetze Environment-Variablen
        self._raw_config = self._substitute_env_vars(raw)
        
        # 3. Validiere und erstelle Config-Objekt
        self._config = UnifiedConfig(**self._raw_config)
    
    def _substitute_env_vars(self, obj: Any) -> Any:
        """Rekursiv Environment-Variablen ersetzen"""
        if isinstance(obj, str):
            # Pattern: ${VAR_NAME:default_value}
            pattern = r'\$\{([^:}]+)(?::([^}]+))?\}'
            
            def replacer(match):
                var_name = match.group(1)
                default = match.group(2)
                return os.environ.get(var_name, default or '')
            
            return re.sub(pattern, replacer, obj)
        elif isinstance(obj, dict):
            return {k: self._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        return obj
    
    def get(self, path: str, default: Any = None) -> Any:
        """Hole Wert mit Punkt-Notation (z.B. 'services.vllm.url')"""
        keys = path.split('.')
        value = self._raw_config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    @property
    def config(self) -> 'UnifiedConfig':
        """Validiertes Config-Objekt"""
        return self._config
    
    def reload(self) -> None:
        """Konfiguration neu laden"""
        self.load()
    
    def save(self, data: Dict[str, Any]) -> None:
        """Speichere Konfiguration (für GUI später)"""
        with open(self.config_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        self.reload()
    
    def export_schema(self) -> Dict[str, Any]:
        """Exportiere Schema für GUI"""
        return UnifiedConfig.schema()

# Einzige Konfigurationsklasse
class UnifiedConfig(BaseModel):
    """Unified configuration model"""
    profile: str = Field("dev", description="Environment profile")
    general: GeneralConfig
    domain: DomainConfig  
    services: ServicesConfig
    models: ModelsConfig
    parsing: ParsingConfig
    chunking: ChunkingConfig
    triples: TriplesConfig
    storage: StorageConfig
    batch: BatchConfig
    templates: TemplatesConfig
    features: FeaturesConfig
    monitoring: MonitoringConfig
    
    class Config:
        extra = "forbid"  # Keine unbekannten Felder

# Globale Instanz
_config_manager: Optional[ConfigManager] = None

def get_config_manager() -> ConfigManager:
    """Hole Config Manager Singleton"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_config() -> UnifiedConfig:
    """Hole validierte Konfiguration"""
    return get_config_manager().config
```

### 3. Migration der bestehenden Konfiguration

```python
# scripts/migrate_config.py
"""Migriere alte Konfiguration zur neuen einheitlichen Struktur"""

def migrate_configs():
    # 1. Lade alte Configs
    old_default = load_yaml("config/default.yaml")
    old_chunking = load_yaml("config/chunking.yaml")
    env_vars = load_env(".env.example")
    
    # 2. Mappe zur neuen Struktur
    new_config = {
        "profile": "dev",
        "general": {
            "name": old_default.get("domain", {}).get("name", "Generic KG Pipeline"),
            # ...
        },
        # ... mapping logic
    }
    
    # 3. Speichere neue Config
    save_yaml("config.yaml", new_config)
    
    # 4. Erstelle Backup
    backup_old_configs()
```

## 📝 Implementierungs-Schritte

### Phase 1: Vorbereitung (Tag 1)
1. ✅ Analyse der aktuellen Konfiguration
2. ⏳ Backup aller Config-Dateien erstellen
3. ⏳ Neue Config-Struktur entwerfen
4. ⏳ Schema für GUI definieren

### Phase 2: Core Implementation (Tag 2-3)
1. ⏳ `ConfigManager` Klasse implementieren
2. ⏳ `UnifiedConfig` mit allen Sub-Configs
3. ⏳ Environment-Variablen-Substitution
4. ⏳ Validierung und Defaults

### Phase 3: Migration (Tag 4-5)
1. ⏳ Migration-Script schreiben
2. ⏳ Alle `get_config()` Aufrufe anpassen
3. ⏳ Hardcoded Werte entfernen
4. ⏳ Tests anpassen

### Phase 4: Integration (Tag 6-7)
1. ⏳ Docker-Compose anpassen
2. ⏳ CI/CD Updates
3. ⏳ Dokumentation
4. ⏳ GUI-Schema exportieren

## 🎯 Vorteile der neuen Lösung

1. **Eine Datei** statt 5+ Dateien
2. **Klare Hierarchie** für GUI-Navigation
3. **Typsicherheit** durch Pydantic
4. **Environment-Variablen** einheitlich
5. **Versionierung** möglich
6. **Schema-Export** für GUI
7. **Hot-Reload** Unterstützung
8. **Profile** (dev/test/prod)

## 🔧 GUI-Integration (Zukunft)

```python
# api/routers/config.py
@router.get("/config/schema")
async def get_config_schema():
    """Schema für GUI"""
    return get_config_manager().export_schema()

@router.get("/config")
async def get_current_config():
    """Aktuelle Konfiguration"""
    return get_config_manager()._raw_config

@router.put("/config")
async def update_config(updates: Dict[str, Any]):
    """Update Konfiguration"""
    manager = get_config_manager()
    current = manager._raw_config
    merged = deep_merge(current, updates)
    manager.save(merged)
    return {"status": "updated"}
```

## ⚠️ Wichtige Überlegungen

1. **Backwards Compatibility**: Alte Config-Dateien behalten bis Migration abgeschlossen
2. **Sensitive Daten**: API-Keys nur in Environment-Variablen
3. **Validierung**: Strenge Validierung aller Werte
4. **Dokumentation**: Jedes Feld dokumentieren für GUI
5. **Defaults**: Sinnvolle Defaults für alle Werte

## 📊 Erfolgs-Metriken

- Config-Dateien: 5+ → 1
- Config-Code: 300+ Zeilen → <100 Zeilen
- Hardcoded Werte: 50+ → 0
- Load-Zeit: <100ms
- GUI-Ready: ✅

Diese Lösung schafft eine solide Basis für die spätere GUI-Integration!