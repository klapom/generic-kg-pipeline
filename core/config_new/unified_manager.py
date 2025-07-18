"""
Unified Configuration Manager for Generic Knowledge Graph Pipeline
Ersetzt die fragmentierte Konfiguration durch eine einheitliche Lösung
"""
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
from pydantic import BaseModel, Field, validator
import json
from datetime import datetime


class ConfigManager:
    """
    Zentraler Konfigurations-Manager mit folgenden Features:
    - Eine einzige Konfigurationsdatei
    - Environment-Variablen-Substitution
    - Validierung durch Pydantic
    - Hot-Reload Unterstützung
    - Schema-Export für GUI
    """
    
    def __init__(self, config_path: Union[str, Path] = "config.yaml"):
        self.config_path = Path(config_path)
        self._raw_config: Dict[str, Any] = {}
        self._config: Optional['UnifiedConfig'] = None
        self._last_modified: Optional[float] = None
        self.load()
    
    def load(self) -> None:
        """Lade und parse Konfiguration"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        # 1. Lade YAML
        with open(self.config_path, 'r') as f:
            raw = yaml.safe_load(f)
        
        # 2. Ersetze Environment-Variablen
        self._raw_config = self._substitute_env_vars(raw)
        
        # 3. Validiere und erstelle Config-Objekt
        self._config = UnifiedConfig(**self._raw_config)
        
        # 4. Speichere Modification Time für Hot-Reload
        self._last_modified = self.config_path.stat().st_mtime
    
    def _substitute_env_vars(self, obj: Any) -> Any:
        """
        Rekursiv Environment-Variablen ersetzen
        Format: ${VAR_NAME:default_value}
        """
        if isinstance(obj, str):
            # Pattern für ${VAR:default} oder ${VAR}
            pattern = r'\$\{([^:}]+)(?::([^}]+))?\}'
            
            def replacer(match):
                var_name = match.group(1)
                default = match.group(2)
                value = os.environ.get(var_name, default if default is not None else '')
                
                # Konvertiere Strings zu korrekten Typen
                if value.lower() in ('true', 'false'):
                    return value.lower() == 'true'
                elif value.isdigit():
                    return int(value)
                elif '.' in value and all(part.isdigit() for part in value.split('.', 1)):
                    return float(value)
                return value
            
            # Funktion die immer String zurückgibt
            def safe_replacer(match):
                result = replacer(match)
                return str(result)
            
            # Spezialfall: Wenn der ganze String eine Variable ist
            if obj.startswith('${') and obj.endswith('}'):
                # Direkter Ersatz mit Typ-Erhaltung
                match = re.match(pattern, obj)
                if match:
                    return replacer(match)
                return obj
            else:
                # String-Ersatz für eingebettete Variablen
                return re.sub(pattern, safe_replacer, obj)
                
        elif isinstance(obj, dict):
            return {k: self._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        return obj
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Hole Wert mit Punkt-Notation
        Beispiel: config.get('services.vllm.url')
        """
        keys = path.split('.')
        value = self._raw_config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, path: str, value: Any) -> None:
        """
        Setze Wert mit Punkt-Notation (für Runtime-Updates)
        """
        keys = path.split('.')
        target = self._raw_config
        
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        
        target[keys[-1]] = value
        
        # Re-validiere
        self._config = UnifiedConfig(**self._raw_config)
    
    @property
    def config(self) -> 'UnifiedConfig':
        """Validiertes Config-Objekt"""
        if self._config is None:
            self.load()
        return self._config
    
    def is_modified(self) -> bool:
        """Prüfe ob Config-Datei geändert wurde"""
        if not self.config_path.exists():
            return False
        current_mtime = self.config_path.stat().st_mtime
        return current_mtime != self._last_modified
    
    def reload(self) -> None:
        """Konfiguration neu laden"""
        self.load()
    
    def save(self, backup: bool = True) -> None:
        """Speichere aktuelle Konfiguration"""
        if backup and self.config_path.exists():
            backup_path = self.config_path.with_suffix(
                f'.backup.{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            )
            self.config_path.rename(backup_path)
        
        with open(self.config_path, 'w') as f:
            yaml.dump(self._raw_config, f, default_flow_style=False, sort_keys=False)
    
    def export_schema(self) -> Dict[str, Any]:
        """Exportiere Schema für GUI"""
        schema = UnifiedConfig.schema()
        # Füge Metadaten hinzu
        schema['_metadata'] = {
            'version': '1.0.0',
            'generated_at': datetime.now().isoformat(),
            'config_file': str(self.config_path)
        }
        return schema
    
    def export_for_gui(self) -> Dict[str, Any]:
        """Exportiere aktuelle Config mit Schema für GUI"""
        return {
            'schema': self.export_schema(),
            'values': self._raw_config,
            'environment': self._get_env_vars_info()
        }
    
    def _get_env_vars_info(self) -> Dict[str, Any]:
        """Sammle Infos über verwendete Environment-Variablen"""
        env_vars = {}
        pattern = r'\$\{([^:}]+)(?::([^}]+))?\}'
        
        def extract_vars(obj: Any, path: str = ''):
            if isinstance(obj, str):
                matches = re.findall(pattern, obj)
                for var_name, default in matches:
                    env_vars[var_name] = {
                        'path': path,
                        'default': default,
                        'current': os.environ.get(var_name),
                        'is_set': var_name in os.environ
                    }
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    new_path = f"{path}.{k}" if path else k
                    extract_vars(v, new_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    extract_vars(item, f"{path}[{i}]")
        
        # Lade Original-YAML ohne Substitution
        with open(self.config_path, 'r') as f:
            raw = yaml.safe_load(f)
        extract_vars(raw)
        
        return env_vars
    
    def validate(self) -> Dict[str, Any]:
        """Validiere Konfiguration und gebe Probleme zurück"""
        issues = {
            'errors': [],
            'warnings': [],
            'info': []
        }
        
        # Prüfe Service-Erreichbarkeit
        if self.config.services.vllm.url.startswith('http://localhost'):
            issues['warnings'].append(
                "vLLM service points to localhost - ensure it's running"
            )
        
        # Prüfe GPU-Memory Summe
        total_gpu = (
            self.config.models.vision.smoldocling.gpu_memory_utilization +
            self.config.models.vision.qwen_vl.gpu_memory_utilization
        )
        if total_gpu > 1.0:
            issues['errors'].append(
                f"Total GPU memory utilization exceeds 100%: {total_gpu*100}%"
            )
        
        # Prüfe Pfade
        for path_config in [
            self.config.domain.ontology_path,
            self.config.templates.paths.base
        ]:
            if not Path(path_config).exists():
                issues['warnings'].append(f"Path does not exist: {path_config}")
        
        return issues


# Basis-Konfigurationsmodelle
class GeneralConfig(BaseModel):
    name: str = "Generic Knowledge Graph Pipeline"
    version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"


class DomainConfig(BaseModel):
    name: str = "general"
    ontology_path: str = "plugins/ontologies/general.ttl"
    enabled_formats: list[str] = ["pdf", "docx", "xlsx", "pptx", "txt"]


class ServiceConfig(BaseModel):
    url: str
    timeout: int = 60
    retry_attempts: int = 3
    
    @validator('url')
    def validate_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v


class VLLMServiceConfig(ServiceConfig):
    health_check_enabled: bool = True


class HochschulLLMConfig(ServiceConfig):
    api_key: str = ""
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"


class OllamaServiceConfig(ServiceConfig):
    model: str = "llama2"
    enabled: bool = False


class FusekiServiceConfig(ServiceConfig):
    dataset: str = "kg_dataset"


class ChromaDBServiceConfig(ServiceConfig):
    collection: str = "documents"


class ServicesConfig(BaseModel):
    vllm: VLLMServiceConfig
    hochschul_llm: HochschulLLMConfig
    fuseki: FusekiServiceConfig
    chromadb: ChromaDBServiceConfig
    ollama: OllamaServiceConfig


class LLMConfig(BaseModel):
    provider: str = "hochschul"
    temperature: float = 0.1
    max_tokens: int = 4000
    top_p: float = 0.9
    stream: bool = False


class VisionModelConfig(BaseModel):
    enabled: bool = True
    model_id: str
    gpu_memory_utilization: float = 0.2
    
    @validator('gpu_memory_utilization')
    def validate_gpu_memory(cls, v):
        if not 0 < v <= 1:
            raise ValueError('GPU memory utilization must be between 0 and 1')
        return v


class SmolDoclingConfig(VisionModelConfig):
    max_pages: int = 15
    dtype: str = "float16"
    trust_remote_code: bool = True


class QwenVLConfig(VisionModelConfig):
    max_image_size: int = 1024


class VisionModelsConfig(BaseModel):
    smoldocling: SmolDoclingConfig
    qwen_vl: QwenVLConfig


class ModelsConfig(BaseModel):
    llm: LLMConfig
    vision: VisionModelsConfig


class LayoutConfig(BaseModel):
    use_layout: bool = True
    table_x_tolerance: int = 3
    table_y_tolerance: int = 3
    text_x_tolerance: int = 5
    text_y_tolerance: int = 5


class PDFParsingConfig(BaseModel):
    provider: str = "hybrid"
    pdfplumber_mode: int = 1
    layout: LayoutConfig
    complex_detection: Dict[str, Any]
    table_extraction: Dict[str, Any]


class ParsingConfig(BaseModel):
    pdf: PDFParsingConfig
    office: Dict[str, Any]
    common: Dict[str, Any]


class ChunkingConfig(BaseModel):
    default: Dict[str, Any]
    strategies: Dict[str, Any]
    context: Dict[str, Any]


class TriplesConfig(BaseModel):
    generation: Dict[str, Any]
    extraction: Dict[str, Any]
    validation: Dict[str, Any]


class StorageConfig(BaseModel):
    directories: Dict[str, str]
    files: Dict[str, Any]


class BatchConfig(BaseModel):
    processing: Dict[str, Any]
    retry: Dict[str, Any]
    resources: Dict[str, Any]


class TemplatePathsConfig(BaseModel):
    base: str = "plugins/templates"
    custom: str = ""

class TemplatesConfig(BaseModel):
    paths: TemplatePathsConfig
    active: Dict[str, str]


class FeaturesConfig(BaseModel):
    visual_analysis: bool = True
    table_extraction: bool = True
    context_enhancement: bool = True
    rag_generation: bool = False
    multi_language: bool = False
    caching: bool = True
    parallel_processing: bool = True


class MonitoringConfig(BaseModel):
    metrics: Dict[str, Any]
    health: Dict[str, Any]
    logging: Dict[str, Any]


class APIConfig(BaseModel):
    server: Dict[str, Any]
    security: Dict[str, Any]
    docs: Dict[str, Any]


class UnifiedConfig(BaseModel):
    """Hauptkonfigurationsmodell"""
    profile: str = "dev"
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
    api: Optional[APIConfig] = None
    
    class Config:
        extra = "forbid"
        validate_assignment = True


# Globale Instanz
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """Hole Config Manager Singleton"""
    global _config_manager
    if _config_manager is None or config_path:
        _config_manager = ConfigManager(config_path or "config.yaml")
    return _config_manager


def get_config() -> UnifiedConfig:
    """Hole validierte Konfiguration"""
    return get_config_manager().config


def reload_config() -> None:
    """Lade Konfiguration neu"""
    get_config_manager().reload()

