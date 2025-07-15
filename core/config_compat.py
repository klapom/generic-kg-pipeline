"""
Kompatibilitäts-Layer für die alte Config API
Leitet alle Aufrufe an die neue unified_manager weiter
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional, List
import warnings

from pydantic import BaseModel, Field, validator

# Import neue Config
from .config_new.unified_manager import (
    get_config as new_get_config,
    get_config_manager,
    UnifiedConfig
)

# Für vollständige Kompatibilität
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


# Wrapper-Klassen für Backwards Compatibility
class PDFParsingConfig(BaseModel):
    """Configuration for PDF parsing"""
    provider: str = "vllm_smoldocling"
    vllm_endpoint: str = Field(default="http://localhost:8002")
    gpu_optimization: bool = True
    max_pages: int = 100
    
    @classmethod
    def from_unified(cls, config: UnifiedConfig):
        """Create from unified config"""
        return cls(
            provider=config.parsing.pdf.provider,
            vllm_endpoint=config.services.vllm.url,
            gpu_optimization=True,
            max_pages=config.models.vision.smoldocling.max_pages
        )


class OfficeParsingConfig(BaseModel):
    """Configuration for Office document parsing"""
    provider: str = "native"
    
    @classmethod
    def from_unified(cls, config: UnifiedConfig):
        return cls(provider="native")


class TextParsingConfig(BaseModel):
    """Configuration for text file parsing"""
    provider: str = "native"
    
    @classmethod
    def from_unified(cls, config: UnifiedConfig):
        return cls(provider="native")


class ParsingConfig(BaseModel):
    """Combined parsing configuration"""
    pdf: PDFParsingConfig = PDFParsingConfig()
    office: OfficeParsingConfig = OfficeParsingConfig()
    text: TextParsingConfig = TextParsingConfig()
    
    @classmethod
    def from_unified(cls, config: UnifiedConfig):
        return cls(
            pdf=PDFParsingConfig.from_unified(config),
            office=OfficeParsingConfig.from_unified(config),
            text=TextParsingConfig.from_unified(config)
        )


class HochschulLLMConfig(BaseModel):
    """Configuration for Hochschul LLM"""
    endpoint: str
    api_key: str
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    temperature: float = 0.1
    max_tokens: int = 4000
    timeout: int = 60
    
    @classmethod
    def from_unified(cls, config: UnifiedConfig):
        return cls(
            endpoint=config.services.hochschul_llm.url,
            api_key=config.services.hochschul_llm.api_key,
            model=config.services.hochschul_llm.model,
            temperature=config.models.llm.temperature,
            max_tokens=config.models.llm.max_tokens,
            timeout=config.services.hochschul_llm.timeout
        )


class OllamaConfig(BaseModel):
    """Configuration for Ollama (fallback LLM)"""
    endpoint: str = "http://localhost:11434"
    model: str = "llama2"
    temperature: float = 0.1
    
    @classmethod
    def from_unified(cls, config: UnifiedConfig):
        return cls(
            endpoint=config.services.ollama.url,
            model=config.services.ollama.model,
            temperature=config.models.llm.temperature
        )


class LLMConfig(BaseModel):
    """LLM configuration for triple extraction"""
    provider: str = "hochschul"
    hochschul: Optional[HochschulLLMConfig] = None
    fallback_provider: str = "ollama"
    ollama: OllamaConfig = OllamaConfig()
    temperature: float = 0.1
    max_tokens: int = 4000
    top_p: float = 0.9
    
    @classmethod
    def from_unified(cls, config: UnifiedConfig):
        llm_config = cls(
            provider=config.models.llm.provider,
            fallback_provider="ollama",
            temperature=config.models.llm.temperature,
            max_tokens=config.models.llm.max_tokens,
            top_p=config.models.llm.top_p,
            ollama=OllamaConfig.from_unified(config)
        )
        
        # Add Hochschul config if it's the provider
        if config.models.llm.provider == "hochschul":
            llm_config.hochschul = HochschulLLMConfig.from_unified(config)
        
        return llm_config


class ChunkingConfig(BaseModel):
    """Chunking configuration"""
    strategy: str = "semantic"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    @classmethod
    def from_unified(cls, config: UnifiedConfig):
        return cls(
            strategy=config.chunking.default.get('strategy', 'semantic'),
            chunk_size=config.chunking.default.get('chunk_size', 1000),
            chunk_overlap=config.chunking.default.get('chunk_overlap', 200)
        )


class TripleStoreConfig(BaseModel):
    """Triple store configuration"""
    provider: str = "fuseki"
    endpoint: str = "http://localhost:3030"
    dataset: str = "kg_dataset"
    username: Optional[str] = None
    password: Optional[str] = None
    
    @classmethod
    def from_unified(cls, config: UnifiedConfig):
        return cls(
            provider="fuseki",
            endpoint=config.services.fuseki.url,
            dataset=config.services.fuseki.dataset
        )


class VectorStoreConfig(BaseModel):
    """Vector store configuration"""
    provider: str = "chroma"
    host: str = "http://localhost:8000"
    collection: str = "documents"
    
    @classmethod
    def from_unified(cls, config: UnifiedConfig):
        return cls(
            provider="chroma",
            host=config.services.chromadb.url,
            collection=config.services.chromadb.collection
        )


class StorageConfig(BaseModel):
    """Storage configuration"""
    output_dir: str = "data/output"
    processed_dir: str = "data/processed"
    triple_store: TripleStoreConfig = TripleStoreConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    
    @classmethod
    def from_unified(cls, config: UnifiedConfig):
        return cls(
            output_dir=config.storage.directories.get('output', 'data/output'),
            processed_dir=config.storage.directories.get('processed', 'data/processed'),
            triple_store=TripleStoreConfig.from_unified(config),
            vector_store=VectorStoreConfig.from_unified(config)
        )


class DomainConfig(BaseModel):
    """Domain configuration"""
    name: str = "general"
    ontology_path: str = "plugins/ontologies/general.ttl"
    
    @classmethod
    def from_unified(cls, config: UnifiedConfig):
        return cls(
            name=config.domain.name,
            ontology_path=config.domain.ontology_path
        )


class RAGConfig(BaseModel):
    """RAG configuration"""
    enabled: bool = False
    retrieval_method: str = "hybrid"
    top_k: int = 5
    
    @classmethod
    def from_unified(cls, config: UnifiedConfig):
        return cls(
            enabled=config.features.rag_generation,
            retrieval_method="hybrid",
            top_k=5
        )


class VLLMConfig(BaseModel):
    """vLLM configuration"""
    base_url: str = "http://localhost:8002"
    model_id: str = "numinamath/SmolDocling-256M-Preview"
    gpu_memory_utilization: float = 0.2
    max_model_len: Optional[int] = None
    
    @classmethod
    def from_unified(cls, config: UnifiedConfig):
        return cls(
            base_url=config.services.vllm.url,
            model_id=config.models.vision.smoldocling.model_id,
            gpu_memory_utilization=config.models.vision.smoldocling.gpu_memory_utilization
        )


class BatchProcessingConfig(BaseModel):
    """Configuration for batch processing"""
    default_mode: str = "vllm"
    max_concurrent: int = 3
    timeout_seconds: int = 600
    enable_chunking: bool = True
    enable_context_inheritance: bool = True
    enable_visual_analysis: bool = True
    auto_gpu_memory_optimization: bool = True
    model_warmup_enabled: bool = True
    cleanup_after_batch: bool = True
    
    @classmethod
    def from_unified(cls, config: UnifiedConfig):
        return cls(
            max_concurrent=config.batch.processing.get('max_workers', 3),
            timeout_seconds=config.batch.processing.get('chunk_timeout', 600),
            enable_chunking=True,
            enable_context_inheritance=config.chunking.context.get('inherit_metadata', True),
            enable_visual_analysis=config.features.visual_analysis
        )


class Config(BaseSettings):
    """
    Kompatibilitäts-Wrapper für die alte Config-Klasse
    Leitet alle Zugriffe an die neue UnifiedConfig weiter
    """
    # Definiere alle Felder für BaseSettings
    domain: DomainConfig = Field(default_factory=DomainConfig)
    parsing: ParsingConfig = Field(default_factory=ParsingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    vllm: VLLMConfig = Field(default_factory=VLLMConfig)
    batch_processing: BatchProcessingConfig = Field(default_factory=BatchProcessingConfig)
    
    def __init__(self, **kwargs):
        # BaseSettings init (für Kompatibilität)
        super().__init__(**kwargs)
        self._unified = new_get_config()
        self._init_attributes()
    
    class Config:
        # Für BaseSettings Kompatibilität
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def _init_attributes(self):
        """Initialize all config attributes from unified config"""
        self.domain = DomainConfig.from_unified(self._unified)
        self.parsing = ParsingConfig.from_unified(self._unified)
        self.llm = LLMConfig.from_unified(self._unified)
        self.chunking = ChunkingConfig.from_unified(self._unified)
        self.storage = StorageConfig.from_unified(self._unified)
        self.rag = RAGConfig.from_unified(self._unified)
        self.vllm = VLLMConfig.from_unified(self._unified)
        self.batch_processing = BatchProcessingConfig.from_unified(self._unified)
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "Config":
        """Legacy method - just returns new Config instance"""
        warnings.warn(
            "Config.from_yaml is deprecated. Configuration is now loaded from config.yaml",
            DeprecationWarning,
            stacklevel=2
        )
        return cls()
    
    def validate_config(self) -> None:
        """Legacy validation method"""
        # Validation happens in UnifiedConfig now
        pass
    
    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'domain': self.domain.dict(),
            'parsing': self.parsing.dict(),
            'llm': self.llm.dict(),
            'chunking': self.chunking.dict(),
            'storage': self.storage.dict(),
            'rag': self.rag.dict(),
            'vllm': self.vllm.dict(),
            'batch_processing': self.batch_processing.dict()
        }


# Globale Config-Instanz für Kompatibilität
_config: Optional[Config] = None


def get_config(config_path: Optional[Path] = None) -> Config:
    """
    Get configuration - Kompatibilitäts-Funktion
    Ignoriert config_path, da wir jetzt unified config nutzen
    """
    global _config
    
    if _config is None:
        if config_path is not None:
            warnings.warn(
                "config_path parameter is deprecated. Configuration is now loaded from config.yaml",
                DeprecationWarning,
                stacklevel=2
            )
        _config = Config()
    
    return _config


def load_chunking_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load chunking configuration - Kompatibilitäts-Funktion
    """
    if config_path is not None:
        warnings.warn(
            "config_path parameter is deprecated. Chunking config is now in config.yaml",
            DeprecationWarning,
            stacklevel=2
        )
    
    # Return chunking config from unified config
    unified_config = new_get_config()
    return unified_config.chunking.dict()


# Re-export für einfachen Import
__all__ = [
    'Config',
    'get_config',
    'load_chunking_config',
    'PDFParsingConfig',
    'LLMConfig',
    'ChunkingConfig',
    'StorageConfig',
    'DomainConfig',
    'VLLMConfig',
    'BatchProcessingConfig'
]