"""Configuration management for the Knowledge Graph Pipeline System"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field, validator
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class PDFParsingConfig(BaseModel):
    """Configuration for PDF parsing"""
    provider: str = "vllm_smoldocling"
    vllm_endpoint: str = Field(default="http://localhost:8002")
    gpu_optimization: bool = True
    max_pages: int = 100


class OfficeParsingConfig(BaseModel):
    """Configuration for Office document parsing"""
    provider: str = "native"


class TextParsingConfig(BaseModel):
    """Configuration for text file parsing"""
    provider: str = "native"


class ParsingConfig(BaseModel):
    """Combined parsing configuration"""
    pdf: PDFParsingConfig = PDFParsingConfig()
    office: OfficeParsingConfig = OfficeParsingConfig()
    text: TextParsingConfig = TextParsingConfig()


class HochschulLLMConfig(BaseModel):
    """Configuration for Hochschul LLM"""
    endpoint: str
    api_key: str
    model: str = "qwen1.5-72b"
    temperature: float = 0.1
    max_tokens: int = 4000
    timeout: int = 60


class OllamaConfig(BaseModel):
    """Configuration for Ollama (fallback LLM)"""
    endpoint: str = "http://localhost:11434"
    model: str = "qwen:7b"
    temperature: float = 0.1


class LLMConfig(BaseModel):
    """LLM configuration for triple extraction"""
    provider: str = "hochschul"
    hochschul: Optional[HochschulLLMConfig] = None
    fallback_provider: str = "ollama"
    ollama: OllamaConfig = OllamaConfig()


class ChunkingConfig(BaseModel):
    """Configuration for document chunking"""
    max_tokens: int = 2000
    overlap_ratio: float = 0.2
    preserve_context: bool = True
    
    @validator("overlap_ratio")
    def validate_overlap_ratio(cls, v):
        if not 0 <= v <= 0.5:
            raise ValueError("overlap_ratio must be between 0 and 0.5")
        return v


class TripleStoreConfig(BaseModel):
    """Configuration for triple store"""
    type: str = "fuseki"
    endpoint: str = "http://localhost:3030"
    dataset: str = "kg_dataset"


class VectorStoreConfig(BaseModel):
    """Configuration for vector store"""
    type: str = "chromadb"
    endpoint: str = "http://localhost:8001"
    collection: str = "document_chunks"


class StorageConfig(BaseModel):
    """Combined storage configuration"""
    triple_store: TripleStoreConfig = TripleStoreConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()


class RAGConfig(BaseModel):
    """Configuration for RAG (Retrieval-Augmented Generation)"""
    similarity_threshold: float = 0.7
    max_context_chunks: int = 3
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


class DomainConfig(BaseModel):
    """Domain-specific configuration"""
    name: str = "general"
    ontology_path: str = "plugins/ontologies/general.ttl"
    enabled_formats: list[str] = ["pdf", "docx", "xlsx", "txt"]


class Config(BaseSettings):
    """Main configuration class"""
    domain: DomainConfig = DomainConfig()
    parsing: ParsingConfig = ParsingConfig()
    llm: LLMConfig = LLMConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    storage: StorageConfig = StorageConfig()
    rag: RAGConfig = RAGConfig()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "Config":
        """Load configuration from YAML file with environment variable substitution"""
        with open(yaml_path, "r") as f:
            yaml_content = f.read()
            
        # Substitute environment variables in format ${VAR_NAME}
        import re
        pattern = r'\$\{([^}]+)\}'
        
        def replace_env_var(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))
        
        yaml_content = re.sub(pattern, replace_env_var, yaml_content)
        
        # Parse YAML
        config_dict = yaml.safe_load(yaml_content)
        
        # Create config object
        return cls(**config_dict)
    
    def validate_config(self) -> None:
        """Validate the configuration"""
        # Check if Hochschul LLM is configured when it's the primary provider
        if self.llm.provider == "hochschul" and not self.llm.hochschul:
            # Try to create it from environment variables
            endpoint = os.environ.get("HOCHSCHUL_LLM_ENDPOINT")
            api_key = os.environ.get("HOCHSCHUL_LLM_API_KEY")
            
            if endpoint and api_key:
                self.llm.hochschul = HochschulLLMConfig(
                    endpoint=endpoint,
                    api_key=api_key
                )
            else:
                raise ValueError(
                    "Hochschul LLM is configured as primary provider but "
                    "HOCHSCHUL_LLM_ENDPOINT and HOCHSCHUL_LLM_API_KEY are not set"
                )


_config: Optional[Config] = None


def get_config(config_path: Optional[Path] = None) -> Config:
    """Get or create configuration singleton"""
    global _config
    
    if _config is None:
        if config_path is None:
            config_path = Path("config/default.yaml")
        
        if config_path.exists():
            _config = Config.from_yaml(config_path)
        else:
            _config = Config()
        
        _config.validate_config()
    
    return _config