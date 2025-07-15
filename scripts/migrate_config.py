#!/usr/bin/env python3
"""
Migriere alte Konfigurationsdateien zur neuen einheitlichen Struktur
"""
import yaml
import json
from pathlib import Path
from typing import Dict, Any
import shutil
from datetime import datetime


def load_yaml(path: str) -> Dict[str, Any]:
    """Lade YAML-Datei"""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_env(path: str) -> Dict[str, str]:
    """Lade .env.example"""
    env_vars = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                env_vars[key] = value
    return env_vars


def migrate_configs():
    """Migriere alle alten Konfigurationen zur neuen Struktur"""
    
    print("🔄 Starting configuration migration...")
    
    # 1. Backup erstellen
    backup_dir = Path(f"config/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    backup_dir.mkdir(exist_ok=True)
    
    # Backup alte Dateien
    files_to_backup = [
        "config/default.yaml",
        "config/chunking.yaml",
        ".env.example"
    ]
    
    for file in files_to_backup:
        if Path(file).exists():
            shutil.copy(file, backup_dir / Path(file).name)
            print(f"✓ Backed up {file}")
    
    # 2. Lade alte Konfigurationen
    old_default = load_yaml("config/default.yaml") if Path("config/default.yaml").exists() else {}
    old_chunking = load_yaml("config/chunking.yaml") if Path("config/chunking.yaml").exists() else {}
    env_vars = load_env(".env.example") if Path(".env.example").exists() else {}
    
    # 3. Erstelle neue Konfiguration
    new_config = {
        "profile": "${PROFILE:dev}",
        
        "general": {
            "name": "Generic Knowledge Graph Pipeline",
            "version": "1.0.0",
            "debug": "${DEBUG:false}",
            "log_level": "${LOG_LEVEL:INFO}"
        },
        
        "domain": {
            "name": "${DOMAIN_NAME:" + old_default.get('domain', {}).get('name', 'general') + "}",
            "ontology_path": "plugins/ontologies/${domain.name}.ttl",
            "enabled_formats": old_default.get('domain', {}).get('formats', {}).get('enabled', 
                ["pdf", "docx", "xlsx", "pptx", "txt"])
        },
        
        "services": {
            "vllm": {
                "url": "${VLLM_URL:" + old_default.get('vllm', {}).get('base_url', 'http://localhost:8002') + "}",
                "timeout": 300,
                "health_check_enabled": True,
                "retry_attempts": 3
            },
            "hochschul_llm": {
                "url": "${HOCHSCHUL_LLM_URL:" + old_default.get('llm', {}).get('hochschul', {}).get('base_url', 'http://localhost:8001') + "}",
                "api_key": "${HOCHSCHUL_LLM_API_KEY:}",
                "timeout": old_default.get('llm', {}).get('hochschul', {}).get('timeout', 60),
                "model": old_default.get('llm', {}).get('hochschul', {}).get('model', 'meta-llama/Meta-Llama-3.1-8B-Instruct')
            },
            "fuseki": {
                "url": "${FUSEKI_URL:" + old_default.get('storage', {}).get('triple_store', {}).get('endpoint', 'http://localhost:3030') + "}",
                "dataset": "${FUSEKI_DATASET:" + old_default.get('storage', {}).get('triple_store', {}).get('dataset', 'kg_dataset') + "}",
                "timeout": 30
            },
            "chromadb": {
                "url": "${CHROMADB_URL:" + old_default.get('storage', {}).get('vector_store', {}).get('host', 'http://localhost:8000') + "}",
                "collection": "${CHROMADB_COLLECTION:" + old_default.get('storage', {}).get('vector_store', {}).get('collection', 'documents') + "}"
            },
            "ollama": {
                "url": "${OLLAMA_URL:" + old_default.get('llm', {}).get('ollama', {}).get('base_url', 'http://localhost:11434') + "}",
                "model": "${OLLAMA_MODEL:" + old_default.get('llm', {}).get('ollama', {}).get('model', 'llama2') + "}",
                "enabled": "${OLLAMA_ENABLED:false}"
            }
        },
        
        "models": {
            "llm": {
                "provider": "${LLM_PROVIDER:" + old_default.get('llm', {}).get('provider', 'hochschul') + "}",
                "temperature": "${LLM_TEMPERATURE:" + str(old_default.get('llm', {}).get('temperature', 0.1)) + "}",
                "max_tokens": "${LLM_MAX_TOKENS:" + str(old_default.get('llm', {}).get('max_tokens', 4000)) + "}",
                "top_p": "${LLM_TOP_P:0.9}",
                "stream": False
            },
            "vision": {
                "smoldocling": {
                    "enabled": "${SMOLDOCLING_ENABLED:true}",
                    "model_id": "numinamath/SmolDocling-256M-Preview",
                    "gpu_memory_utilization": "${SMOLDOCLING_GPU_MEMORY:0.2}",
                    "max_pages": "${SMOLDOCLING_MAX_PAGES:15}",
                    "dtype": "float16",
                    "trust_remote_code": True
                },
                "qwen_vl": {
                    "enabled": "${QWEN_VL_ENABLED:true}",
                    "model_id": "Qwen/Qwen2-VL-7B-Instruct",
                    "gpu_memory_utilization": "${QWEN_VL_GPU_MEMORY:0.8}",
                    "max_image_size": 1024
                }
            }
        },
        
        "parsing": {
            "pdf": {
                "provider": "${PDF_PARSER:" + old_default.get('parsing', {}).get('pdf', {}).get('provider', 'hybrid') + "}",
                "pdfplumber_mode": "${PDFPLUMBER_MODE:1}",
                "layout": {
                    "use_layout": True,
                    "table_x_tolerance": 3,
                    "table_y_tolerance": 3,
                    "text_x_tolerance": 5,
                    "text_y_tolerance": 5
                },
                "complex_detection": {
                    "enabled": True,
                    "min_text_blocks": 2,
                    "min_tables": 1,
                    "coverage_threshold": 0.8
                },
                "table_extraction": {
                    "enabled": True,
                    "separate_tables": True,
                    "preserve_structure": True
                }
            },
            "office": {
                "preserve_formatting": True,
                "extract_images": True,
                "extract_tables": True
            },
            "common": {
                "max_file_size_mb": "${MAX_FILE_SIZE:100}",
                "timeout_seconds": 120,
                "encoding": "utf-8",
                "language_detection": True
            }
        },
        
        "chunking": old_chunking if old_chunking else {
            "default": {
                "strategy": "${CHUNKING_STRATEGY:semantic}",
                "chunk_size": "${CHUNK_SIZE:1000}",
                "chunk_overlap": "${CHUNK_OVERLAP:200}"
            },
            "strategies": {
                "pdf": {
                    "preserve_tables": True,
                    "table_as_single_chunk": True,
                    "respect_page_boundaries": True,
                    "max_chunk_size": 1500
                }
            },
            "context": {
                "inherit_metadata": True,
                "max_inheritance_depth": 3,
                "include_visual_context": True
            }
        },
        
        "triples": {
            "generation": {
                "batch_size": "${TRIPLE_BATCH_SIZE:10}",
                "context_window": 2000,
                "include_metadata": True,
                "use_coreferences": True
            },
            "extraction": {
                "patterns": ["entity_relation_entity", "subject_predicate_object"],
                "confidence_threshold": 0.7,
                "min_triple_length": 3
            },
            "validation": {
                "check_ontology": True,
                "allow_new_predicates": False
            }
        },
        
        "storage": {
            "directories": {
                "output": "${OUTPUT_DIR:" + old_default.get('storage', {}).get('output_dir', 'data/output') + "}",
                "processed": "${PROCESSED_DIR:" + old_default.get('storage', {}).get('processed_dir', 'data/processed') + "}",
                "temp": "${TEMP_DIR:/tmp/kg_pipeline}",
                "logs": "${LOG_DIR:logs}"
            },
            "files": {
                "keep_processed": True,
                "compression": False,
                "timestamp_format": "%Y%m%d_%H%M%S"
            }
        },
        
        "batch": {
            "processing": {
                "max_workers": "${MAX_WORKERS:4}",
                "queue_size": 100,
                "chunk_timeout": 300
            },
            "retry": {
                "max_attempts": 3,
                "delay_seconds": 5,
                "exponential_backoff": True
            },
            "resources": {
                "memory_limit_gb": 8,
                "cpu_limit": 4
            }
        },
        
        "templates": {
            "paths": {
                "base": "plugins/templates",
                "custom": "${CUSTOM_TEMPLATES_PATH:}"
            },
            "active": {
                "triple_extraction": old_default.get('templates', {}).get('prompts', {}).get('triple_extraction', 'kg_extraction_prompt.txt'),
                "context_generation": "context_generation.txt",
                "task_with_context": "task_with_context.txt"
            }
        },
        
        "features": {
            "visual_analysis": "${ENABLE_VISUAL_ANALYSIS:true}",
            "table_extraction": "${ENABLE_TABLE_EXTRACTION:true}",
            "context_enhancement": "${ENABLE_CONTEXT:true}",
            "rag_generation": "${ENABLE_RAG:false}",
            "multi_language": "${ENABLE_MULTILANG:false}",
            "caching": "${ENABLE_CACHING:true}",
            "parallel_processing": "${ENABLE_PARALLEL:true}"
        },
        
        "monitoring": {
            "metrics": {
                "enabled": "${METRICS_ENABLED:false}",
                "port": "${METRICS_PORT:9090}",
                "interval_seconds": 30
            },
            "health": {
                "enabled": True,
                "interval_seconds": 60,
                "timeout_seconds": 10
            },
            "logging": {
                "structured": True,
                "include_context": True,
                "max_file_size_mb": 100,
                "retention_days": 7
            }
        },
        
        "api": {
            "server": {
                "host": "${API_HOST:0.0.0.0}",
                "port": "${API_PORT:8000}",
                "workers": "${API_WORKERS:4}"
            },
            "security": {
                "cors_enabled": True,
                "cors_origins": ["*"],
                "rate_limiting": False
            },
            "docs": {
                "enabled": True,
                "path": "/docs"
            }
        }
    }
    
    # 4. Speichere neue Konfiguration
    output_path = Path("config.yaml")
    with open(output_path, 'w') as f:
        yaml.dump(new_config, f, default_flow_style=False, sort_keys=False, width=120)
    
    print(f"✓ Created unified configuration: {output_path}")
    
    # 5. Erstelle neue .env.example
    new_env = []
    new_env.append("# Generic Knowledge Graph Pipeline - Environment Variables")
    new_env.append("# Generated from migration script")
    new_env.append("")
    new_env.append("# Profile")
    new_env.append("PROFILE=dev")
    new_env.append("")
    new_env.append("# Services")
    new_env.append("VLLM_URL=http://localhost:8002")
    new_env.append("HOCHSCHUL_LLM_URL=http://localhost:8001")
    new_env.append("HOCHSCHUL_LLM_API_KEY=")
    new_env.append("FUSEKI_URL=http://localhost:3030")
    new_env.append("CHROMADB_URL=http://localhost:8000")
    new_env.append("OLLAMA_URL=http://localhost:11434")
    new_env.append("")
    new_env.append("# Model Settings")
    new_env.append("LLM_PROVIDER=hochschul")
    new_env.append("SMOLDOCLING_GPU_MEMORY=0.2")
    new_env.append("QWEN_VL_GPU_MEMORY=0.8")
    new_env.append("")
    new_env.append("# Directories")
    new_env.append("OUTPUT_DIR=data/output")
    new_env.append("PROCESSED_DIR=data/processed")
    new_env.append("")
    
    with open(".env.example.new", 'w') as f:
        f.write('\n'.join(new_env))
    
    print("✓ Created new .env.example")
    
    # 6. Report
    print("\n📋 Migration Summary:")
    print(f"  - Backed up old configs to: {backup_dir}")
    print(f"  - Created unified config: config.yaml")
    print(f"  - Created new .env.example")
    print("\n⚠️  Next steps:")
    print("  1. Review the generated config.yaml")
    print("  2. Update your .env file with values from .env.example.new")
    print("  3. Test with: python -c 'from core.config.unified_manager import get_config; print(get_config())'")
    print("  4. Update imports in code from 'core.config' to 'core.config.unified_manager'")


if __name__ == "__main__":
    migrate_configs()