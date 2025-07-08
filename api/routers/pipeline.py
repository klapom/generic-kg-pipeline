"""Pipeline management and control endpoints"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.config import get_config

logger = logging.getLogger(__name__)

router = APIRouter()


class PipelineConfig(BaseModel):
    """Pipeline configuration model"""
    domain: str
    enabled_formats: List[str]
    gpu_workloads: Dict[str, str]
    llm_provider: str
    fallback_provider: str
    chunking_config: Dict[str, float]
    storage_config: Dict[str, Dict[str, str]]


class PipelineStats(BaseModel):
    """Pipeline statistics model"""
    total_documents_processed: int
    documents_by_status: Dict[str, int]
    documents_by_format: Dict[str, int]
    total_triples_extracted: int
    total_chunks_created: int
    average_processing_time_seconds: float
    gpu_utilization: Dict[str, Dict[str, float]]


class PipelineOperation(BaseModel):
    """Pipeline operation request model"""
    operation: str
    parameters: Dict = {}


@router.get("/config", response_model=PipelineConfig)
async def get_pipeline_config():
    """Get current pipeline configuration"""
    config = get_config()
    
    return PipelineConfig(
        domain=config.domain.name,
        enabled_formats=config.domain.enabled_formats,
        gpu_workloads={
            "gpu_1_vllm_smoldocling": "PDF parsing and document understanding",
            "gpu_2_hochschul_llm": "Triple extraction and semantic analysis"
        },
        llm_provider=config.llm.provider,
        fallback_provider=config.llm.fallback_provider,
        chunking_config={
            "max_tokens": float(config.chunking.max_tokens),
            "overlap_ratio": config.chunking.overlap_ratio
        },
        storage_config={
            "triple_store": {
                "type": config.storage.triple_store.type,
                "endpoint": config.storage.triple_store.endpoint,
                "dataset": config.storage.triple_store.dataset
            },
            "vector_store": {
                "type": config.storage.vector_store.type,
                "endpoint": config.storage.vector_store.endpoint,
                "collection": config.storage.vector_store.collection
            }
        }
    )


@router.get("/stats", response_model=PipelineStats)
async def get_pipeline_stats():
    """Get pipeline processing statistics"""
    # TODO: Implement actual statistics collection
    # For now, return placeholder data
    
    return PipelineStats(
        total_documents_processed=0,
        documents_by_status={
            "uploaded": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0
        },
        documents_by_format={
            "pdf": 0,
            "docx": 0,
            "xlsx": 0,
            "txt": 0
        },
        total_triples_extracted=0,
        total_chunks_created=0,
        average_processing_time_seconds=0.0,
        gpu_utilization={
            "vllm_smoldocling": {
                "memory_usage_percent": 0.0,
                "gpu_usage_percent": 0.0,
                "temperature_celsius": 0.0
            },
            "hochschul_llm": {
                "memory_usage_percent": 0.0,
                "gpu_usage_percent": 0.0,
                "temperature_celsius": 0.0
            }
        }
    )


@router.get("/status")
async def get_pipeline_status():
    """Get overall pipeline status and health"""
    config = get_config()
    
    # Check service connectivity
    services_status = {}
    
    # vLLM SmolDocling (GPU 1)
    try:
        # TODO: Actual service check
        services_status["vllm_smoldocling"] = {
            "status": "healthy",
            "endpoint": config.parsing.pdf.vllm_endpoint,
            "purpose": "PDF parsing (GPU 1)",
            "last_check": datetime.now().isoformat()
        }
    except Exception as e:
        services_status["vllm_smoldocling"] = {
            "status": "unhealthy",
            "error": str(e),
            "last_check": datetime.now().isoformat()
        }
    
    # Hochschul-LLM (GPU 2)
    try:
        # TODO: Actual service check
        if config.llm.hochschul:
            services_status["hochschul_llm"] = {
                "status": "healthy",
                "endpoint": config.llm.hochschul.endpoint,
                "model": config.llm.hochschul.model,
                "purpose": "Triple extraction (GPU 2)",
                "last_check": datetime.now().isoformat()
            }
        else:
            services_status["hochschul_llm"] = {
                "status": "not_configured",
                "message": "Hochschul-LLM credentials not provided"
            }
    except Exception as e:
        services_status["hochschul_llm"] = {
            "status": "unhealthy",
            "error": str(e),
            "last_check": datetime.now().isoformat()
        }
    
    # Storage services
    services_status["fuseki"] = {
        "status": "healthy",  # TODO: Actual check
        "endpoint": config.storage.triple_store.endpoint,
        "purpose": "RDF triple storage",
        "last_check": datetime.now().isoformat()
    }
    
    services_status["chromadb"] = {
        "status": "healthy",  # TODO: Actual check
        "endpoint": config.storage.vector_store.endpoint,
        "purpose": "Vector embeddings storage",
        "last_check": datetime.now().isoformat()
    }
    
    # Determine overall status
    unhealthy_services = [
        name for name, info in services_status.items()
        if info.get("status") not in ["healthy", "not_configured"]
    ]
    
    overall_status = "healthy" if not unhealthy_services else "degraded"
    
    return {
        "overall_status": overall_status,
        "timestamp": datetime.now().isoformat(),
        "services": services_status,
        "gpu_workloads": {
            "separation_enabled": True,
            "gpu_1": {
                "service": "vLLM SmolDocling",
                "purpose": "PDF parsing and document understanding",
                "status": services_status["vllm_smoldocling"]["status"]
            },
            "gpu_2": {
                "service": "Hochschul-LLM",
                "purpose": "Triple extraction and semantic analysis",
                "status": services_status["hochschul_llm"]["status"]
            }
        },
        "unhealthy_services": unhealthy_services
    }


@router.post("/operations")
async def execute_pipeline_operation(operation: PipelineOperation):
    """
    Execute pipeline operations
    
    Supported operations:
    - restart_services: Restart all pipeline services
    - clear_cache: Clear processing caches
    - rebuild_indexes: Rebuild vector indexes
    - validate_config: Validate current configuration
    """
    
    if operation.operation == "validate_config":
        try:
            config = get_config()
            config.validate_config()
            return {
                "operation": "validate_config",
                "status": "success",
                "message": "Configuration is valid",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "operation": "validate_config",
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    elif operation.operation == "restart_services":
        # TODO: Implement service restart logic
        return {
            "operation": "restart_services",
            "status": "not_implemented",
            "message": "Service restart functionality will be implemented",
            "timestamp": datetime.now().isoformat()
        }
    
    elif operation.operation == "clear_cache":
        # TODO: Implement cache clearing logic
        return {
            "operation": "clear_cache",
            "status": "not_implemented", 
            "message": "Cache clearing functionality will be implemented",
            "timestamp": datetime.now().isoformat()
        }
    
    elif operation.operation == "rebuild_indexes":
        # TODO: Implement index rebuilding logic
        return {
            "operation": "rebuild_indexes",
            "status": "not_implemented",
            "message": "Index rebuilding functionality will be implemented",
            "timestamp": datetime.now().isoformat()
        }
    
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown operation: {operation.operation}. "
                   f"Supported: validate_config, restart_services, clear_cache, rebuild_indexes"
        )


@router.get("/gpu-status")
async def get_gpu_status():
    """Get GPU utilization and status for both workloads"""
    # TODO: Implement actual GPU monitoring
    # This would typically use nvidia-ml-py or similar
    
    return {
        "gpu_1_vllm_smoldocling": {
            "gpu_id": 0,
            "name": "NVIDIA GPU 1",
            "memory_total_mb": 24576,  # Example: 24GB
            "memory_used_mb": 8192,
            "memory_free_mb": 16384,
            "utilization_percent": 45.0,
            "temperature_celsius": 72.0,
            "power_draw_watts": 180.0,
            "processes": [
                {
                    "pid": 12345,
                    "name": "vllm",
                    "memory_mb": 8192
                }
            ]
        },
        "gpu_2_hochschul_llm": {
            "status": "external",
            "note": "Hochschul-LLM runs on external infrastructure",
            "endpoint": "https://llm.hochschule.example/api",
            "last_request_latency_ms": 250.0,
            "requests_per_minute": 12.0
        },
        "timestamp": datetime.now().isoformat()
    }


@router.get("/performance")
async def get_performance_metrics():
    """Get detailed performance metrics for the pipeline"""
    return {
        "document_processing": {
            "throughput_docs_per_hour": 0.0,
            "average_processing_time_seconds": 0.0,
            "success_rate_percent": 100.0,
            "by_format": {
                "pdf": {
                    "average_time_seconds": 0.0,
                    "success_rate_percent": 100.0,
                    "gpu_utilization": "vLLM SmolDocling (GPU 1)"
                },
                "docx": {
                    "average_time_seconds": 0.0,
                    "success_rate_percent": 100.0,
                    "gpu_utilization": "CPU only"
                },
                "xlsx": {
                    "average_time_seconds": 0.0,
                    "success_rate_percent": 100.0,
                    "gpu_utilization": "CPU only"
                },
                "txt": {
                    "average_time_seconds": 0.0,
                    "success_rate_percent": 100.0,
                    "gpu_utilization": "CPU only"
                }
            }
        },
        "triple_extraction": {
            "triples_per_minute": 0.0,
            "accuracy_score": 0.0,
            "llm_provider": "Hochschul-LLM",
            "average_confidence": 0.0,
            "gpu_utilization": "External (GPU 2)"
        },
        "storage": {
            "triple_store_write_ops_per_second": 0.0,
            "vector_store_write_ops_per_second": 0.0,
            "query_response_time_ms": 0.0
        },
        "timestamp": datetime.now().isoformat()
    }