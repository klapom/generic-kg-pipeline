"""Health check endpoints"""

import logging
from datetime import datetime
from typing import Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.config_new.unified_manager import get_config
from core.clients.vllm_smoldocling import VLLMSmolDoclingClient
from core.clients.hochschul_llm import HochschulLLMClient

logger = logging.getLogger(__name__)

router = APIRouter()


class HealthStatus(BaseModel):
    """Health status response model"""
    status: str
    timestamp: datetime
    version: str
    services: Dict[str, Dict[str, str]]


class ServiceHealth(BaseModel):
    """Individual service health model"""
    status: str
    response_time_ms: float
    endpoint: str
    last_check: datetime
    details: Dict[str, str] = {}


@router.get("/", response_model=HealthStatus)
async def health_check():
    """Basic health check endpoint"""
    config = get_config()
    
    # Check all services
    services = {}
    
    # vLLM SmolDocling (GPU 1)
    try:
        async with VLLMSmolDoclingClient() as vllm_client:
            health_info = await vllm_client.health_check()
            services["vllm_smoldocling"] = {
                "status": health_info["status"],
                "endpoint": config.services.vllm.url,
                "purpose": "PDF parsing (GPU 1)",
                "response_time_ms": health_info.get("response_time_ms", 0),
                "model_info": health_info.get("model_info", {}),
                "gpu_info": health_info.get("gpu_info", {})
            }
    except Exception as e:
        logger.error(f"vLLM SmolDocling health check failed: {e}")
        services["vllm_smoldocling"] = {
            "status": "unhealthy",
            "endpoint": config.services.vllm.url,
            "error": str(e)
        }
    
    # Hochschul-LLM (GPU 2)
    try:
        hochschul_client = HochschulLLMClient()
        health_info = await hochschul_client.health_check()
        
        if config.models.llm.provider == "hochschul":
            services["hochschul_llm"] = {
                "status": health_info["status"],
                "endpoint": config.services.hochschul_llm.url,
                "model": config.services.hochschul_llm.model,
                "purpose": "Triple extraction (GPU 2)",
                "message": health_info.get("message", "")
            }
        else:
            services["hochschul_llm"] = {
                "status": "not_configured",
                "message": "Hochschul-LLM credentials not provided"
            }
    except Exception as e:
        logger.error(f"Hochschul-LLM health check failed: {e}")
        services["hochschul_llm"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # Fuseki Triple Store
    try:
        # TODO: Implement actual health check
        services["fuseki"] = {
            "status": "healthy",
            "endpoint": config.services.fuseki.url,
            "purpose": "RDF triple storage"
        }
    except Exception as e:
        logger.error(f"Fuseki health check failed: {e}")
        services["fuseki"] = {
            "status": "unhealthy", 
            "error": str(e)
        }
    
    # ChromaDB Vector Store  
    try:
        # TODO: Implement actual health check
        services["chromadb"] = {
            "status": "healthy",
            "endpoint": config.services.chromadb.url,
            "purpose": "Vector embeddings storage"
        }
    except Exception as e:
        logger.error(f"ChromaDB health check failed: {e}")
        services["chromadb"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # Ollama (Fallback)
    try:
        # TODO: Implement actual health check
        services["ollama"] = {
            "status": "healthy",
            "endpoint": config.llm.ollama.endpoint,
            "purpose": "Fallback LLM for development"
        }
    except Exception as e:
        logger.error(f"Ollama health check failed: {e}")
        services["ollama"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # Determine overall status
    unhealthy_services = [name for name, info in services.items() 
                         if info.get("status") not in ["healthy", "not_configured"]]
    
    overall_status = "healthy" if not unhealthy_services else "degraded"
    
    return HealthStatus(
        status=overall_status,
        timestamp=datetime.now(),
        version="0.1.0",
        services=services
    )


@router.get("/liveness")
async def liveness_probe():
    """Kubernetes liveness probe endpoint"""
    return {"status": "alive", "timestamp": datetime.now()}


@router.get("/readiness") 
async def readiness_probe():
    """Kubernetes readiness probe endpoint"""
    config = get_config()
    
    # Check critical services for readiness
    critical_services = []
    
    # Check if configuration is valid
    try:
        config.validate_config()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Configuration invalid: {e}")
    
    # TODO: Check if critical services are responding
    # For now, just return ready
    
    return {
        "status": "ready",
        "timestamp": datetime.now(),
        "critical_services": critical_services
    }


@router.get("/services/{service_name}")
async def service_health(service_name: str):
    """Get detailed health information for a specific service"""
    config = get_config()
    
    service_configs = {
        "vllm_smoldocling": {
            "endpoint": config.services.vllm.url,
            "purpose": "PDF parsing with vLLM SmolDocling (GPU 1)",
        },
        "hochschul_llm": {
            "endpoint": config.services.hochschul_llm.url if config.models.llm.provider == "hochschul" else None,
            "purpose": "Triple extraction with Hochschul-LLM (GPU 2)",
        },
        "fuseki": {
            "endpoint": config.services.fuseki.url,
            "purpose": "RDF triple storage with Apache Jena Fuseki",
        },
        "chromadb": {
            "endpoint": config.services.chromadb.url,
            "purpose": "Vector embeddings storage with ChromaDB",
        },
        "ollama": {
            "endpoint": config.llm.ollama.endpoint,
            "purpose": "Fallback LLM with Ollama",
        }
    }
    
    if service_name not in service_configs:
        raise HTTPException(
            status_code=404,
            detail=f"Service '{service_name}' not found. Available: {list(service_configs.keys())}"
        )
    
    service_config = service_configs[service_name]
    
    # TODO: Implement actual detailed health check for the service
    
    return ServiceHealth(
        status="healthy",  # TODO: Actual status
        response_time_ms=0.0,  # TODO: Actual response time
        endpoint=service_config["endpoint"] or "Not configured",
        last_check=datetime.now(),
        details={
            "purpose": service_config["purpose"],
            "configured": str(service_config["endpoint"] is not None)
        }
    )