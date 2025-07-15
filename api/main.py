"""FastAPI main application for Generic Knowledge Graph Pipeline System"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.config_new.unified_manager import get_config
from core.config_new.hot_reload import enable_hot_reload, disable_hot_reload
from api.routers import documents, pipeline, query, health

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SystemInfo(BaseModel):
    """System information response model"""
    name: str
    version: str
    description: str
    gpu_workloads: Dict[str, str]
    supported_formats: List[str]
    services_status: Dict[str, str]


class ProcessingStatus(BaseModel):
    """Document processing status response model"""
    document_id: str
    status: str  # pending, processing, completed, failed
    progress: float  # 0-100
    stage: str
    message: Optional[str] = None
    created_at: str
    updated_at: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting Generic Knowledge Graph Pipeline System")
    config = get_config()
    
    # Configuration is automatically validated by Pydantic
    logger.info("Configuration loaded successfully")
    logger.info(f"Profile: {config.profile}")
    logger.info(f"Debug mode: {config.general.debug}")
    
    # Initialize services (placeholder for actual service initialization)
    logger.info("Initializing services...")
    
    # Enable hot-reload for configuration
    if config.general.debug:
        logger.info("🔄 Enabling config hot-reload (debug mode)")
        await enable_hot_reload(check_interval=3.0)  # Check every 3 seconds in debug
    else:
        logger.info("📌 Config hot-reload disabled (production mode)")
    
    # Check service connectivity
    services_status = await check_services_health()
    logger.info(f"Service status: {services_status}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Generic Knowledge Graph Pipeline System")
    
    # Disable hot-reload
    await disable_hot_reload()


async def check_services_health() -> Dict[str, str]:
    """Check health of all external services"""
    config = get_config()
    status = {}
    
    # Check vLLM SmolDocling (GPU 1)
    try:
        # TODO: Actual health check implementation
        status["vllm_smoldocling"] = "healthy"
    except Exception as e:
        logger.warning(f"vLLM SmolDocling health check failed: {e}")
        status["vllm_smoldocling"] = "unhealthy"
    
    # Check Hochschul-LLM (GPU 2)
    try:
        # TODO: Actual health check implementation
        status["hochschul_llm"] = "healthy"
    except Exception as e:
        logger.warning(f"Hochschul-LLM health check failed: {e}")
        status["hochschul_llm"] = "unhealthy"
    
    # Check Fuseki Triple Store
    try:
        # TODO: Actual health check implementation
        status["fuseki"] = "healthy"
    except Exception as e:
        logger.warning(f"Fuseki health check failed: {e}")
        status["fuseki"] = "unhealthy"
    
    # Check ChromaDB Vector Store
    try:
        # TODO: Actual health check implementation
        status["chromadb"] = "healthy"
    except Exception as e:
        logger.warning(f"ChromaDB health check failed: {e}")
        status["chromadb"] = "unhealthy"
    
    return status


# Create FastAPI application
app = FastAPI(
    title="Generic Knowledge Graph Pipeline System",
    description="""
    A flexible, plugin-based pipeline system for extracting knowledge graphs from documents.
    
    ## GPU Workload Separation
    
    **GPU 1 - vLLM SmolDocling:** Advanced PDF parsing and document understanding
    - Complex document layouts, tables, and images
    - High-throughput batch processing
    
    **GPU 2 - Hochschul-LLM:** Knowledge graph triple extraction  
    - Semantic understanding and relationship extraction
    - Qwen1.5-based high-performance inference
    
    ## Supported Document Formats
    - PDF (via vLLM SmolDocling)
    - DOCX (via native parsing)
    - XLSX (via native parsing)
    - TXT (via native parsing)
    """,
    version="0.1.0",
    contact={
        "name": "Knowledge Graph Pipeline Team",
        "url": "https://github.com/klapom/generic-kg-pipeline",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(documents.router, prefix="/documents", tags=["Documents"])
app.include_router(pipeline.router, prefix="/pipeline", tags=["Pipeline"])
app.include_router(query.router, prefix="/query", tags=["Query"])


@app.get("/", response_model=SystemInfo)
async def root():
    """Get system information and status"""
    config = get_config()
    services_status = await check_services_health()
    
    return SystemInfo(
        name="Generic Knowledge Graph Pipeline System",
        version="0.1.0",
        description="Flexible pipeline for extracting knowledge graphs from documents",
        gpu_workloads={
            "gpu_1_vllm_smoldocling": "PDF parsing and document understanding",
            "gpu_2_hochschul_llm": "Triple extraction and semantic analysis"
        },
        supported_formats=config.domain.enabled_formats,
        services_status=services_status
    )


@app.get("/info")
async def system_info():
    """Get detailed system information"""
    config = get_config()
    
    return {
        "system": {
            "name": "Generic Knowledge Graph Pipeline System",
            "version": "0.1.0",
            "python_version": "3.11+",
        },
        "configuration": {
            "domain": config.domain.name,
            "enabled_formats": config.domain.enabled_formats,
            "llm_provider": config.llm.provider,
            "fallback_provider": config.llm.fallback_provider,
        },
        "gpu_workloads": {
            "vllm_smoldocling": {
                "purpose": "PDF parsing and document understanding",
                "endpoint": config.parsing.pdf.vllm_endpoint,
                "gpu_optimization": config.parsing.pdf.gpu_optimization,
            },
            "hochschul_llm": {
                "purpose": "Triple extraction and semantic analysis",
                "model": config.llm.hochschul.model if config.llm.hochschul else "Not configured",
                "temperature": config.llm.hochschul.temperature if config.llm.hochschul else None,
            }
        },
        "storage": {
            "triple_store": {
                "type": config.storage.triple_store.type,
                "endpoint": config.storage.triple_store.endpoint,
                "dataset": config.storage.triple_store.dataset,
            },
            "vector_store": {
                "type": config.storage.vector_store.type,
                "endpoint": config.storage.vector_store.endpoint,
                "collection": config.storage.vector_store.collection,
            }
        },
        "processing": {
            "chunking": {
                "max_tokens": config.chunking.max_tokens,
                "overlap_ratio": config.chunking.overlap_ratio,
                "preserve_context": config.chunking.preserve_context,
            },
            "rag": {
                "similarity_threshold": config.rag.similarity_threshold,
                "max_context_chunks": config.rag.max_context_chunks,
                "embedding_model": config.rag.embedding_model,
            }
        }
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    logger.error(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )


def main():
    """Main entry point for the application"""
    config = get_config()
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()