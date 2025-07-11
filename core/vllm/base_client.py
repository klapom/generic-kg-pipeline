"""
Base vLLM Client

Abstrakte Basis für alle vLLM-basierten Clients mit gemeinsamer Funktionalität
für Performance Monitoring, Error Handling und Model Management.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, AsyncGenerator

from .model_manager import VLLMModelManager, ModelConfig, SamplingConfig, model_manager
from .exceptions import ModelNotLoadedError, InferenceError, ConfigurationError

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """Request for vLLM inference"""
    prompts: Union[str, List[str], List[Dict[str, Any]]]
    sampling_config: Optional[SamplingConfig] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class InferenceResult:
    """Result from vLLM inference"""
    outputs: List[Any]
    processing_time: float
    token_count: Optional[int] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceMetrics:
    """Performance tracking for vLLM operations"""
    operation_name: str
    start_time: float
    end_time: float
    processing_time: float
    input_size: int
    output_size: int
    success: bool
    error_message: Optional[str] = None
    
    @property
    def throughput(self) -> float:
        """Items processed per second"""
        if self.processing_time == 0:
            return 0.0
        return self.input_size / self.processing_time


class BaseVLLMClient(ABC):
    """
    Abstract base class for vLLM-based clients
    
    Provides common functionality:
    - Model lifecycle management
    - Performance monitoring
    - Error handling
    - Async context management
    """
    
    def __init__(
        self,
        model_id: str,
        model_config: ModelConfig,
        sampling_config: Optional[SamplingConfig] = None,
        auto_load: bool = False
    ):
        self.model_id = model_id
        self.model_config = model_config
        self.sampling_config = sampling_config or SamplingConfig()
        self.model_manager = model_manager
        self.performance_metrics: List[PerformanceMetrics] = []
        self._initialized = False
        
        # Register model configuration
        self.model_manager.register_model(
            self.model_id,
            self.model_config,
            self.sampling_config
        )
        
        if auto_load:
            self.ensure_model_loaded()
        
        logger.info(f"Initialized {self.__class__.__name__} for model: {model_id}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.ensure_model_loaded_async()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        # Note: We don't automatically unload here as other clients might be using the model
        # Use explicit cleanup or batch context for unloading
        pass
    
    def ensure_model_loaded(self) -> bool:
        """Ensure the model is loaded and ready"""
        try:
            if not self.model_manager.is_model_loaded(self.model_id):
                success = self.model_manager.load_model(self.model_id)
                if success:
                    # Skip warmup to avoid hanging with multimodal models
                    logger.info("⚡ Skipping warmup for faster startup")
                    # self.model_manager.warmup_model(self.model_id, self.get_warmup_prompts())
                return success
            return True
        except Exception as e:
            logger.error(f"Failed to ensure model loaded: {e}")
            return False
    
    async def ensure_model_loaded_async(self) -> bool:
        """Async version of ensure_model_loaded"""
        # Run in thread pool since vLLM operations are blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.ensure_model_loaded)
    
    def get_warmup_prompts(self) -> List[str]:
        """Get model-specific warmup prompts. Override in subclasses."""
        return ["This is a warmup prompt."]
    
    def track_performance(self, func):
        """Decorator for tracking performance metrics"""
        def wrapper(*args, **kwargs):
            operation_name = f"{self.__class__.__name__}.{func.__name__}"
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                
                # Extract size information
                input_size = self._get_input_size(args, kwargs)
                output_size = self._get_output_size(result)
                
                metrics = PerformanceMetrics(
                    operation_name=operation_name,
                    start_time=start_time,
                    end_time=end_time,
                    processing_time=end_time - start_time,
                    input_size=input_size,
                    output_size=output_size,
                    success=True
                )
                
                self.performance_metrics.append(metrics)
                
                logger.debug(f"{operation_name}: {metrics.processing_time:.2f}s "
                           f"({metrics.throughput:.1f} items/s)")
                
                return result
                
            except Exception as e:
                end_time = time.time()
                
                metrics = PerformanceMetrics(
                    operation_name=operation_name,
                    start_time=start_time,
                    end_time=end_time,
                    processing_time=end_time - start_time,
                    input_size=self._get_input_size(args, kwargs),
                    output_size=0,
                    success=False,
                    error_message=str(e)
                )
                
                self.performance_metrics.append(metrics)
                raise
                
        return wrapper
    
    def _get_input_size(self, args, kwargs) -> int:
        """Extract input size from function arguments. Override in subclasses."""
        return 1
    
    def _get_output_size(self, result) -> int:
        """Extract output size from function result. Override in subclasses."""
        if isinstance(result, list):
            return len(result)
        return 1
    
    def generate(
        self,
        request: InferenceRequest
    ) -> InferenceResult:
        """
        Generate using vLLM model
        
        Args:
            request: Inference request with prompts and config
            
        Returns:
            Inference result with outputs and metrics
        """
        if not self.model_manager.is_model_loaded(self.model_id):
            raise ModelNotLoadedError(f"Model {self.model_id} not loaded")
        
        start_time = time.time()
        
        try:
            # Use request sampling config or default
            sampling_config = request.sampling_config or self.sampling_config
            
            # Call model manager for generation
            outputs = self.model_manager.generate(
                self.model_id,
                request.prompts,
                sampling_config
            )
            
            processing_time = time.time() - start_time
            
            # Post-process outputs (override in subclasses)
            processed_outputs = self.post_process_outputs(outputs)
            
            return InferenceResult(
                outputs=processed_outputs,
                processing_time=processing_time,
                success=True,
                metadata=request.metadata
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Generation failed: {e}")
            
            return InferenceResult(
                outputs=[],
                processing_time=processing_time,
                success=False,
                error_message=str(e),
                metadata=request.metadata
            )
    
    async def generate_async(self, request: InferenceRequest) -> InferenceResult:
        """Async wrapper for generate"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, request)
    
    def post_process_outputs(self, outputs: List[Any]) -> List[Any]:
        """Post-process vLLM outputs. Override in subclasses."""
        return outputs
    
    @abstractmethod
    def parse_model_output(self, output: Any) -> Any:
        """Parse model-specific output format. Must be implemented by subclasses."""
        pass
    
    def batch_generate(
        self,
        requests: List[InferenceRequest],
        batch_size: Optional[int] = None
    ) -> List[InferenceResult]:
        """
        Generate for multiple requests efficiently
        
        Args:
            requests: List of inference requests
            batch_size: Optional batch size for processing
            
        Returns:
            List of inference results
        """
        if not requests:
            return []
        
        batch_size = batch_size or len(requests)
        results = []
        
        logger.info(f"Processing {len(requests)} requests in batches of {batch_size}")
        
        for i in range(0, len(requests), batch_size):
            batch = requests[i:i + batch_size]
            
            logger.debug(f"Processing batch {i//batch_size + 1}: {len(batch)} requests")
            
            # Process batch
            batch_results = []
            for request in batch:
                result = self.generate(request)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        successful = sum(1 for r in results if r.success)
        logger.info(f"Batch processing completed: {successful}/{len(results)} successful")
        
        return results
    
    async def batch_generate_async(
        self,
        requests: List[InferenceRequest],
        batch_size: Optional[int] = None,
        max_concurrent: int = 3
    ) -> List[InferenceResult]:
        """
        Async batch generation with concurrency control
        
        Args:
            requests: List of inference requests
            batch_size: Optional batch size
            max_concurrent: Maximum concurrent operations
            
        Returns:
            List of inference results
        """
        if not requests:
            return []
        
        batch_size = batch_size or max_concurrent
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_request(request: InferenceRequest) -> InferenceResult:
            async with semaphore:
                return await self.generate_async(request)
        
        # Process in batches
        all_results = []
        for i in range(0, len(requests), batch_size):
            batch = requests[i:i + batch_size]
            
            tasks = [process_request(req) for req in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Batch request {i+j} failed: {result}")
                    batch_results[j] = InferenceResult(
                        outputs=[],
                        processing_time=0.0,
                        success=False,
                        error_message=str(result)
                    )
            
            all_results.extend(batch_results)
        
        successful = sum(1 for r in all_results if r.success)
        logger.info(f"Async batch processing completed: {successful}/{len(all_results)} successful")
        
        return all_results
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the model
        
        Returns:
            Health status information
        """
        try:
            # Check if model is loaded
            is_loaded = self.model_manager.is_model_loaded(self.model_id)
            
            if not is_loaded:
                return {
                    "status": "unhealthy",
                    "model_id": self.model_id,
                    "error": "Model not loaded",
                    "last_check": datetime.now().isoformat()
                }
            
            # Test inference
            test_request = InferenceRequest(
                prompts=["Health check test"],
                sampling_config=SamplingConfig(max_tokens=10, temperature=0)
            )
            
            result = self.generate(test_request)
            
            if result.success:
                return {
                    "status": "healthy",
                    "model_id": self.model_id,
                    "response_time": result.processing_time,
                    "model_loaded": True,
                    "last_check": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "unhealthy",
                    "model_id": self.model_id,
                    "error": result.error_message,
                    "last_check": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "model_id": self.model_id,
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        if not self.performance_metrics:
            return {"no_data": True}
        
        successful_metrics = [m for m in self.performance_metrics if m.success]
        failed_metrics = [m for m in self.performance_metrics if not m.success]
        
        if successful_metrics:
            avg_time = sum(m.processing_time for m in successful_metrics) / len(successful_metrics)
            avg_throughput = sum(m.throughput for m in successful_metrics) / len(successful_metrics)
            
            return {
                "total_operations": len(self.performance_metrics),
                "successful_operations": len(successful_metrics),
                "failed_operations": len(failed_metrics),
                "success_rate": len(successful_metrics) / len(self.performance_metrics),
                "average_processing_time": avg_time,
                "average_throughput": avg_throughput,
                "model_id": self.model_id
            }
        else:
            return {
                "total_operations": len(self.performance_metrics),
                "successful_operations": 0,
                "failed_operations": len(failed_metrics),
                "success_rate": 0.0,
                "model_id": self.model_id
            }
    
    @asynccontextmanager
    async def batch_context(self):
        """Context manager for batch operations with automatic model management"""
        try:
            await self.ensure_model_loaded_async()
            logger.info(f"Batch context ready for {self.model_id}")
            yield self
        finally:
            # Note: Model cleanup is handled by VLLMModelManager batch_context
            logger.debug(f"Batch context finished for {self.model_id}")
    
    def cleanup(self):
        """Cleanup resources"""
        # Note: Actual model unloading should be coordinated through VLLMModelManager
        self.performance_metrics.clear()
        logger.info(f"Cleaned up {self.__class__.__name__} for {self.model_id}")


