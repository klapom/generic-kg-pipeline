"""
VLLMModelManager - Zentrales Model Lifecycle Management

Verwaltet das Laden, Caching und Entladen von vLLM Models für optimale Performance.
Implementiert Singleton Pattern für effizienten Ressourcenverbrauch.
"""

import logging
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Any, List, Union
from pathlib import Path

from .exceptions import ModelNotLoadedError, GPUMemoryError, ModelLoadError, ConfigurationError

logger = logging.getLogger(__name__)

# Try to import vLLM and dependencies
try:
    import torch
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError as e:
    logger.warning(f"vLLM not available: {e}")
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None
    torch = None


@dataclass
class ModelConfig:
    """Configuration for a vLLM model"""
    model_name: str
    model_path: Optional[str] = None
    gpu_memory_utilization: float = 0.8
    max_model_len: int = 8192
    trust_remote_code: bool = True
    enforce_eager: bool = False
    limit_mm_per_prompt: Optional[Dict[str, int]] = None
    dtype: str = "auto"
    tensor_parallel_size: int = 1
    
    def to_vllm_kwargs(self) -> Dict[str, Any]:
        """Convert to vLLM LLM constructor kwargs"""
        kwargs = {
            "model": self.model_path or self.model_name,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "trust_remote_code": self.trust_remote_code,
            "enforce_eager": self.enforce_eager,
            "dtype": self.dtype,
            "tensor_parallel_size": self.tensor_parallel_size
        }
        
        if self.limit_mm_per_prompt:
            kwargs["limit_mm_per_prompt"] = self.limit_mm_per_prompt
            
        return {k: v for k, v in kwargs.items() if v is not None}


@dataclass 
class SamplingConfig:
    """Configuration for vLLM sampling parameters"""
    temperature: float = 0.1
    max_tokens: int = 8192
    top_p: float = 1.0
    top_k: int = -1
    stop: Optional[List[str]] = None
    
    def to_sampling_params(self) -> 'SamplingParams':
        """Convert to vLLM SamplingParams"""
        if not VLLM_AVAILABLE:
            raise ConfigurationError("vLLM not available")
        
        # Base parameters
        params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "stop": self.stop or ["</s>", "<|endoftext|>"]
        }
        
        # Add custom parameters if they exist
        if hasattr(self, '_custom_params'):
            params.update(self._custom_params)
            
        return SamplingParams(**params)


@dataclass
class ModelStats:
    """Statistics for model usage"""
    load_time: float = 0.0
    warmup_time: float = 0.0
    total_inference_time: float = 0.0
    inference_count: int = 0
    last_used: Optional[datetime] = None
    
    @property
    def average_inference_time(self) -> float:
        """Average inference time per request"""
        if self.inference_count == 0:
            return 0.0
        return self.total_inference_time / self.inference_count


class VLLMModelManager:
    """
    Singleton Model Manager für vLLM Models
    
    Verwaltet das Lifecycle von vLLM Models:
    - Loading mit GPU Memory Management
    - Caching für wiederholte Nutzung
    - Warmup für konsistente Performance
    - Cleanup für Resource Management
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        self._models: Dict[str, LLM] = {}
        self._configs: Dict[str, ModelConfig] = {}
        self._sampling_configs: Dict[str, SamplingConfig] = {}
        self._stats: Dict[str, ModelStats] = {}
        self._lock = threading.Lock()
        
        logger.info("Initialized VLLMModelManager singleton")
    
    def check_vllm_availability(self) -> bool:
        """Check if vLLM and GPU are available"""
        if not VLLM_AVAILABLE:
            return False
            
        if not torch.cuda.is_available():
            logger.warning("CUDA not available - vLLM requires GPU")
            return False
            
        return True
    
    def get_gpu_memory_info(self) -> Dict[str, Any]:
        """Get current GPU memory information"""
        if not torch.cuda.is_available():
            return {"available": False}
            
        gpu_count = torch.cuda.device_count()
        gpu_info = {
            "available": True,
            "device_count": gpu_count,
            "devices": []
        }
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            
            gpu_info["devices"].append({
                "device_id": i,
                "name": props.name,
                "total_memory": props.total_memory,
                "allocated_memory": allocated,
                "reserved_memory": reserved,
                "free_memory": props.total_memory - reserved,
                "utilization": reserved / props.total_memory
            })
        
        return gpu_info
    
    def optimize_gpu_memory_utilization(self, model_size: str = "medium") -> float:
        """Calculate optimal GPU memory utilization"""
        gpu_info = self.get_gpu_memory_info()
        
        if not gpu_info["available"] or not gpu_info["devices"]:
            return 0.4
            
        # Use first GPU for calculation
        device = gpu_info["devices"][0]
        total_gb = device["total_memory"] / (1024**3)
        
        # Model size categories
        size_map = {
            "small": {"min_gb": 4, "utilization": 0.8},    # <1B parameters
            "medium": {"min_gb": 8, "utilization": 0.7},   # 1-7B parameters  
            "large": {"min_gb": 16, "utilization": 0.6},   # 7-30B parameters
            "xlarge": {"min_gb": 32, "utilization": 0.5}   # >30B parameters
        }
        
        config = size_map.get(model_size, size_map["medium"])
        
        if total_gb < config["min_gb"]:
            logger.warning(f"GPU memory ({total_gb:.1f}GB) may be insufficient for {model_size} model")
            return max(0.4, config["utilization"] - 0.2)
        
        return config["utilization"]
    
    def register_model(
        self, 
        model_id: str, 
        model_config: ModelConfig,
        sampling_config: Optional[SamplingConfig] = None
    ):
        """Register a model configuration without loading it"""
        with self._lock:
            self._configs[model_id] = model_config
            self._sampling_configs[model_id] = sampling_config or SamplingConfig()
            self._stats[model_id] = ModelStats()
            
        logger.info(f"Registered model config: {model_id}")
    
    def load_model(self, model_id: str, force_reload: bool = False) -> bool:
        """
        Load a model into GPU memory
        
        Args:
            model_id: Unique identifier for the model
            force_reload: Force reload even if already loaded
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.check_vllm_availability():
            raise ConfigurationError("vLLM not available - GPU required")
            
        with self._lock:
            # Check if already loaded
            if not force_reload and model_id in self._models:
                logger.info(f"Model {model_id} already loaded")
                return True
            
            # Get configuration
            if model_id not in self._configs:
                raise ModelLoadError(f"Model {model_id} not registered")
            
            config = self._configs[model_id]
            stats = self._stats[model_id]
            
            logger.info(f"Loading vLLM model: {model_id}")
            logger.info(f"  Model: {config.model_name}")
            logger.info(f"  GPU Memory: {config.gpu_memory_utilization:.1%}")
            
            start_time = time.time()
            
            try:
                # Clear GPU memory before loading
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Load model with vLLM
                vllm_kwargs = config.to_vllm_kwargs()
                model = LLM(**vllm_kwargs)
                
                # Store loaded model
                self._models[model_id] = model
                
                # Update statistics
                load_time = time.time() - start_time
                stats.load_time = load_time
                
                logger.info(f"✅ Model {model_id} loaded successfully in {load_time:.1f}s")
                
                # Log GPU memory usage
                gpu_info = self.get_gpu_memory_info()
                if gpu_info["available"]:
                    device = gpu_info["devices"][0]
                    util = device["utilization"]
                    logger.info(f"   GPU Memory Utilization: {util:.1%}")
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to load model {model_id}: {e}")
                # Cleanup on failure
                if model_id in self._models:
                    del self._models[model_id]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise ModelLoadError(f"Failed to load {model_id}: {str(e)}")
    
    def warmup_model(self, model_id: str, warmup_prompts: Optional[List[str]] = None) -> bool:
        """
        Warm up a model for consistent performance
        
        Args:
            model_id: Model to warm up
            warmup_prompts: Optional custom warmup prompts
            
        Returns:
            True if warmup successful
        """
        if model_id not in self._models:
            raise ModelNotLoadedError(f"Model {model_id} not loaded")
        
        model = self._models[model_id]
        sampling_config = self._sampling_configs[model_id]
        stats = self._stats[model_id]
        
        logger.info(f"🔥 Warming up model: {model_id}")
        
        start_time = time.time()
        
        try:
            # Default warmup prompts
            if warmup_prompts is None:
                warmup_prompts = [
                    "This is a warmup prompt for model initialization.",
                    "Process this test document with optimal performance."
                ]
            
            sampling_params = sampling_config.to_sampling_params()
            
            # Run warmup inference
            for prompt in warmup_prompts:
                model.generate([prompt], sampling_params)
            
            warmup_time = time.time() - start_time
            stats.warmup_time = warmup_time
            
            logger.info(f"✅ Model {model_id} warmed up in {warmup_time:.1f}s")
            return True
            
        except Exception as e:
            logger.error(f"Model warmup failed for {model_id}: {e}")
            return False
    
    def generate(
        self, 
        model_id: str, 
        prompts: Union[str, List[str], List[Dict[str, Any]]],
        sampling_config: Optional[SamplingConfig] = None
    ) -> List[Any]:
        """
        Generate text using a loaded model
        
        Args:
            model_id: Model to use
            prompts: Text prompts or multimodal inputs
            sampling_config: Optional sampling override
            
        Returns:
            Generated outputs
        """
        if model_id not in self._models:
            raise ModelNotLoadedError(f"Model {model_id} not loaded")
        
        model = self._models[model_id]
        config = sampling_config or self._sampling_configs[model_id]
        stats = self._stats[model_id]
        
        # Convert single prompt to list
        if isinstance(prompts, str):
            prompts = [prompts]
        
        start_time = time.time()
        
        try:
            sampling_params = config.to_sampling_params()
            
            # Handle multimodal input format
            if isinstance(prompts, dict) and "multi_modal_data" in prompts:
                # Pass the multimodal dict directly to vLLM
                outputs = model.generate(prompts, sampling_params)
            else:
                # For text-only prompts
                outputs = model.generate(prompts, sampling_params)
            
            # Update statistics
            inference_time = time.time() - start_time
            stats.total_inference_time += inference_time
            stats.inference_count += 1
            stats.last_used = datetime.now()
            
            logger.debug(f"Generated {len(outputs)} outputs in {inference_time:.2f}s")
            
            return outputs
            
        except Exception as e:
            logger.error(f"Generation failed for {model_id}: {e}")
            raise
    
    def unload_model(self, model_id: str) -> bool:
        """
        Unload a model from GPU memory
        
        Args:
            model_id: Model to unload
            
        Returns:
            True if unloaded successfully
        """
        with self._lock:
            if model_id not in self._models:
                logger.warning(f"Model {model_id} not loaded")
                return False
            
            logger.info(f"🧹 Unloading model: {model_id}")
            
            try:
                # Delete model
                del self._models[model_id]
                
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logger.info(f"✅ Model {model_id} unloaded")
                return True
                
            except Exception as e:
                logger.error(f"Failed to unload model {model_id}: {e}")
                return False
    
    def unload_all_models(self):
        """Unload all models and clear GPU memory"""
        logger.info("🧹 Unloading all models...")
        
        with self._lock:
            model_ids = list(self._models.keys())
            
            for model_id in model_ids:
                try:
                    del self._models[model_id]
                    logger.info(f"   Unloaded: {model_id}")
                except Exception as e:
                    logger.error(f"   Failed to unload {model_id}: {e}")
            
            self._models.clear()
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("✅ All models unloaded")
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models"""
        return list(self._models.keys())
    
    def is_model_loaded(self, model_id: str) -> bool:
        """Check if a model is currently loaded"""
        return model_id in self._models
    
    def get_model_stats(self, model_id: str) -> Optional[ModelStats]:
        """Get statistics for a model"""
        return self._stats.get(model_id)
    
    def print_statistics(self):
        """Print comprehensive statistics for all models"""
        logger.info("📊 vLLM Model Manager Statistics:")
        
        if not self._stats:
            logger.info("   No models registered")
            return
        
        for model_id, stats in self._stats.items():
            loaded = "✅" if self.is_model_loaded(model_id) else "❌"
            logger.info(f"   {loaded} {model_id}:")
            logger.info(f"      Load time: {stats.load_time:.1f}s")
            logger.info(f"      Warmup time: {stats.warmup_time:.1f}s")
            logger.info(f"      Inference count: {stats.inference_count}")
            if stats.inference_count > 0:
                logger.info(f"      Avg inference time: {stats.average_inference_time:.2f}s")
                logger.info(f"      Last used: {stats.last_used}")
        
        # GPU info
        gpu_info = self.get_gpu_memory_info()
        if gpu_info["available"]:
            logger.info("   GPU Status:")
            for device in gpu_info["devices"]:
                util = device["utilization"]
                logger.info(f"      GPU {device['device_id']}: {util:.1%} utilized")
    
    @contextmanager
    def batch_context(self, model_ids: List[str]):
        """
        Context manager for batch processing with automatic cleanup
        
        Args:
            model_ids: List of models to load for batch processing
        """
        loaded_models = []
        
        try:
            # Load all required models
            for model_id in model_ids:
                if self.load_model(model_id):
                    loaded_models.append(model_id)
                    # Skip warmup for faster startup
                    # self.warmup_model(model_id)
            
            logger.info(f"🚀 Batch context ready with {len(loaded_models)} models")
            yield self
            
        finally:
            # Cleanup: unload all models
            for model_id in loaded_models:
                self.unload_model(model_id)
            
            logger.info("🏁 Batch context cleanup completed")


# Global singleton instance
model_manager = VLLMModelManager()