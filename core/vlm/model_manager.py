#!/usr/bin/env python3
"""
VLM Model Manager for efficient model loading and switching.
Manages GPU memory by loading only one model at a time.
"""

import logging
import torch
import gc
from typing import Optional, Union, Dict, Any
from pathlib import Path

from core.clients.transformers_qwen25_vl_client import TransformersQwen25VLClient
from core.clients.transformers_pixtral_client import TransformersPixtralClient
from core.clients.transformers_llava_client import TransformersLLaVAClient

logger = logging.getLogger(__name__)


class VLMModelManager:
    """
    Manages VLM model loading and unloading for memory-efficient processing.
    Only one model is kept in memory at a time.
    """
    
    def __init__(self):
        """Initialize the model manager."""
        self.current_model: Optional[Union[TransformersQwen25VLClient, TransformersPixtralClient, TransformersLLaVAClient]] = None
        self.current_model_name: Optional[str] = None
        self._model_configs = self._get_default_configs()
        
    def _get_default_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get default configurations for each model."""
        return {
            "qwen": {
                "temperature": 0.2,
                "max_new_tokens": 512,
                "load_in_8bit": False  # Keep current working config
            },
            "pixtral": {
                "temperature": 0.3,
                "max_new_tokens": 512,
                "load_in_8bit": True
            },
            "llava": {
                "model_name": "llava-hf/llava-v1.6-mistral-7b-hf",
                "load_in_8bit": True,
                "temperature": 0.2,
                "max_new_tokens": 512
            }
        }
    
    def load_qwen(self, **kwargs) -> TransformersQwen25VLClient:
        """
        Load Qwen2.5-VL model for general image processing.
        
        Args:
            **kwargs: Override default configuration
            
        Returns:
            TransformersQwen25VLClient: Loaded Qwen model
        """
        if self.current_model_name == "qwen" and self.current_model is not None:
            logger.info("Qwen2.5-VL already loaded")
            return self.current_model
            
        logger.info("Loading Qwen2.5-VL model...")
        self.cleanup()
        
        # Merge configs
        config = {**self._model_configs["qwen"], **kwargs}
        
        try:
            self.current_model = TransformersQwen25VLClient(**config)
            self.current_model_name = "qwen"
            logger.info("✅ Qwen2.5-VL loaded successfully")
            return self.current_model
        except Exception as e:
            logger.error(f"Failed to load Qwen2.5-VL: {e}")
            raise
    
    def load_pixtral(self, **kwargs) -> TransformersPixtralClient:
        """
        Load Pixtral model for diagram and complex visual processing.
        
        Args:
            **kwargs: Override default configuration
            
        Returns:
            TransformersPixtralClient: Loaded Pixtral model
        """
        if self.current_model_name == "pixtral" and self.current_model is not None:
            logger.info("Pixtral already loaded")
            return self.current_model
            
        logger.info("Loading Pixtral model...")
        self.cleanup()
        
        # Merge configs
        config = {**self._model_configs["pixtral"], **kwargs}
        
        try:
            self.current_model = TransformersPixtralClient(**config)
            self.current_model_name = "pixtral"
            logger.info("✅ Pixtral loaded successfully")
            return self.current_model
        except Exception as e:
            logger.error(f"Failed to load Pixtral: {e}")
            raise
    
    def load_llava(self, **kwargs) -> TransformersLLaVAClient:
        """
        Load LLaVA model for detailed descriptions.
        
        Args:
            **kwargs: Override default configuration
            
        Returns:
            TransformersLLaVAClient: Loaded LLaVA model
        """
        if self.current_model_name == "llava" and self.current_model is not None:
            logger.info("LLaVA already loaded")
            return self.current_model
            
        logger.info("Loading LLaVA model...")
        self.cleanup()
        
        # Merge configs
        config = {**self._model_configs["llava"], **kwargs}
        
        try:
            self.current_model = TransformersLLaVAClient(**config)
            self.current_model_name = "llava"
            logger.info("✅ LLaVA loaded successfully")
            return self.current_model
        except Exception as e:
            logger.error(f"Failed to load LLaVA: {e}")
            raise
    
    def get_current_model(self) -> Optional[Union[TransformersQwen25VLClient, TransformersPixtralClient, TransformersLLaVAClient]]:
        """
        Get the currently loaded model.
        
        Returns:
            Current model instance or None if no model is loaded
        """
        return self.current_model
    
    def get_current_model_name(self) -> Optional[str]:
        """
        Get the name of the currently loaded model.
        
        Returns:
            Model name ('qwen', 'pixtral', 'llava') or None
        """
        return self.current_model_name
    
    def cleanup(self):
        """
        Clean up the current model and free GPU memory.
        """
        if self.current_model is not None:
            logger.info(f"Cleaning up {self.current_model_name} model...")
            
            # Call model's cleanup if available
            if hasattr(self.current_model, 'cleanup'):
                try:
                    self.current_model.cleanup()
                except Exception as e:
                    logger.warning(f"Error during model cleanup: {e}")
            
            # Delete model reference
            del self.current_model
            self.current_model = None
            self.current_model_name = None
            
            # Force garbage collection
            gc.collect()
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
            logger.info("✅ Model cleanup completed")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current GPU memory usage.
        
        Returns:
            Dict with memory statistics in GB
        """
        if not torch.cuda.is_available():
            return {"allocated": 0, "reserved": 0, "free": 0}
            
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - allocated
        
        return {
            "allocated": round(allocated, 2),
            "reserved": round(reserved, 2),
            "free": round(free, 2),
            "total": round(total, 2)
        }
    
    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()