"""
Model Cache Manager for VLMs

Manages local caching of Visual Language Models to avoid repeated downloads.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelCacheManager:
    """Manages local caching of VLM models"""
    
    # Default cache directory
    DEFAULT_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"
    
    # Model specifications
    MODEL_SPECS = {
        "Qwen/Qwen2.5-VL-7B-Instruct": {
            "size_gb": 15.2,
            "files": ["model-*.safetensors", "config.json", "preprocessor_config.json"],
            "family": "qwen2_vl"
        },
        "llava-hf/llava-v1.6-mistral-7b-hf": {
            "size_gb": 14.5,
            "files": ["model-*.safetensors", "config.json", "preprocessor_config.json"],
            "family": "llava"
        },
        "mistral-community/pixtral-12b": {
            "size_gb": 24.2,
            "files": ["model-*.safetensors", "config.json", "processor_config.json"],
            "family": "pixtral"
        }
    }
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize cache manager
        
        Args:
            cache_dir: Custom cache directory (defaults to HuggingFace cache)
        """
        self.cache_dir = cache_dir or self.DEFAULT_CACHE_DIR
        self.cache_info_file = self.cache_dir / "vlm_cache_info.json"
        self._ensure_cache_dir()
        
    def _ensure_cache_dir(self):
        """Ensure cache directory exists"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_model_cache_path(self, model_name: str) -> Path:
        """Get the cache path for a specific model"""
        # HuggingFace uses a specific naming convention
        safe_model_name = model_name.replace("/", "--")
        return self.cache_dir / f"models--{safe_model_name}"
        
    def is_model_cached(self, model_name: str) -> bool:
        """Check if a model is already cached locally"""
        cache_path = self.get_model_cache_path(model_name)
        
        if not cache_path.exists():
            return False
            
        # Check if key files exist
        snapshots_dir = cache_path / "snapshots"
        if not snapshots_dir.exists():
            return False
            
        # Check if at least one snapshot exists with model files
        for snapshot in snapshots_dir.iterdir():
            if snapshot.is_dir():
                # Check for model weights
                has_weights = any(
                    f.name.startswith("model") and f.suffix in [".safetensors", ".bin"]
                    for f in snapshot.iterdir() if f.is_file()
                )
                if has_weights:
                    return True
                    
        return False
        
    def get_cache_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about cached model"""
        cache_path = self.get_model_cache_path(model_name)
        
        if not self.is_model_cached(model_name):
            return {
                "cached": False,
                "model_name": model_name,
                "cache_path": str(cache_path)
            }
            
        # Calculate cache size
        total_size = 0
        file_count = 0
        
        for root, dirs, files in os.walk(cache_path):
            for file in files:
                file_path = Path(root) / file
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1
                    
        # Get last modified time
        snapshots_dir = cache_path / "snapshots"
        last_modified = None
        
        for snapshot in snapshots_dir.iterdir():
            if snapshot.is_dir():
                mtime = snapshot.stat().st_mtime
                if last_modified is None or mtime > last_modified:
                    last_modified = mtime
                    
        return {
            "cached": True,
            "model_name": model_name,
            "cache_path": str(cache_path),
            "size_gb": total_size / (1024**3),
            "file_count": file_count,
            "last_modified": datetime.fromtimestamp(last_modified).isoformat() if last_modified else None,
            "expected_size_gb": self.MODEL_SPECS.get(model_name, {}).get("size_gb", "unknown")
        }
        
    def get_all_cached_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all cached VLM models"""
        cached_models = {}
        
        for model_name in self.MODEL_SPECS:
            info = self.get_cache_info(model_name)
            if info["cached"]:
                cached_models[model_name] = info
                
        return cached_models
        
    def estimate_download_time(self, model_name: str, bandwidth_mbps: float = 100) -> float:
        """
        Estimate download time for a model
        
        Args:
            model_name: Name of the model
            bandwidth_mbps: Download bandwidth in Mbps
            
        Returns:
            Estimated time in seconds
        """
        model_spec = self.MODEL_SPECS.get(model_name, {})
        size_gb = model_spec.get("size_gb", 20)  # Default 20GB
        
        # Convert to seconds
        size_mb = size_gb * 1024
        time_seconds = (size_mb * 8) / bandwidth_mbps
        
        return time_seconds
        
    def preload_model(self, model_name: str, force: bool = False) -> bool:
        """
        Preload a model to cache
        
        Args:
            model_name: Name of the model to preload
            force: Force re-download even if cached
            
        Returns:
            True if successful
        """
        if self.is_model_cached(model_name) and not force:
            logger.info(f"Model {model_name} already cached")
            return True
            
        logger.info(f"Preloading model {model_name}...")
        
        try:
            # Import here to avoid circular dependencies
            from transformers import AutoModel, AutoProcessor
            
            # Just loading the config and processor triggers cache
            logger.info("Downloading model configuration...")
            _ = AutoProcessor.from_pretrained(model_name)
            
            # For Pixtral, we need to explicitly download the model
            if "pixtral" in model_name.lower():
                logger.info("Downloading model weights (this may take a while)...")
                from transformers import LlavaForConditionalGeneration
                _ = LlavaForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype="auto",
                    device_map="cpu"  # Just download, don't load to GPU
                )
                
            logger.info(f"Model {model_name} successfully cached")
            return True
            
        except Exception as e:
            logger.error(f"Failed to preload model {model_name}: {e}")
            return False
            
    def clear_model_cache(self, model_name: str) -> bool:
        """Clear cache for a specific model"""
        cache_path = self.get_model_cache_path(model_name)
        
        if cache_path.exists():
            try:
                import shutil
                shutil.rmtree(cache_path)
                logger.info(f"Cleared cache for {model_name}")
                return True
            except Exception as e:
                logger.error(f"Failed to clear cache for {model_name}: {e}")
                return False
        else:
            logger.info(f"No cache found for {model_name}")
            return True
            
    def get_cache_summary(self) -> str:
        """Get a summary of all cached models"""
        cached_models = self.get_all_cached_models()
        
        if not cached_models:
            return "No VLM models currently cached."
            
        summary_lines = ["📦 Cached VLM Models:"]
        total_size = 0
        
        for model_name, info in cached_models.items():
            size_gb = info.get("size_gb", 0)
            total_size += size_gb
            summary_lines.append(
                f"  • {model_name}: {size_gb:.1f}GB "
                f"(expected: {info.get('expected_size_gb', 'unknown')}GB)"
            )
            
        summary_lines.append(f"\n💾 Total cache size: {total_size:.1f}GB")
        summary_lines.append(f"📁 Cache directory: {self.cache_dir}")
        
        return "\n".join(summary_lines)


def main():
    """Test cache manager functionality"""
    manager = ModelCacheManager()
    
    print("🔍 Checking VLM Model Cache Status...")
    print("=" * 60)
    
    for model_name in manager.MODEL_SPECS:
        info = manager.get_cache_info(model_name)
        
        if info["cached"]:
            print(f"✅ {model_name}")
            print(f"   Size: {info['size_gb']:.1f}GB")
            print(f"   Files: {info['file_count']}")
            print(f"   Last modified: {info['last_modified']}")
        else:
            print(f"❌ {model_name} (not cached)")
            est_time = manager.estimate_download_time(model_name)
            print(f"   Estimated download time: {est_time/60:.1f} minutes @ 100Mbps")
            
        print()
        
    print("\n" + manager.get_cache_summary())


if __name__ == "__main__":
    main()