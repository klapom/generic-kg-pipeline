#!/usr/bin/env python3
"""
Check and manage VLM model cache
"""

import sys
sys.path.insert(0, '.')

from core.utils.model_cache_manager import ModelCacheManager
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Check and manage VLM model cache")
    parser.add_argument("--preload", type=str, help="Preload a specific model")
    parser.add_argument("--preload-all", action="store_true", help="Preload all VLM models")
    parser.add_argument("--clear", type=str, help="Clear cache for a specific model")
    args = parser.parse_args()
    
    manager = ModelCacheManager()
    
    if args.preload:
        logger.info(f"Preloading model: {args.preload}")
        success = manager.preload_model(args.preload)
        if success:
            logger.info("✅ Model preloaded successfully")
        else:
            logger.error("❌ Failed to preload model")
            
    elif args.preload_all:
        logger.info("Preloading all VLM models...")
        for model_name in manager.MODEL_SPECS:
            logger.info(f"\n{'='*60}")
            logger.info(f"Preloading {model_name}...")
            manager.preload_model(model_name)
            
    elif args.clear:
        logger.info(f"Clearing cache for: {args.clear}")
        success = manager.clear_model_cache(args.clear)
        if success:
            logger.info("✅ Cache cleared")
        else:
            logger.error("❌ Failed to clear cache")
            
    else:
        # Default: show cache status
        print("\n🔍 VLM Model Cache Status")
        print("=" * 60)
        
        all_cached = True
        for model_name in manager.MODEL_SPECS:
            info = manager.get_cache_info(model_name)
            
            if info["cached"]:
                print(f"✅ {model_name}")
                print(f"   📦 Size: {info['size_gb']:.1f}GB")
                print(f"   📁 Files: {info['file_count']}")
                print(f"   🕒 Last modified: {info['last_modified']}")
            else:
                all_cached = False
                print(f"❌ {model_name} (not cached)")
                est_time = manager.estimate_download_time(model_name, bandwidth_mbps=100)
                print(f"   ⏱️  Estimated download: {est_time/60:.1f} min @ 100Mbps")
                
            print()
            
        print(manager.get_cache_summary())
        
        if not all_cached:
            print("\n💡 Tip: Use --preload-all to download all models")
            print("   Example: python tests/test_check_model_cache.py --preload-all")


if __name__ == "__main__":
    main()