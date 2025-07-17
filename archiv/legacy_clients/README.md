# Legacy SmolDocling Clients Archive

This directory contains archived legacy SmolDocling client implementations that have been replaced by the unified `VLLMSmolDoclingFinalClient`.

## Archived Files

- **vllm_smoldocling.py** - Original deprecated service implementation
- **vllm_smoldocling_local.py** - Legacy local vLLM implementation
- **vllm_smoldocling_docling.py** - Experimental docling integration
- **vllm_smoldocling_docling_example.py** - Example implementation
- **vllm_smoldocling_docling_improved.py** - Improved experimental version

## Migration Status

These files were archived on 2025-07-17 as part of the SmolDocling architecture cleanup.

All functionality has been consolidated into:
- `core/clients/vllm_smoldocling_final.py` - Production-ready client with docling integration

## Import Migration

All imports have been updated:
- `from core.clients.vllm_smoldocling_local import VLLMSmolDoclingClient` → `from core.clients import VLLMSmolDoclingClient`
- `VLLMSmolDoclingClient` is now an alias for `VLLMSmolDoclingFinalClient`

## Important Note

These files are kept for reference only. DO NOT import or use these implementations in new code.
