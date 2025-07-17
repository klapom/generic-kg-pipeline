# Legacy Configuration Archive

This directory contains archived legacy configuration files that have been replaced by the unified configuration system.

## Archived Files

- **config_old.py** - Original configuration implementation
- **config_compat.py** - Compatibility bridge layer

## Migration Status

These files were archived on 2025-07-17 as part of the configuration consolidation.

All configuration is now handled by:
- `core/config_new/unified_manager.py` - Main configuration system
- `config/docling_config.py` - Docling-specific configuration with environment variables

## Environment Variables

The following environment variables are now available for UI configuration:

### Docling Feature Flags
- `USE_DOCLING` (default: true) - Enable/disable docling integration
- `EXTRACT_IMAGES_DIRECTLY` (default: true) - Extract images during parsing
- `FALLBACK_TO_LEGACY` (default: false) - Fallback to legacy parser
- `DOCLING_ROLLOUT_PERCENTAGE` (default: 100) - Percentage rollout

### Image Extraction
- `MAX_IMAGE_SIZE` (default: 2048) - Maximum image dimension
- `IMAGE_QUALITY` (default: 95) - JPEG compression quality
- `EXTRACT_TABLES` (default: true) - Extract tables as images
- `EXTRACT_FORMULAS` (default: true) - Extract formulas as images

### Performance Settings
- `LOG_PERFORMANCE` (default: true) - Enable performance logging
- `PERFORMANCE_THRESHOLD_SECONDS` (default: 30.0) - Slow processing threshold
- `MAX_PDF_SIZE_MB` (default: 50) - Maximum PDF size to process
- `MAX_PAGES_BATCH` (default: 10) - Pages per batch

### Error Handling
- `CONTINUE_ON_ERROR` (default: true) - Continue processing on page errors

## Migration Path

All imports should be updated:
- `from core.config import get_config` → `from core.config_new.unified_manager import get_config`
- Direct use of environment variables for dynamic configuration
