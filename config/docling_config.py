"""
Docling Integration Configuration
Feature flags and settings for gradual rollout
"""

import os

def get_env_bool(name: str, default: bool) -> bool:
    """Get boolean environment variable with default"""
    value = os.getenv(name, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')

def get_env_int(name: str, default: int) -> int:
    """Get integer environment variable with default"""
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default

# Feature Flags for Docling Integration
DOCLING_CONFIG = {
    # Main feature flag - controls whether to use docling at all
    "use_docling": get_env_bool("USE_DOCLING", True),  # Enabled after migration to unified client
    
    # Direct image extraction during parsing (recommended)
    "extract_images_directly": get_env_bool("EXTRACT_IMAGES_DIRECTLY", True),
    
    # Fallback to legacy parser if docling fails
    "fallback_to_legacy": get_env_bool("FALLBACK_TO_LEGACY", False),  # Disabled - no legacy client
    
    # Performance monitoring
    "log_performance": get_env_bool("LOG_PERFORMANCE", True),
    "performance_threshold_seconds": float(os.getenv("PERFORMANCE_THRESHOLD_SECONDS", "30.0")),
    
    # Rollout controls
    "rollout_percentage": get_env_int("DOCLING_ROLLOUT_PERCENTAGE", 100),  # Full rollout after migration
    
    # Image extraction settings
    "image_extraction": {
        "max_image_size": get_env_int("MAX_IMAGE_SIZE", 2048),
        "image_quality": get_env_int("IMAGE_QUALITY", 95),
        "extract_tables_as_images": get_env_bool("EXTRACT_TABLES", True),
        "extract_formulas_as_images": get_env_bool("EXTRACT_FORMULAS", True)
    },
    
    # Memory management
    "memory_limits": {
        "max_pdf_size_mb": get_env_int("MAX_PDF_SIZE_MB", 50),
        "max_pages_per_batch": get_env_int("MAX_PAGES_BATCH", 10)
    },
    
    # Error handling
    "error_handling": {
        "max_retries": 1,
        "timeout_seconds": 300,  # 5 minutes per document
        "continue_on_page_error": get_env_bool("CONTINUE_ON_ERROR", True)  # Don't fail entire document for one page
    }
}

# Environment-specific overrides
DEVELOPMENT_OVERRIDES = {
    "use_docling": True,  # Enable in development for testing
    "rollout_percentage": 100,
    "log_performance": True,
    "performance_threshold_seconds": 10.0
}

PRODUCTION_OVERRIDES = {
    "use_docling": False,  # Gradual rollout in production
    "rollout_percentage": 0,
    "log_performance": True,
    "performance_threshold_seconds": 30.0
}

def get_config(environment: str = "development") -> dict:
    """
    Get configuration for specific environment
    
    Args:
        environment: "development", "production", or "testing"
    
    Returns:
        Combined configuration dict
    """
    config = DOCLING_CONFIG.copy()
    
    if environment == "development":
        config.update(DEVELOPMENT_OVERRIDES)
    elif environment == "production":
        config.update(PRODUCTION_OVERRIDES)
    # testing uses base config
    
    return config

# Environment-specific overrides
DEVELOPMENT_OVERRIDES = {
    # Conservative settings for development
    "use_docling": False,
    "rollout_percentage": 0
}

PRODUCTION_OVERRIDES = {
    # Full rollout in production
    "use_docling": True,
    "rollout_percentage": 100
}

def is_docling_enabled(environment: str = "development") -> bool:
    """Check if docling is enabled for current environment"""
    config = get_config(environment)
    return config["use_docling"]

def should_use_docling_for_document(document_hash: str, environment: str = "development") -> bool:
    """
    Determine if docling should be used for a specific document
    Based on rollout percentage and document hash for consistent routing
    
    Args:
        document_hash: Hash of document content for consistent routing
        environment: Current environment
    
    Returns:
        True if docling should be used for this document
    """
    config = get_config(environment)
    
    if not config["use_docling"]:
        return False
    
    rollout_pct = config["rollout_percentage"]
    if rollout_pct >= 100:
        return True
    if rollout_pct <= 0:
        return False
    
    # Use document hash for consistent routing (same document always gets same treatment)
    import hashlib
    hash_int = int(hashlib.md5(document_hash.encode()).hexdigest()[:8], 16)
    return (hash_int % 100) < rollout_pct