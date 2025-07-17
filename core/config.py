"""
Configuration module - Bridge to new unified config system
This file exists to maintain backward compatibility with imports like:
from core.config import Config, get_config
"""

import warnings
from core.config_new.unified_manager import get_config as new_get_config

# Show deprecation warning
warnings.warn(
    "Importing from core.config is deprecated. Use core.config_new.unified_manager instead.",
    DeprecationWarning,
    stacklevel=2
)

# Redirect to new config system
def get_config(config_path=None):
    """Legacy get_config function - redirects to new system"""
    if config_path is not None:
        warnings.warn(
            "config_path parameter is ignored. Configuration is loaded from config.yaml",
            DeprecationWarning,
            stacklevel=2
        )
    return new_get_config()

# For now, we'll raise an error for other imports
def __getattr__(name):
    raise ImportError(
        f"'{name}' is no longer available in core.config. "
        f"Please use core.config_new.unified_manager instead."
    )

__all__ = ['get_config']