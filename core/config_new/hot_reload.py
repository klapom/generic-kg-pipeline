"""
Hot-Reload Feature für Konfigurationsänderungen
Überwacht config.yaml und lädt Änderungen automatisch
"""
import asyncio
import logging
from datetime import datetime
from typing import Optional, Callable, Set
from pathlib import Path

from .unified_manager import get_config_manager

logger = logging.getLogger(__name__)


class ConfigWatcher:
    """
    Überwacht Konfigurationsdatei auf Änderungen
    und triggert automatisches Neuladen
    """
    
    def __init__(self, check_interval: float = 5.0):
        """
        Initialize config watcher
        
        Args:
            check_interval: Sekunden zwischen Checks (default: 5)
        """
        self.check_interval = check_interval
        self.manager = get_config_manager()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._callbacks: Set[Callable] = set()
        
    def add_callback(self, callback: Callable) -> None:
        """Füge Callback hinzu, der bei Config-Änderung aufgerufen wird"""
        self._callbacks.add(callback)
        
    def remove_callback(self, callback: Callable) -> None:
        """Entferne Callback"""
        self._callbacks.discard(callback)
    
    async def _run_callbacks(self) -> None:
        """Führe alle registrierten Callbacks aus"""
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    async def _watch_loop(self) -> None:
        """Hauptschleife für Config-Überwachung"""
        logger.info("🔄 Config hot-reload watcher started")
        logger.info(f"📁 Watching: {self.manager.config_path}")
        logger.info(f"⏱️  Check interval: {self.check_interval}s")
        
        while self._running:
            try:
                if self.manager.is_modified():
                    logger.info("📝 Configuration file changed - reloading...")
                    
                    # Reload config
                    old_profile = self.manager.config.profile
                    self.manager.reload()
                    new_profile = self.manager.config.profile
                    
                    logger.info("✅ Configuration reloaded successfully")
                    
                    # Log wichtige Änderungen
                    if old_profile != new_profile:
                        logger.info(f"🔄 Profile changed: {old_profile} → {new_profile}")
                    
                    # Validiere neue Config
                    validation = self.manager.validate()
                    if validation['errors']:
                        logger.error(f"❌ Config validation errors: {validation['errors']}")
                    if validation['warnings']:
                        logger.warning(f"⚠️  Config validation warnings: {validation['warnings']}")
                    
                    # Callbacks ausführen
                    await self._run_callbacks()
                    
            except Exception as e:
                logger.error(f"Error in config watcher: {e}")
            
            await asyncio.sleep(self.check_interval)
    
    async def start(self) -> None:
        """Starte Config-Überwachung"""
        if self._running:
            logger.warning("Config watcher already running")
            return
            
        self._running = True
        self._task = asyncio.create_task(self._watch_loop())
        logger.info("✅ Config hot-reload enabled")
    
    async def stop(self) -> None:
        """Stoppe Config-Überwachung"""
        if not self._running:
            return
            
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        logger.info("🛑 Config hot-reload stopped")


# Globale Watcher-Instanz
_watcher: Optional[ConfigWatcher] = None


def get_config_watcher() -> ConfigWatcher:
    """Get singleton config watcher instance"""
    global _watcher
    if _watcher is None:
        _watcher = ConfigWatcher()
    return _watcher


async def enable_hot_reload(check_interval: float = 5.0) -> ConfigWatcher:
    """
    Aktiviere Hot-Reload für Config
    
    Args:
        check_interval: Sekunden zwischen Checks
        
    Returns:
        ConfigWatcher instance
    """
    watcher = get_config_watcher()
    watcher.check_interval = check_interval
    await watcher.start()
    return watcher


async def disable_hot_reload() -> None:
    """Deaktiviere Hot-Reload"""
    watcher = get_config_watcher()
    await watcher.stop()