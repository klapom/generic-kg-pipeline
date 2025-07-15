"""
Configuration Management API Endpoints
Für die zukünftige GUI-Integration
"""
from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
import asyncio
import json
import logging

from core.config_new.unified_manager import get_config_manager, get_config
from core.config_new.hot_reload import get_config_watcher

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/api/v1/config", tags=["configuration"])


class ConfigUpdate(BaseModel):
    """Model für Config-Updates"""
    path: str
    value: Any


class ConfigResponse(BaseModel):
    """Response model für Config-Anfragen"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


@router.get("/", response_model=ConfigResponse)
async def get_configuration():
    """
    Hole aktuelle Konfiguration
    
    Returns:
        Aktuelle Konfigurationswerte
    """
    try:
        manager = get_config_manager()
        return ConfigResponse(
            success=True,
            data=manager._raw_config
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schema", response_model=ConfigResponse)
async def get_configuration_schema():
    """
    Hole Konfigurations-Schema für GUI
    
    Returns:
        JSON Schema der Konfiguration
    """
    try:
        manager = get_config_manager()
        schema = manager.export_schema()
        return ConfigResponse(
            success=True,
            data=schema
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/gui-export", response_model=ConfigResponse)
async def get_gui_export():
    """
    Exportiere Konfiguration mit allen Metadaten für GUI
    
    Returns:
        Konfiguration mit Schema und Environment-Variablen Info
    """
    try:
        manager = get_config_manager()
        export = manager.export_for_gui()
        return ConfigResponse(
            success=True,
            data=export
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/value/{path:path}")
async def get_config_value(path: str):
    """
    Hole spezifischen Konfigurations-Wert
    
    Args:
        path: Punkt-separierter Pfad (z.B. "services.vllm.url")
    
    Returns:
        Konfigurations-Wert
    """
    try:
        manager = get_config_manager()
        value = manager.get(path)
        if value is None:
            raise HTTPException(status_code=404, detail=f"Configuration key not found: {path}")
        return {"path": path, "value": value}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/value", response_model=ConfigResponse)
async def update_config_value(update: ConfigUpdate):
    """
    Update einzelnen Konfigurations-Wert (Runtime only)
    
    Args:
        update: Path und neuer Wert
    
    Returns:
        Success status
    """
    try:
        manager = get_config_manager()
        manager.set(update.path, update.value)
        return ConfigResponse(
            success=True,
            message=f"Updated {update.path}"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/save", response_model=ConfigResponse)
async def save_configuration(backup: bool = True):
    """
    Speichere aktuelle Konfiguration
    
    Args:
        backup: Erstelle Backup vor dem Speichern
    
    Returns:
        Success status
    """
    try:
        manager = get_config_manager()
        manager.save(backup=backup)
        return ConfigResponse(
            success=True,
            message="Configuration saved successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reload", response_model=ConfigResponse)
async def reload_configuration():
    """
    Lade Konfiguration neu von Datei
    
    Returns:
        Success status
    """
    try:
        manager = get_config_manager()
        manager.reload()
        return ConfigResponse(
            success=True,
            message="Configuration reloaded successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validate", response_model=ConfigResponse)
async def validate_configuration():
    """
    Validiere aktuelle Konfiguration
    
    Returns:
        Validierungs-Ergebnisse mit Errors, Warnings und Info
    """
    try:
        manager = get_config_manager()
        issues = manager.validate()
        
        has_errors = len(issues['errors']) > 0
        
        return ConfigResponse(
            success=not has_errors,
            data=issues,
            message="Configuration is valid" if not has_errors else "Configuration has errors"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/environment-variables")
async def get_environment_variables():
    """
    Hole Information über verwendete Environment-Variablen
    
    Returns:
        Liste der Environment-Variablen mit Status
    """
    try:
        manager = get_config_manager()
        env_info = manager._get_env_vars_info()
        
        # Gruppiere nach Status
        set_vars = [k for k, v in env_info.items() if v['is_set']]
        unset_vars = [k for k, v in env_info.items() if not v['is_set']]
        
        return {
            "total": len(env_info),
            "set": len(set_vars),
            "unset": len(unset_vars),
            "variables": env_info,
            "summary": {
                "set_variables": set_vars,
                "unset_variables": unset_vars
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/profiles")
async def get_available_profiles():
    """
    Hole verfügbare Konfigurations-Profile
    
    Returns:
        Liste der Profile (dev, test, prod)
    """
    return {
        "profiles": ["dev", "test", "prod"],
        "current": get_config().profile,
        "description": {
            "dev": "Development profile with debug enabled",
            "test": "Testing profile with reduced resources",
            "prod": "Production profile with optimized settings"
        }
    }


# Liste der aktiven WebSocket-Verbindungen
active_connections: List[WebSocket] = []

@router.websocket("/ws")
async def config_websocket(websocket: WebSocket):
    """
    WebSocket für Live-Config-Updates in der GUI
    Sendet automatisch Updates wenn sich die Config ändert
    """
    await websocket.accept()
    active_connections.append(websocket)
    
    # Registriere Callback für Config-Änderungen
    async def on_config_change():
        """Callback der bei Config-Änderung aufgerufen wird"""
        manager = get_config_manager()
        update_data = {
            "type": "config_update",
            "timestamp": asyncio.get_event_loop().time(),
            "data": manager._raw_config,
            "profile": manager.config.profile,
            "validation": manager.validate()
        }
        
        # Sende an alle aktiven Connections
        for connection in active_connections:
            try:
                await connection.send_json(update_data)
            except:
                # Connection ist tot
                pass
    
    # Registriere Callback
    watcher = get_config_watcher()
    watcher.add_callback(on_config_change)
    
    try:
        # Sende initiale Config
        manager = get_config_manager()
        await websocket.send_json({
            "type": "initial_config",
            "data": manager._raw_config,
            "profile": manager.config.profile,
            "schema": manager.export_schema()
        })
        
        # Halte Connection offen
        while True:
            # Warte auf Client-Messages (z.B. ping/pong)
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
                
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        watcher.remove_callback(on_config_change)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)
        watcher.remove_callback(on_config_change)