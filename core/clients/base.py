"""
Base Model Client - Standardisierte Client-Architektur
Bietet einheitliche Funktionalität für alle Model Clients
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generic, Optional, TypeVar, Union, List
from contextlib import asynccontextmanager

import httpx
from pydantic import BaseModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

from core.config_new.unified_manager import get_config


# Typ-Variablen für Request/Response
RequestType = TypeVar('RequestType', bound=BaseModel)
ResponseType = TypeVar('ResponseType', bound=BaseModel)
ConfigType = TypeVar('ConfigType', bound=BaseModel)

logger = logging.getLogger(__name__)


class ClientStatus(Enum):
    """Status eines Model Clients"""
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    TIMEOUT = "timeout"
    ERROR = "error"


class HealthCheckResult(BaseModel):
    """Standardisiertes Health Check Ergebnis"""
    status: ClientStatus
    endpoint: str
    response_time_ms: Optional[float] = None
    last_check: datetime
    details: Dict[str, Any] = {}
    error_message: Optional[str] = None


class ClientMetrics(BaseModel):
    """Client Performance Metriken"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time_ms: float = 0.0
    average_response_time_ms: float = 0.0
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None


class BaseModelClient(ABC, Generic[RequestType, ResponseType, ConfigType]):
    """
    Basis-Klasse für alle Model Clients
    
    Bietet:
    - Einheitliche Konfiguration
    - Automatische Retry-Logik
    - Health Checks
    - Metriken-Sammlung
    - Fehlerbehandlung
    - Async Context Manager
    """
    
    def __init__(self, 
                 service_name: str,
                 config: Optional[ConfigType] = None,
                 custom_endpoint: Optional[str] = None):
        """
        Initialize base model client
        
        Args:
            service_name: Name des Services (z.B. "vllm", "hochschul_llm")
            config: Client-spezifische Konfiguration
            custom_endpoint: Überschreibt Endpoint aus System-Config
        """
        self.service_name = service_name
        self.config = config or self._get_default_config()
        self.metrics = ClientMetrics()
        
        # Hole Endpoint aus System-Config oder nutze custom
        system_config = get_config()
        service_config = getattr(system_config.services, service_name, None)
        
        if custom_endpoint:
            self.endpoint = custom_endpoint
        elif service_config:
            self.endpoint = service_config.url
            self.timeout = service_config.timeout
            self.retry_attempts = service_config.retry_attempts
        else:
            raise ValueError(f"No configuration found for service: {service_name}")
        
        # HTTP Client mit Timeout
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
        )
        
        logger.info(f"Initialized {self.__class__.__name__} for {service_name}: {self.endpoint}")
    
    @abstractmethod
    def _get_default_config(self) -> ConfigType:
        """Hole Default-Konfiguration für diesen Client"""
        pass
    
    @abstractmethod
    async def _process_internal(self, request: RequestType) -> ResponseType:
        """
        Interne Verarbeitung - muss von Subklassen implementiert werden
        
        Args:
            request: Typ-spezifische Anfrage
            
        Returns:
            Typ-spezifische Antwort
        """
        pass
    
    @abstractmethod
    async def _health_check_internal(self) -> Dict[str, Any]:
        """
        Service-spezifischer Health Check
        
        Returns:
            Service-spezifische Health-Informationen
        """
        pass
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def close(self):
        """Schließe Client-Verbindungen"""
        if hasattr(self, 'client') and self.client:
            await self.client.aclose()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def process(self, request: RequestType) -> ResponseType:
        """
        Verarbeite Anfrage mit automatischer Retry-Logik
        
        Args:
            request: Anfrage-Objekt
            
        Returns:
            Response-Objekt
            
        Raises:
            Exception: Bei Verarbeitungsfehlern nach allen Retries
        """
        start_time = time.time()
        
        try:
            # Metriken: Request gestartet
            self.metrics.total_requests += 1
            
            # Delegiere an interne Implementierung
            response = await self._process_internal(request)
            
            # Metriken: Erfolg
            self.metrics.successful_requests += 1
            response_time_ms = (time.time() - start_time) * 1000
            self.metrics.total_response_time_ms += response_time_ms
            self.metrics.average_response_time_ms = (
                self.metrics.total_response_time_ms / self.metrics.successful_requests
            )
            
            logger.debug(f"{self.service_name} processed request in {response_time_ms:.2f}ms")
            return response
            
        except Exception as e:
            # Metriken: Fehler
            self.metrics.failed_requests += 1
            self.metrics.last_error = str(e)
            self.metrics.last_error_time = datetime.now()
            
            logger.error(f"{self.service_name} processing failed: {e}")
            raise
    
    async def health_check(self) -> HealthCheckResult:
        """
        Führe Health Check durch
        
        Returns:
            Standardisiertes Health Check Ergebnis
        """
        start_time = time.time()
        
        try:
            # Service-spezifischer Health Check
            details = await self._health_check_internal()
            
            response_time_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                status=ClientStatus.HEALTHY,
                endpoint=self.endpoint,
                response_time_ms=response_time_ms,
                last_check=datetime.now(),
                details=details
            )
            
        except httpx.TimeoutException:
            return HealthCheckResult(
                status=ClientStatus.TIMEOUT,
                endpoint=self.endpoint,
                last_check=datetime.now(),
                error_message="Health check timed out"
            )
            
        except Exception as e:
            return HealthCheckResult(
                status=ClientStatus.ERROR,
                endpoint=self.endpoint,
                last_check=datetime.now(),
                error_message=str(e)
            )
    
    def get_metrics(self) -> ClientMetrics:
        """Hole aktuelle Client-Metriken"""
        return self.metrics.copy()
    
    def reset_metrics(self):
        """Setze Metriken zurück"""
        self.metrics = ClientMetrics()
    
    async def wait_until_ready(self, 
                              max_attempts: int = 30,
                              delay_seconds: float = 2.0) -> bool:
        """
        Warte bis Service bereit ist
        
        Args:
            max_attempts: Maximale Anzahl Versuche
            delay_seconds: Wartezeit zwischen Versuchen
            
        Returns:
            True wenn Service bereit, False sonst
        """
        for attempt in range(max_attempts):
            health = await self.health_check()
            if health.status == ClientStatus.HEALTHY:
                logger.info(f"{self.service_name} is ready after {attempt + 1} attempts")
                return True
            
            if attempt < max_attempts - 1:
                logger.info(f"Waiting for {self.service_name} to be ready... ({attempt + 1}/{max_attempts})")
                await asyncio.sleep(delay_seconds)
        
        logger.error(f"{self.service_name} did not become ready after {max_attempts} attempts")
        return False


class BatchProcessingMixin:
    """
    Mixin für Batch-Verarbeitung
    Kann von Clients verwendet werden, die Batch-Processing unterstützen
    """
    
    async def process_batch(self, 
                           requests: List[RequestType],
                           batch_size: int = 10,
                           concurrent_batches: int = 3) -> List[ResponseType]:
        """
        Verarbeite mehrere Anfragen in Batches
        
        Args:
            requests: Liste von Anfragen
            batch_size: Größe eines Batches
            concurrent_batches: Anzahl paralleler Batches
            
        Returns:
            Liste von Antworten
        """
        results = []
        semaphore = asyncio.Semaphore(concurrent_batches)
        
        async def process_single_batch(batch: List[RequestType]) -> List[ResponseType]:
            async with semaphore:
                batch_results = []
                for request in batch:
                    try:
                        result = await self.process(request)
                        batch_results.append(result)
                    except Exception as e:
                        logger.error(f"Batch item failed: {e}")
                        # Je nach Use-Case: None, Default-Wert oder Exception
                        batch_results.append(None)
                return batch_results
        
        # Teile in Batches auf
        batches = [requests[i:i + batch_size] for i in range(0, len(requests), batch_size)]
        
        # Verarbeite Batches parallel
        batch_results = await asyncio.gather(
            *[process_single_batch(batch) for batch in batches]
        )
        
        # Flatten results
        for batch_result in batch_results:
            results.extend(batch_result)
        
        return results