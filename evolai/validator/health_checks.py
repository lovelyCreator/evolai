
import time
import threading
import logging
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    name: str
    status: HealthStatus
    last_check: datetime
    last_heartbeat: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "last_check": self.last_check.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


class HealthChecker:
    
    def __init__(self, heartbeat_timeout: int = 300):
        self.heartbeat_timeout = heartbeat_timeout
        self.components: Dict[str, ComponentHealth] = {}
        self._lock = threading.Lock()
        self._check_functions: Dict[str, Callable[[], bool]] = {}
    
    def register_component(self, name: str, check_function: Optional[Callable[[], bool]] = None):
        with self._lock:
            self.components[name] = ComponentHealth(
                name=name,
                status=HealthStatus.UNKNOWN,
                last_check=datetime.utcnow()
            )
            if check_function:
                self._check_functions[name] = check_function
        
        logger.info(f"Registered health check for component: {name}")
    
    def heartbeat(self, component_name: str, metadata: Optional[Dict] = None):
        with self._lock:
            if component_name not in self.components:
                self.register_component(component_name)
            
            component = self.components[component_name]
            component.last_heartbeat = datetime.utcnow()
            component.last_check = datetime.utcnow()
            component.status = HealthStatus.HEALTHY
            component.error_message = None
            
            if metadata:
                component.metadata.update(metadata)
    
    def mark_unhealthy(self, component_name: str, error: str):
        with self._lock:
            if component_name not in self.components:
                self.register_component(component_name)
            
            component = self.components[component_name]
            component.status = HealthStatus.UNHEALTHY
            component.error_message = error
            component.last_check = datetime.utcnow()
        
        logger.error(f"Component {component_name} marked unhealthy: {error}")
    
    def mark_degraded(self, component_name: str, reason: str):
        with self._lock:
            if component_name not in self.components:
                self.register_component(component_name)
            
            component = self.components[component_name]
            component.status = HealthStatus.DEGRADED
            component.error_message = reason
            component.last_check = datetime.utcnow()
        
        logger.warning(f"Component {component_name} degraded: {reason}")
    
    def check_all(self) -> Dict[str, ComponentHealth]:
        with self._lock:
            now = datetime.utcnow()
            

            for name, component in self.components.items():
                if component.last_heartbeat:
                    time_since_heartbeat = (now - component.last_heartbeat).total_seconds()
                    
                    if time_since_heartbeat > self.heartbeat_timeout:
                        component.status = HealthStatus.UNHEALTHY
                        component.error_message = f"No heartbeat for {time_since_heartbeat:.0f}s"
                    elif time_since_heartbeat > self.heartbeat_timeout * 0.5:
                        component.status = HealthStatus.DEGRADED
                        component.error_message = f"Slow heartbeat: {time_since_heartbeat:.0f}s"
                

                if name in self._check_functions:
                    try:
                        is_healthy = self._check_functions[name]()
                        if not is_healthy and component.status == HealthStatus.HEALTHY:
                            component.status = HealthStatus.DEGRADED
                            component.error_message = "Active health check failed"
                    except Exception as e:
                        component.status = HealthStatus.UNHEALTHY
                        component.error_message = f"Health check error: {str(e)}"
                
                component.last_check = now
            
            return dict(self.components)
    
    def get_overall_status(self) -> HealthStatus:
        with self._lock:
            if not self.components:
                return HealthStatus.UNKNOWN
            
            statuses = [c.status for c in self.components.values()]
            
            if HealthStatus.UNHEALTHY in statuses:
                return HealthStatus.UNHEALTHY
            elif HealthStatus.DEGRADED in statuses:
                return HealthStatus.DEGRADED
            elif HealthStatus.UNKNOWN in statuses:
                return HealthStatus.UNKNOWN
            else:
                return HealthStatus.HEALTHY
    
    def get_health_report(self) -> Dict:
        components = self.check_all()
        
        return {
            "overall_status": self.get_overall_status().value,
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                name: health.to_dict()
                for name, health in components.items()
            }
        }


class WatchdogTimer:
    
    def __init__(self, timeout: float, callback: Callable, name: str = "watchdog"):
        self.timeout = timeout
        self.callback = callback
        self.name = name
        self._timer: Optional[threading.Timer] = None
        self._cancelled = False
    
    def start(self):
        self._cancelled = False
        self._timer = threading.Timer(self.timeout, self._on_timeout)
        self._timer.daemon = True
        self._timer.start()
        logger.debug(f"Watchdog {self.name} started ({self.timeout}s)")
    
    def cancel(self):
        self._cancelled = True
        if self._timer:
            self._timer.cancel()
            logger.debug(f"Watchdog {self.name} cancelled")
    
    def _on_timeout(self):
        if not self._cancelled:
            logger.error(f"Watchdog {self.name} timeout after {self.timeout}s")
            try:
                self.callback()
            except Exception as e:
                logger.error(f"Watchdog callback error: {e}")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cancel()
        return False


class PeriodicHealthMonitor:
    
    def __init__(
        self,
        health_checker: HealthChecker,
        check_interval: int = 60,
        log_interval: int = 300
    ):
        self.health_checker = health_checker
        self.check_interval = check_interval
        self.log_interval = log_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_log = datetime.utcnow()
    
    def start(self):
        if self._running:
            logger.warning("Health monitor already running")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Health monitor started")
    
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Health monitor stopped")
    
    def _monitor_loop(self):
        while self._running:
            try:

                self.health_checker.check_all()
                

                now = datetime.utcnow()
                if (now - self._last_log).total_seconds() >= self.log_interval:
                    report = self.health_checker.get_health_report()
                    logger.info(f"Health report: {report['overall_status']}")
                    

                    for name, component in report['components'].items():
                        if component['status'] != 'healthy':
                            logger.warning(
                                f"Component {name}: {component['status']} - "
                                f"{component.get('error_message', 'No details')}"
                            )
                    
                    self._last_log = now
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
            
            time.sleep(self.check_interval)
