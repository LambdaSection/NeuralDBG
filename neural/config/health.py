"""Health check system for Neural DSL services."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import os
import socket
import time
from typing import Any, Dict, List, Optional


class HealthStatus(Enum):
    """Health status of a service."""
    
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ServiceHealth:
    """Health information for a single service."""
    
    name: str
    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    response_time_ms: Optional[float] = None
    checked_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'status': self.status.value,
            'message': self.message,
            'details': self.details,
            'response_time_ms': self.response_time_ms,
            'checked_at': self.checked_at,
        }


class HealthChecker:
    """Health checker for Neural DSL services."""
    
    def __init__(self):
        """Initialize health checker."""
        self.service_configs = {
            'api': {
                'host': os.getenv('API_HOST', '0.0.0.0'),
                'port': int(os.getenv('API_PORT', 8000)),
                'endpoint': '/health',
            },
            'dashboard': {
                'host': os.getenv('DASHBOARD_HOST', '0.0.0.0'),
                'port': int(os.getenv('DASHBOARD_PORT', 8050)),
            },
            'aquarium': {
                'host': os.getenv('AQUARIUM_HOST', '0.0.0.0'),
                'port': int(os.getenv('AQUARIUM_PORT', 8051)),
            },
            'marketplace': {
                'host': os.getenv('MARKETPLACE_HOST', '0.0.0.0'),
                'port': int(os.getenv('MARKETPLACE_PORT', 5000)),
                'endpoint': '/api/stats',
            },
            'redis': {
                'host': os.getenv('REDIS_HOST', 'localhost'),
                'port': int(os.getenv('REDIS_PORT', 6379)),
            },
            'celery': {
                'redis_host': os.getenv('REDIS_HOST', 'localhost'),
                'redis_port': int(os.getenv('REDIS_PORT', 6379)),
            },
        }
    
    def check_all(self) -> Dict[str, ServiceHealth]:
        """Check health of all services.
        
        Returns
        -------
        dict
            Dictionary mapping service names to their health status
        """
        results = {}
        
        for service in self.service_configs.keys():
            results[service] = self.check_service(service)
        
        return results
    
    def check_service(self, service: str) -> ServiceHealth:
        """Check health of a specific service.
        
        Parameters
        ----------
        service : str
            Service name ('api', 'dashboard', 'aquarium', 'marketplace', 'redis', 'celery')
        
        Returns
        -------
        ServiceHealth
            Health status of the service
        """
        if service not in self.service_configs:
            return ServiceHealth(
                name=service,
                status=HealthStatus.UNKNOWN,
                message=f"Unknown service: {service}"
            )
        
        check_method = getattr(self, f'_check_{service}', None)
        if check_method:
            return check_method()
        
        # Default port check for services without specific health endpoints
        return self._check_port(service)
    
    def _check_api(self) -> ServiceHealth:
        """Check API service health."""
        config = self.service_configs['api']
        
        try:
            import requests
            
            start_time = time.time()
            url = f"http://{config['host']}:{config['port']}{config['endpoint']}"
            
            # Use localhost if host is 0.0.0.0
            if config['host'] == '0.0.0.0':
                url = f"http://localhost:{config['port']}{config['endpoint']}"
            
            response = requests.get(url, timeout=5)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                return ServiceHealth(
                    name='api',
                    status=HealthStatus.HEALTHY,
                    message='API is healthy',
                    details=data,
                    response_time_ms=response_time
                )
            else:
                return ServiceHealth(
                    name='api',
                    status=HealthStatus.DEGRADED,
                    message=f'API returned status {response.status_code}',
                    response_time_ms=response_time
                )
        
        except ImportError:
            return ServiceHealth(
                name='api',
                status=HealthStatus.UNKNOWN,
                message='requests library not available for health check'
            )
        except Exception as e:
            return ServiceHealth(
                name='api',
                status=HealthStatus.UNHEALTHY,
                message=f'API health check failed: {str(e)}'
            )
    
    def _check_dashboard(self) -> ServiceHealth:
        """Check dashboard service health."""
        return self._check_port('dashboard')
    
    def _check_aquarium(self) -> ServiceHealth:
        """Check Aquarium service health."""
        return self._check_port('aquarium')
    
    def _check_marketplace(self) -> ServiceHealth:
        """Check marketplace service health."""
        config = self.service_configs['marketplace']
        
        try:
            import requests
            
            start_time = time.time()
            url = f"http://{config['host']}:{config['port']}{config['endpoint']}"
            
            # Use localhost if host is 0.0.0.0
            if config['host'] == '0.0.0.0':
                url = f"http://localhost:{config['port']}{config['endpoint']}"
            
            response = requests.get(url, timeout=5)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                return ServiceHealth(
                    name='marketplace',
                    status=HealthStatus.HEALTHY,
                    message='Marketplace is healthy',
                    details=data.get('stats', {}),
                    response_time_ms=response_time
                )
            else:
                return ServiceHealth(
                    name='marketplace',
                    status=HealthStatus.DEGRADED,
                    message=f'Marketplace returned status {response.status_code}',
                    response_time_ms=response_time
                )
        
        except ImportError:
            return ServiceHealth(
                name='marketplace',
                status=HealthStatus.UNKNOWN,
                message='requests library not available for health check'
            )
        except Exception as e:
            return ServiceHealth(
                name='marketplace',
                status=HealthStatus.UNHEALTHY,
                message=f'Marketplace health check failed: {str(e)}'
            )
    
    def _check_redis(self) -> ServiceHealth:
        """Check Redis service health."""
        config = self.service_configs['redis']
        
        try:
            import redis
            
            start_time = time.time()
            client = redis.Redis(
                host=config['host'],
                port=config['port'],
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Try to ping Redis
            pong = client.ping()
            response_time = (time.time() - start_time) * 1000
            
            if pong:
                info = client.info()
                return ServiceHealth(
                    name='redis',
                    status=HealthStatus.HEALTHY,
                    message='Redis is healthy',
                    details={
                        'version': info.get('redis_version', 'unknown'),
                        'uptime_days': info.get('uptime_in_days', 0),
                        'connected_clients': info.get('connected_clients', 0),
                        'used_memory_human': info.get('used_memory_human', 'unknown'),
                    },
                    response_time_ms=response_time
                )
            else:
                return ServiceHealth(
                    name='redis',
                    status=HealthStatus.UNHEALTHY,
                    message='Redis did not respond to ping'
                )
        
        except ImportError:
            return ServiceHealth(
                name='redis',
                status=HealthStatus.UNKNOWN,
                message='redis library not available for health check'
            )
        except Exception as e:
            return ServiceHealth(
                name='redis',
                status=HealthStatus.UNHEALTHY,
                message=f'Redis health check failed: {str(e)}'
            )
    
    def _check_celery(self) -> ServiceHealth:
        """Check Celery service health."""
        try:
            from celery import Celery
            
            broker_url = os.getenv('CELERY_BROKER_URL')
            if not broker_url:
                redis_host = os.getenv('REDIS_HOST', 'localhost')
                redis_port = os.getenv('REDIS_PORT', '6379')
                broker_url = f'redis://{redis_host}:{redis_port}/0'
            
            start_time = time.time()
            app = Celery(broker=broker_url)
            
            # Try to get active workers
            inspector = app.control.inspect()
            active = inspector.active()
            response_time = (time.time() - start_time) * 1000
            
            if active is not None:
                num_workers = len(active)
                total_tasks = sum(len(tasks) for tasks in active.values())
                
                return ServiceHealth(
                    name='celery',
                    status=HealthStatus.HEALTHY if num_workers > 0 else HealthStatus.DEGRADED,
                    message=f'{num_workers} worker(s) active' if num_workers > 0 else 'No workers available',
                    details={
                        'workers': num_workers,
                        'active_tasks': total_tasks,
                    },
                    response_time_ms=response_time
                )
            else:
                return ServiceHealth(
                    name='celery',
                    status=HealthStatus.UNHEALTHY,
                    message='Unable to connect to Celery broker'
                )
        
        except ImportError:
            return ServiceHealth(
                name='celery',
                status=HealthStatus.UNKNOWN,
                message='celery library not available for health check'
            )
        except Exception as e:
            return ServiceHealth(
                name='celery',
                status=HealthStatus.UNHEALTHY,
                message=f'Celery health check failed: {str(e)}'
            )
    
    def _check_port(self, service: str) -> ServiceHealth:
        """Check if a service port is open.
        
        Parameters
        ----------
        service : str
            Service name
        
        Returns
        -------
        ServiceHealth
            Health status based on port availability
        """
        config = self.service_configs[service]
        host = config['host']
        port = config['port']
        
        # Use localhost if host is 0.0.0.0
        if host == '0.0.0.0':
            host = 'localhost'
        
        try:
            start_time = time.time()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            response_time = (time.time() - start_time) * 1000
            sock.close()
            
            if result == 0:
                return ServiceHealth(
                    name=service,
                    status=HealthStatus.HEALTHY,
                    message=f'Service is listening on port {port}',
                    details={'host': host, 'port': port},
                    response_time_ms=response_time
                )
            else:
                return ServiceHealth(
                    name=service,
                    status=HealthStatus.UNHEALTHY,
                    message=f'Service is not listening on port {port}',
                    details={'host': host, 'port': port}
                )
        
        except Exception as e:
            return ServiceHealth(
                name=service,
                status=HealthStatus.UNHEALTHY,
                message=f'Port check failed: {str(e)}',
                details={'host': host, 'port': port}
            )
    
    def get_readiness_status(self, services: Optional[List[str]] = None) -> bool:
        """Check if services are ready (for Kubernetes readiness probe).
        
        Parameters
        ----------
        services : list of str, optional
            List of services to check. If None, checks all services.
        
        Returns
        -------
        bool
            True if all specified services are healthy
        """
        if services is None:
            services = list(self.service_configs.keys())
        
        for service in services:
            health = self.check_service(service)
            if health.status == HealthStatus.UNHEALTHY:
                return False
        
        return True
    
    def get_liveness_status(self) -> bool:
        """Check if the application is alive (for Kubernetes liveness probe).
        
        Returns
        -------
        bool
            Always True unless the process is completely dead
        """
        return True
