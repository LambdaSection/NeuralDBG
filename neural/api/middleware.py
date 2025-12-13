"""
Custom middleware for Neural API.
"""

import logging
import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all requests."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Log request details and response time.
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
            
        Returns:
            Response
        """
        start_time = time.time()
        
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "client": request.client.host if request.client else None,
            }
        )
        
        try:
            response = await call_next(request)
            
            process_time = time.time() - start_time
            
            logger.info(
                f"Request completed: {request.method} {request.url.path} "
                f"- Status: {response.status_code} - Time: {process_time:.3f}s",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "process_time": process_time,
                }
            )
            
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            
            logger.error(
                f"Request failed: {request.method} {request.url.path} "
                f"- Error: {str(e)} - Time: {process_time:.3f}s",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "error": str(e),
                    "process_time": process_time,
                },
                exc_info=True
            )
            
            raise


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect request metrics."""
    
    def __init__(self, app):
        """Initialize middleware."""
        super().__init__(app)
        self.request_count = 0
        self.request_times = []
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Collect request metrics.
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
            
        Returns:
            Response
        """
        self.request_count += 1
        start_time = time.time()
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        self.request_times.append(process_time)
        
        if len(self.request_times) > 1000:
            self.request_times = self.request_times[-1000:]
        
        response.headers["X-Request-Count"] = str(self.request_count)
        
        return response
    
    def get_metrics(self):
        """Get collected metrics."""
        if not self.request_times:
            return {
                "total_requests": self.request_count,
                "avg_response_time": 0,
                "min_response_time": 0,
                "max_response_time": 0,
            }
        
        return {
            "total_requests": self.request_count,
            "avg_response_time": sum(self.request_times) / len(self.request_times),
            "min_response_time": min(self.request_times),
            "max_response_time": max(self.request_times),
        }
