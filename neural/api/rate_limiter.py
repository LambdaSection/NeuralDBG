"""
Rate limiting middleware for Neural API.
"""

import time
from collections import defaultdict
from typing import Callable, Dict

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware

from neural.api.config import settings


class RateLimiter:
    """In-memory rate limiter using sliding window algorithm."""
    
    def __init__(self, requests: int = 100, period: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            requests: Maximum number of requests allowed
            period: Time period in seconds
        """
        self.requests = requests
        self.period = period
        self.clients: Dict[str, list] = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        """
        Check if request from client is allowed.
        
        Args:
            client_id: Client identifier (IP address or API key)
            
        Returns:
            True if request is allowed, False otherwise
        """
        now = time.time()
        client_requests = self.clients[client_id]
        
        cutoff_time = now - self.period
        client_requests[:] = [req_time for req_time in client_requests if req_time > cutoff_time]
        
        if len(client_requests) >= self.requests:
            return False
        
        client_requests.append(now)
        return True
    
    def get_remaining_requests(self, client_id: str) -> int:
        """
        Get remaining requests for client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Number of remaining requests
        """
        now = time.time()
        client_requests = self.clients[client_id]
        cutoff_time = now - self.period
        
        valid_requests = [req_time for req_time in client_requests if req_time > cutoff_time]
        return max(0, self.requests - len(valid_requests))
    
    def get_reset_time(self, client_id: str) -> float:
        """
        Get time until rate limit resets for client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Seconds until reset
        """
        client_requests = self.clients[client_id]
        if not client_requests:
            return 0
        
        oldest_request = min(client_requests)
        reset_time = oldest_request + self.period
        return max(0, reset_time - time.time())


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""
    
    def __init__(self, app, rate_limiter: RateLimiter):
        """
        Initialize middleware.
        
        Args:
            app: FastAPI application
            rate_limiter: RateLimiter instance
        """
        super().__init__(app)
        self.rate_limiter = rate_limiter
    
    async def dispatch(self, request: Request, call_next: Callable):
        """
        Process request with rate limiting.
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
            
        Returns:
            Response
        """
        if not settings.rate_limit_enabled:
            return await call_next(request)
        
        excluded_paths = ["/docs", "/redoc", "/openapi.json", "/health"]
        if request.url.path in excluded_paths:
            return await call_next(request)
        
        client_id = request.client.host if request.client else "unknown"
        
        api_key = request.headers.get(settings.api_key_header)
        if api_key:
            client_id = api_key
        
        if not self.rate_limiter.is_allowed(client_id):
            reset_time = int(self.rate_limiter.get_reset_time(client_id))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Try again in {reset_time} seconds.",
                headers={
                    "X-RateLimit-Limit": str(self.rate_limiter.requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_time),
                }
            )
        
        response = await call_next(request)
        
        remaining = self.rate_limiter.get_remaining_requests(client_id)
        reset_time = int(self.rate_limiter.get_reset_time(client_id))
        
        response.headers["X-RateLimit-Limit"] = str(self.rate_limiter.requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_time)
        
        return response
