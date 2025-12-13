"""
Main FastAPI application for Neural DSL API.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from neural.api.config import settings
from neural.api.models import ErrorResponse, HealthResponse
from neural.api.rate_limiter import RateLimiter, RateLimitMiddleware
from neural.api.routers import compile, deployments, experiments, jobs, models

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Neural API server...")
    
    Path(settings.storage_path).mkdir(parents=True, exist_ok=True)
    Path(settings.experiments_path).mkdir(parents=True, exist_ok=True)
    Path(settings.models_path).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Storage path: {settings.storage_path}")
    logger.info(f"Experiments path: {settings.experiments_path}")
    logger.info(f"Models path: {settings.models_path}")
    
    yield
    
    logger.info("Shutting down Neural API server...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="""
        Neural DSL REST API provides endpoints for:
        
        * **Compilation**: Compile Neural DSL code to TensorFlow, PyTorch, or ONNX
        * **Training**: Submit asynchronous training jobs with experiment tracking
        * **Experiments**: Track and manage ML experiments with metrics and artifacts
        * **Deployments**: Deploy trained models with configurable resources
        * **Jobs**: Monitor and manage async jobs (compilation, training, deployment)
        * **Models**: Manage compiled models
        
        ## Authentication
        
        API uses API key authentication. Include your API key in the request header:
        ```
        X-API-Key: your_api_key_here
        ```
        
        ## Rate Limiting
        
        API requests are rate limited to protect server resources.
        Rate limit information is included in response headers:
        - `X-RateLimit-Limit`: Maximum requests allowed
        - `X-RateLimit-Remaining`: Remaining requests
        - `X-RateLimit-Reset`: Seconds until rate limit resets
        
        ## Webhooks
        
        Async jobs (training, deployment) support webhook notifications.
        Provide a webhook URL to receive real-time updates on job progress.
        
        ## Getting Started
        
        1. Obtain an API key
        2. Compile Neural DSL code using `/compile/` endpoint
        3. Submit training job using `/jobs/train` endpoint
        4. Monitor training progress via `/jobs/{job_id}` endpoint
        5. Track experiments via `/experiments/` endpoints
        6. Deploy trained models using `/deployments/` endpoint
        """,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    rate_limiter = RateLimiter(
        requests=settings.rate_limit_requests,
        period=settings.rate_limit_period
    )
    app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter)
    
    app.include_router(compile.router)
    app.include_router(jobs.router)
    app.include_router(experiments.router)
    app.include_router(deployments.router)
    app.include_router(models.router)
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler."""
        logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="Internal server error",
                detail=str(exc) if settings.debug else "An unexpected error occurred"
            ).model_dump()
        )
    
    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint."""
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
            "openapi": "/openapi.json"
        }
    
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check():
        """Health check endpoint."""
        services = {
            "api": "healthy",
        }
        
        try:
            from neural.api.celery_app import celery_app
            celery_app.control.inspect().active()
            services["celery"] = "healthy"
        except:
            services["celery"] = "unavailable"
        
        return HealthResponse(
            status="healthy",
            version=settings.app_version,
            services=services
        )
    
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "neural.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.workers
    )
