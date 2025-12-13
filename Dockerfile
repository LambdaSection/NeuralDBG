# Multi-stage build for Neural DSL API
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create app user
RUN groupadd -r neural && useradd -r -g neural neural

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt requirements-minimal.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Install Neural DSL with API dependencies
COPY setup.py README.md ./
COPY neural/ ./neural/
RUN pip install -e ".[api]"

# Install additional API dependencies
RUN pip install \
    fastapi>=0.104.0 \
    uvicorn[standard]>=0.24.0 \
    celery>=5.3.0 \
    redis>=5.0.0 \
    python-jose[cryptography]>=3.3.0 \
    passlib[bcrypt]>=1.7.4 \
    python-multipart>=0.0.6 \
    pydantic-settings>=2.0.0 \
    requests>=2.31.0

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p /app/neural_storage /app/neural_experiments /app/neural_models && \
    chown -R neural:neural /app

# Switch to non-root user
USER neural

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Default command
CMD ["uvicorn", "neural.api.main:app", "--host", "0.0.0.0", "--port", "8000"]


# Worker stage for Celery
FROM base as worker

USER neural

CMD ["celery", "-A", "neural.api.celery_app", "worker", "--loglevel=info", "--concurrency=4"]
