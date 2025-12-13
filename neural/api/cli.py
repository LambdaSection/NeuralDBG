"""
CLI commands for managing the Neural API server.
"""

import click
import uvicorn

from neural.api.config import settings


@click.group()
def api_cli():
    """Neural API management commands."""
    pass


@api_cli.command()
@click.option('--host', default=settings.host, help='Host to bind to')
@click.option('--port', default=settings.port, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
@click.option('--workers', default=settings.workers, help='Number of workers')
def run(host: str, port: int, reload: bool, workers: int):
    """Start the API server."""
    click.echo(f"Starting Neural API server on {host}:{port}")
    
    uvicorn.run(
        "neural.api.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=1 if reload else workers,
        log_level="info"
    )


@api_cli.command()
@click.option('--concurrency', default=4, help='Number of worker processes')
@click.option('--loglevel', default='info', help='Log level')
def worker(concurrency: int, loglevel: str):
    """Start a Celery worker."""
    import subprocess
    
    click.echo(f"Starting Celery worker with concurrency={concurrency}")
    
    cmd = [
        'celery',
        '-A', 'neural.api.celery_app',
        'worker',
        f'--loglevel={loglevel}',
        f'--concurrency={concurrency}'
    ]
    
    subprocess.run(cmd)


@api_cli.command()
@click.option('--port', default=5555, help='Port for Flower')
def flower(port: int):
    """Start Celery Flower monitoring."""
    import subprocess
    
    click.echo(f"Starting Flower on port {port}")
    
    cmd = [
        'celery',
        '-A', 'neural.api.celery_app',
        'flower',
        f'--port={port}'
    ]
    
    subprocess.run(cmd)


@api_cli.command()
def check():
    """Check API configuration and dependencies."""
    click.echo("Checking Neural API configuration...")
    
    click.echo(f"✓ API Host: {settings.host}")
    click.echo(f"✓ API Port: {settings.port}")
    click.echo(f"✓ Redis URL: {settings.redis_url}")
    click.echo(f"✓ Storage Path: {settings.storage_path}")
    click.echo(f"✓ Experiments Path: {settings.experiments_path}")
    click.echo(f"✓ Models Path: {settings.models_path}")
    
    try:
        import redis
        r = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password
        )
        r.ping()
        click.echo("✓ Redis connection: OK")
    except Exception as e:
        click.echo(f"✗ Redis connection: FAILED ({str(e)})")
    
    dependencies = [
        'fastapi',
        'uvicorn',
        'celery',
        'redis',
        'pydantic',
        'jose',
        'passlib'
    ]
    
    for dep in dependencies:
        try:
            __import__(dep)
            click.echo(f"✓ {dep}: installed")
        except ImportError:
            click.echo(f"✗ {dep}: NOT INSTALLED")


if __name__ == '__main__':
    api_cli()
