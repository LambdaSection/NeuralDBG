"""
Celery application for async task processing.
"""

from celery import Celery

from neural.api.config import settings


celery_app = Celery(
    "neural_api",
    broker=settings.broker_url,
    backend=settings.result_backend,
    include=["neural.api.tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,
    task_soft_time_limit=3300,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)
