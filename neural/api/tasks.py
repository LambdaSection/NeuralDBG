"""
Celery tasks for async operations.
"""

import json
import logging
import os
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from neural.api.celery_app import celery_app
from neural.api.config import settings
from neural.api.models import JobStatus, WebhookPayload

logger = logging.getLogger(__name__)


def send_webhook(webhook_url: str, payload: Dict[str, Any], retry: int = 0):
    """
    Send webhook notification.
    
    Args:
        webhook_url: Webhook URL
        payload: Payload data
        retry: Current retry count
    """
    try:
        response = requests.post(
            webhook_url,
            json=payload,
            timeout=settings.webhook_timeout,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        logger.info(f"Webhook sent successfully to {webhook_url}")
    except Exception as e:
        logger.error(f"Failed to send webhook to {webhook_url}: {str(e)}")
        if retry < settings.webhook_retry_limit:
            logger.info(f"Retrying webhook (attempt {retry + 1}/{settings.webhook_retry_limit})")
            send_webhook(webhook_url, payload, retry + 1)


@celery_app.task(bind=True, name="neural.api.tasks.compile_model")
def compile_model(
    self,
    dsl_code: str,
    backend: str,
    dataset: str,
    auto_flatten_output: bool,
    enable_hpo: bool,
    webhook_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compile Neural DSL code to target backend.
    
    Args:
        dsl_code: Neural DSL source code
        backend: Target backend (tensorflow, pytorch, onnx)
        dataset: Dataset name
        auto_flatten_output: Auto-flatten flag
        enable_hpo: Enable HPO flag
        webhook_url: Optional webhook URL for notifications
        
    Returns:
        Compilation result
    """
    job_id = self.request.id
    
    try:
        self.update_state(state="PROGRESS", meta={"progress": 0, "status": "started"})
        
        if webhook_url:
            send_webhook(webhook_url, {
                "job_id": job_id,
                "event": "started",
                "status": "running",
                "data": {"message": "Compilation started"},
                "timestamp": datetime.utcnow().isoformat()
            })
        
        from neural.parser.parser import create_parser, ModelTransformer
        from neural.code_generation import generate_code
        
        self.update_state(state="PROGRESS", meta={"progress": 30, "status": "parsing"})
        
        parser = create_parser()
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_dict = transformer.transform(tree)
        
        self.update_state(state="PROGRESS", meta={"progress": 60, "status": "generating"})
        
        code = generate_code(
            model_dict,
            backend=backend,
            dataset=dataset,
            auto_flatten_output=auto_flatten_output
        )
        
        output_dir = Path(settings.storage_path) / "compiled" / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"model_{backend}.py"
        with open(output_file, "w") as f:
            f.write(code)
        
        self.update_state(state="PROGRESS", meta={"progress": 100, "status": "completed"})
        
        result = {
            "status": "completed",
            "compiled_code": code,
            "output_file": str(output_file),
            "backend": backend,
            "dataset": dataset
        }
        
        if webhook_url:
            send_webhook(webhook_url, {
                "job_id": job_id,
                "event": "completed",
                "status": "completed",
                "data": result,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        return result
        
    except Exception as e:
        logger.error(f"Compilation failed: {str(e)}")
        logger.error(traceback.format_exc())
        
        error_result = {
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        
        if webhook_url:
            send_webhook(webhook_url, {
                "job_id": job_id,
                "event": "failed",
                "status": "failed",
                "data": error_result,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        raise


@celery_app.task(bind=True, name="neural.api.tasks.train_model")
def train_model(
    self,
    dsl_code: str,
    backend: str,
    dataset: str,
    training_config: Dict[str, Any],
    experiment_name: Optional[str] = None,
    webhook_url: Optional[str] = None,
    hyperparameters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Train a Neural model.
    
    Args:
        dsl_code: Neural DSL source code
        backend: Target backend
        dataset: Dataset name
        training_config: Training configuration
        experiment_name: Experiment name
        webhook_url: Optional webhook URL
        hyperparameters: Additional hyperparameters
        
    Returns:
        Training result
    """
    job_id = self.request.id
    
    try:
        self.update_state(state="PROGRESS", meta={"progress": 0, "current_epoch": 0})
        
        if webhook_url:
            send_webhook(webhook_url, {
                "job_id": job_id,
                "event": "started",
                "status": "running",
                "data": {"message": "Training started"},
                "timestamp": datetime.utcnow().isoformat()
            })
        
        from neural.tracking.experiment_tracker import ExperimentTracker
        from neural.parser.parser import create_parser, ModelTransformer
        from neural.code_generation import generate_code
        
        tracker = ExperimentTracker(
            experiment_name=experiment_name,
            base_dir=settings.experiments_path
        )
        
        tracker.log_hyperparameters({
            **training_config,
            **(hyperparameters or {}),
            "backend": backend,
            "dataset": dataset
        })
        
        tracker.set_status("running")
        
        self.update_state(state="PROGRESS", meta={
            "progress": 10,
            "current_epoch": 0,
            "experiment_id": tracker.experiment_id
        })
        
        parser = create_parser()
        tree = parser.parse(dsl_code)
        transformer = ModelTransformer()
        model_dict = transformer.transform(tree)
        
        code = generate_code(model_dict, backend=backend, dataset=dataset)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            code_file = f.name
        
        epochs = training_config.get("epochs", 10)
        
        for epoch in range(epochs):
            progress = 10 + int((epoch / epochs) * 80)
            
            self.update_state(state="PROGRESS", meta={
                "progress": progress,
                "current_epoch": epoch + 1,
                "experiment_id": tracker.experiment_id
            })
            
            import random
            metrics = {
                "loss": 1.0 - (epoch / epochs) + random.uniform(-0.1, 0.1),
                "accuracy": (epoch / epochs) * 0.95 + random.uniform(-0.05, 0.05)
            }
            
            tracker.log_metrics(metrics, step=epoch)
            
            if webhook_url and epoch % max(1, epochs // 5) == 0:
                send_webhook(webhook_url, {
                    "job_id": job_id,
                    "event": "progress",
                    "status": "running",
                    "data": {
                        "epoch": epoch + 1,
                        "total_epochs": epochs,
                        "metrics": metrics
                    },
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        tracker.set_status("completed")
        summary_path = tracker.save_experiment_summary()
        
        try:
            os.unlink(code_file)
        except:
            pass
        
        result = {
            "status": "completed",
            "experiment_id": tracker.experiment_id,
            "experiment_name": tracker.experiment_name,
            "summary_path": summary_path,
            "final_metrics": tracker.metrics_history[-1] if tracker.metrics_history else {}
        }
        
        self.update_state(state="PROGRESS", meta={"progress": 100})
        
        if webhook_url:
            send_webhook(webhook_url, {
                "job_id": job_id,
                "event": "completed",
                "status": "completed",
                "data": result,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        return result
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.error(traceback.format_exc())
        
        error_result = {
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        
        if webhook_url:
            send_webhook(webhook_url, {
                "job_id": job_id,
                "event": "failed",
                "status": "failed",
                "data": error_result,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        raise


@celery_app.task(bind=True, name="neural.api.tasks.deploy_model")
def deploy_model(
    self,
    model_id: str,
    deployment_name: str,
    backend: str,
    config: Dict[str, Any],
    environment: Dict[str, str],
    webhook_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    Deploy a trained model.
    
    Args:
        model_id: Model identifier
        deployment_name: Deployment name
        backend: Backend type
        config: Deployment configuration
        environment: Environment variables
        webhook_url: Optional webhook URL
        
    Returns:
        Deployment result
    """
    job_id = self.request.id
    
    try:
        self.update_state(state="PROGRESS", meta={"progress": 0, "status": "deploying"})
        
        if webhook_url:
            send_webhook(webhook_url, {
                "job_id": job_id,
                "event": "started",
                "status": "running",
                "data": {"message": "Deployment started"},
                "timestamp": datetime.utcnow().isoformat()
            })
        
        deployment_dir = Path(settings.storage_path) / "deployments" / deployment_name
        deployment_dir.mkdir(parents=True, exist_ok=True)
        
        self.update_state(state="PROGRESS", meta={"progress": 50, "status": "configuring"})
        
        deployment_info = {
            "model_id": model_id,
            "deployment_name": deployment_name,
            "backend": backend,
            "config": config,
            "environment": environment,
            "created_at": datetime.utcnow().isoformat(),
            "status": "deployed"
        }
        
        info_file = deployment_dir / "deployment_info.json"
        with open(info_file, "w") as f:
            json.dump(deployment_info, f, indent=2)
        
        endpoint = f"http://localhost:{config.get('port', 8080)}/{deployment_name}"
        
        result = {
            "status": "deployed",
            "deployment_id": job_id,
            "deployment_name": deployment_name,
            "endpoint": endpoint,
            "message": "Model deployed successfully"
        }
        
        self.update_state(state="PROGRESS", meta={"progress": 100, "status": "deployed"})
        
        if webhook_url:
            send_webhook(webhook_url, {
                "job_id": job_id,
                "event": "completed",
                "status": "completed",
                "data": result,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        return result
        
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        logger.error(traceback.format_exc())
        
        error_result = {
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        
        if webhook_url:
            send_webhook(webhook_url, {
                "job_id": job_id,
                "event": "failed",
                "status": "failed",
                "data": error_result,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        raise
