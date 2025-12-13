"""
Job management endpoints.
"""

from datetime import datetime
from typing import Optional

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, HTTPException, Path, status

from neural.api.auth import User, get_current_active_user
from neural.api.celery_app import celery_app
from neural.api.models import JobStatus, JobStatusResponse, TrainingJobRequest, TrainingJobResponse
from neural.api.tasks import train_model

router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.post("/train", response_model=TrainingJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_training_job(
    request: TrainingJobRequest,
    current_user: User = Depends(get_current_active_user)
) -> TrainingJobResponse:
    """
    Create a new training job.
    
    Submit a training job that will execute asynchronously. You can provide either
    DSL code or a pre-compiled model ID.
    
    - **dsl_code**: Neural DSL source code (required if model_id not provided)
    - **model_id**: Pre-compiled model ID (required if dsl_code not provided)
    - **backend**: Target backend framework
    - **dataset**: Dataset to use for training
    - **training_config**: Training configuration (epochs, batch_size, etc.)
    - **experiment_name**: Name for experiment tracking
    - **webhook_url**: URL to receive job status notifications
    - **hyperparameters**: Additional hyperparameters
    
    Returns a job ID that can be used to monitor training progress.
    """
    if not request.dsl_code and not request.model_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either dsl_code or model_id must be provided"
        )
    
    try:
        task = train_model.apply_async(
            kwargs={
                "dsl_code": request.dsl_code,
                "backend": request.backend.value,
                "dataset": request.dataset,
                "training_config": request.training_config.model_dump(),
                "experiment_name": request.experiment_name,
                "webhook_url": str(request.webhook_url) if request.webhook_url else None,
                "hyperparameters": request.hyperparameters
            }
        )
        
        return TrainingJobResponse(
            job_id=task.id,
            status=JobStatus.pending,
            message="Training job submitted successfully"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit training job: {str(e)}"
        )


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str = Path(..., description="Job ID"),
    current_user: User = Depends(get_current_active_user)
) -> JobStatusResponse:
    """
    Get the status of a job.
    
    Query the current status and progress of a compilation, training, or deployment job.
    
    Returns detailed information including:
    - Current status (pending, running, completed, failed)
    - Progress percentage (for training jobs)
    - Result data (when completed)
    - Error information (if failed)
    """
    try:
        task = AsyncResult(job_id, app=celery_app)
        
        status_map = {
            "PENDING": JobStatus.pending,
            "STARTED": JobStatus.running,
            "PROGRESS": JobStatus.running,
            "SUCCESS": JobStatus.completed,
            "FAILURE": JobStatus.failed,
            "RETRY": JobStatus.running,
            "REVOKED": JobStatus.cancelled,
        }
        
        job_status = status_map.get(task.state, JobStatus.pending)
        
        result_data = None
        error_msg = None
        progress = None
        started_at = None
        completed_at = None
        
        if task.state == "PROGRESS":
            info = task.info or {}
            progress = info.get("progress")
            result_data = info
        elif task.state == "SUCCESS":
            result_data = task.result
            progress = 100.0
            completed_at = datetime.utcnow()
        elif task.state == "FAILURE":
            error_msg = str(task.info) if task.info else "Unknown error"
            
        if task.state in ["STARTED", "PROGRESS", "SUCCESS", "FAILURE"]:
            started_at = datetime.utcnow()
        
        return JobStatusResponse(
            job_id=job_id,
            status=job_status,
            progress=progress,
            result=result_data,
            error=error_msg,
            created_at=datetime.utcnow(),
            started_at=started_at,
            completed_at=completed_at
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get job status: {str(e)}"
        )


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_job(
    job_id: str = Path(..., description="Job ID"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Cancel a running job.
    
    Attempt to cancel a pending or running job. Jobs that have already completed
    or failed cannot be cancelled.
    """
    try:
        task = AsyncResult(job_id, app=celery_app)
        
        if task.state in ["SUCCESS", "FAILURE"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot cancel job in {task.state} state"
            )
        
        task.revoke(terminate=True)
        return None
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel job: {str(e)}"
        )
