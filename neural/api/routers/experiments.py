"""
Experiment tracking endpoints.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from fastapi.responses import FileResponse

from neural.api.auth import User, get_current_active_user
from neural.api.config import settings
from neural.api.models import ExperimentListResponse, ExperimentMetrics, ExperimentResponse
from neural.tracking.experiment_tracker import ExperimentManager

router = APIRouter(prefix="/experiments", tags=["Experiments"])


@router.get("/", response_model=ExperimentListResponse)
async def list_experiments(
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of items to return"),
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    current_user: User = Depends(get_current_active_user)
) -> ExperimentListResponse:
    """
    List all experiments.
    
    Retrieve a paginated list of all experiments with optional filtering by status.
    
    - **skip**: Number of experiments to skip (for pagination)
    - **limit**: Maximum number of experiments to return
    - **status_filter**: Filter experiments by status (created, running, completed, failed)
    """
    try:
        manager = ExperimentManager(base_dir=settings.experiments_path)
        all_experiments = manager.list_experiments()
        
        if status_filter:
            all_experiments = [
                exp for exp in all_experiments
                if exp.get("status") == status_filter
            ]
        
        total = len(all_experiments)
        experiments = all_experiments[skip:skip + limit]
        
        experiment_responses = []
        for exp in experiments:
            summary = exp.get("summary", {})
            
            experiment_responses.append(ExperimentResponse(
                experiment_id=exp["experiment_id"],
                experiment_name=exp["experiment_name"],
                status=exp["status"],
                hyperparameters=summary.get("hyperparameters", {}) if summary else {},
                metrics=ExperimentMetrics(
                    latest=summary.get("metrics", {}).get("latest", {}) if summary else {},
                    best=summary.get("metrics", {}).get("best", {}) if summary else {},
                    history=[]
                ),
                artifacts=summary.get("artifacts", []) if summary else [],
                created_at=exp.get("start_time", ""),
                updated_at=exp.get("end_time")
            ))
        
        return ExperimentListResponse(
            experiments=experiment_responses,
            total=total,
            skip=skip,
            limit=limit
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list experiments: {str(e)}"
        )


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: str = Path(..., description="Experiment ID"),
    current_user: User = Depends(get_current_active_user)
) -> ExperimentResponse:
    """
    Get details of a specific experiment.
    
    Retrieve detailed information about an experiment including:
    - Hyperparameters
    - Metrics history
    - Artifacts
    - Status and timestamps
    """
    try:
        manager = ExperimentManager(base_dir=settings.experiments_path)
        tracker = manager.get_experiment(experiment_id)
        
        if not tracker:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment {experiment_id} not found"
            )
        
        metrics_history = tracker.get_metrics()
        
        latest_metrics = {}
        if metrics_history:
            latest = metrics_history[-1]
            latest_metrics = {k: v for k, v in latest.items() if k not in ["timestamp", "step"]}
        
        best_metrics = {}
        all_metric_names = set()
        for entry in metrics_history:
            all_metric_names.update([k for k in entry.keys() if k not in ["timestamp", "step"]])
        
        for metric_name in all_metric_names:
            best_value, step = tracker.get_best_metric(metric_name, mode="max")
            if best_value is not None:
                best_metrics[metric_name] = {
                    "value": best_value,
                    "step": step
                }
        
        return ExperimentResponse(
            experiment_id=tracker.experiment_id,
            experiment_name=tracker.experiment_name,
            status=tracker.metadata.get("status", "unknown"),
            hyperparameters=tracker.hyperparameters,
            metrics=ExperimentMetrics(
                latest=latest_metrics,
                best=best_metrics,
                history=metrics_history
            ),
            artifacts=list(tracker.artifacts.keys()),
            created_at=tracker.metadata.get("start_time", ""),
            updated_at=tracker.metadata.get("end_time")
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get experiment: {str(e)}"
        )


@router.delete("/{experiment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_experiment(
    experiment_id: str = Path(..., description="Experiment ID"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Delete an experiment.
    
    Permanently delete an experiment and all associated data including:
    - Metrics history
    - Artifacts
    - Plots and visualizations
    """
    try:
        manager = ExperimentManager(base_dir=settings.experiments_path)
        success = manager.delete_experiment(experiment_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment {experiment_id} not found"
            )
        
        return None
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete experiment: {str(e)}"
        )


@router.get("/{experiment_id}/artifacts/{artifact_name}")
async def get_experiment_artifact(
    experiment_id: str = Path(..., description="Experiment ID"),
    artifact_name: str = Path(..., description="Artifact name"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Download an experiment artifact.
    
    Download a specific artifact (model file, plot, dataset, etc.) from an experiment.
    """
    try:
        manager = ExperimentManager(base_dir=settings.experiments_path)
        tracker = manager.get_experiment(experiment_id)
        
        if not tracker:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment {experiment_id} not found"
            )
        
        if artifact_name not in tracker.artifacts:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Artifact {artifact_name} not found"
            )
        
        artifact_info = tracker.artifacts[artifact_name]
        artifact_path = artifact_info["path"]
        
        return FileResponse(
            path=artifact_path,
            filename=artifact_name,
            media_type="application/octet-stream"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get artifact: {str(e)}"
        )


@router.get("/{experiment_id}/compare")
async def compare_experiments(
    experiment_id: str = Path(..., description="Base experiment ID"),
    compare_with: List[str] = Query(..., description="Experiment IDs to compare with"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Compare multiple experiments.
    
    Generate comparison plots and statistics for multiple experiments.
    Returns comparison data including metrics, hyperparameters, and performance.
    """
    try:
        manager = ExperimentManager(base_dir=settings.experiments_path)
        
        all_ids = [experiment_id] + compare_with
        
        comparison_data = {
            "experiments": [],
            "metrics_comparison": {},
            "hyperparameter_comparison": {}
        }
        
        for exp_id in all_ids:
            tracker = manager.get_experiment(exp_id)
            if tracker:
                comparison_data["experiments"].append({
                    "experiment_id": tracker.experiment_id,
                    "experiment_name": tracker.experiment_name,
                    "status": tracker.metadata.get("status"),
                    "hyperparameters": tracker.hyperparameters,
                    "metrics": tracker.metrics_history
                })
        
        return comparison_data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compare experiments: {str(e)}"
        )
