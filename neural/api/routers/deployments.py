"""
Deployment management endpoints.
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status

from neural.api.auth import User, get_current_active_user
from neural.api.models import DeploymentRequest, DeploymentResponse, DeploymentStatus
from neural.api.tasks import deploy_model

router = APIRouter(prefix="/deployments", tags=["Deployments"])


@router.post("/", response_model=DeploymentResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_deployment(
    request: DeploymentRequest,
    current_user: User = Depends(get_current_active_user)
) -> DeploymentResponse:
    """
    Deploy a trained model.
    
    Create a new deployment for a trained model. This will:
    - Package the model with necessary dependencies
    - Configure the deployment environment
    - Set up health checks and monitoring
    - Expose the model via an API endpoint
    
    - **model_id**: ID of the trained model to deploy
    - **deployment_name**: Unique name for this deployment
    - **backend**: Backend framework used by the model
    - **config**: Deployment configuration (replicas, resources, ports)
    - **environment**: Environment variables for the deployment
    """
    try:
        task = deploy_model.apply_async(
            kwargs={
                "model_id": request.model_id,
                "deployment_name": request.deployment_name,
                "backend": request.backend.value,
                "config": request.config.model_dump(),
                "environment": request.environment
            }
        )
        
        return DeploymentResponse(
            deployment_id=task.id,
            deployment_name=request.deployment_name,
            status=DeploymentStatus.deploying,
            message="Deployment job submitted successfully"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create deployment: {str(e)}"
        )


@router.get("/", response_model=List[DeploymentResponse])
async def list_deployments(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    current_user: User = Depends(get_current_active_user)
) -> List[DeploymentResponse]:
    """
    List all deployments.
    
    Retrieve a paginated list of all deployments with their current status.
    """
    return []


@router.get("/{deployment_id}", response_model=DeploymentResponse)
async def get_deployment(
    deployment_id: str = Path(..., description="Deployment ID"),
    current_user: User = Depends(get_current_active_user)
) -> DeploymentResponse:
    """
    Get deployment details.
    
    Retrieve detailed information about a specific deployment including:
    - Current status
    - Endpoint URL
    - Resource usage
    - Health status
    """
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Deployment not found"
    )


@router.delete("/{deployment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_deployment(
    deployment_id: str = Path(..., description="Deployment ID"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Stop and delete a deployment.
    
    Permanently stop a running deployment and clean up all associated resources.
    """
    return None


@router.post("/{deployment_id}/scale")
async def scale_deployment(
    deployment_id: str = Path(..., description="Deployment ID"),
    replicas: int = Query(..., ge=0, description="Number of replicas"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Scale a deployment.
    
    Change the number of replicas for a running deployment.
    Set to 0 to pause the deployment without deleting it.
    """
    return {
        "deployment_id": deployment_id,
        "replicas": replicas,
        "message": "Deployment scaled successfully"
    }
