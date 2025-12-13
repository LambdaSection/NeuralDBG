"""
Model management endpoints.
"""

import os
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Path as PathParam, Query, status
from fastapi.responses import FileResponse

from neural.api.auth import User, get_current_active_user
from neural.api.config import settings
from neural.api.models import ModelInfo, ModelListResponse, BackendType

router = APIRouter(prefix="/models", tags=["Models"])


@router.get("/", response_model=ModelListResponse)
async def list_models(
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of items to return"),
    backend_filter: BackendType = Query(None, description="Filter by backend"),
    current_user: User = Depends(get_current_active_user)
) -> ModelListResponse:
    """
    List all compiled models.
    
    Retrieve a paginated list of all compiled models with optional filtering by backend.
    """
    try:
        models_path = Path(settings.models_path)
        models_path.mkdir(parents=True, exist_ok=True)
        
        models = []
        
        for model_file in models_path.glob("**/*.py"):
            if model_file.is_file():
                stat = model_file.stat()
                
                backend = BackendType.tensorflow
                if "pytorch" in model_file.name or "_pt" in model_file.name:
                    backend = BackendType.pytorch
                elif "onnx" in model_file.name:
                    backend = BackendType.onnx
                
                if backend_filter and backend != backend_filter:
                    continue
                
                models.append(ModelInfo(
                    model_id=model_file.stem,
                    name=model_file.name,
                    backend=backend,
                    size=stat.st_size,
                    created_at=stat.st_ctime,
                    metadata={}
                ))
        
        total = len(models)
        models = models[skip:skip + limit]
        
        return ModelListResponse(
            models=models,
            total=total,
            skip=skip,
            limit=limit
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )


@router.get("/{model_id}", response_model=ModelInfo)
async def get_model(
    model_id: str = PathParam(..., description="Model ID"),
    current_user: User = Depends(get_current_active_user)
) -> ModelInfo:
    """
    Get model details.
    
    Retrieve detailed information about a specific compiled model.
    """
    try:
        models_path = Path(settings.models_path)
        
        for model_file in models_path.glob(f"**/{model_id}*"):
            if model_file.is_file():
                stat = model_file.stat()
                
                backend = BackendType.tensorflow
                if "pytorch" in model_file.name:
                    backend = BackendType.pytorch
                elif "onnx" in model_file.name:
                    backend = BackendType.onnx
                
                return ModelInfo(
                    model_id=model_id,
                    name=model_file.name,
                    backend=backend,
                    size=stat.st_size,
                    created_at=stat.st_ctime,
                    metadata={}
                )
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model: {str(e)}"
        )


@router.get("/{model_id}/download")
async def download_model(
    model_id: str = PathParam(..., description="Model ID"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Download a compiled model.
    
    Download the source code for a compiled model.
    """
    try:
        models_path = Path(settings.models_path)
        
        for model_file in models_path.glob(f"**/{model_id}*"):
            if model_file.is_file():
                return FileResponse(
                    path=str(model_file),
                    filename=model_file.name,
                    media_type="text/x-python"
                )
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download model: {str(e)}"
        )


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(
    model_id: str = PathParam(..., description="Model ID"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Delete a compiled model.
    
    Permanently delete a compiled model from storage.
    """
    try:
        models_path = Path(settings.models_path)
        
        deleted = False
        for model_file in models_path.glob(f"**/{model_id}*"):
            if model_file.is_file():
                model_file.unlink()
                deleted = True
        
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_id} not found"
            )
        
        return None
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete model: {str(e)}"
        )
