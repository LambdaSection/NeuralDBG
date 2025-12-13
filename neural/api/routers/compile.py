"""
Compilation endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status

from neural.api.auth import User, get_current_active_user
from neural.api.models import CompileRequest, CompileResponse, JobStatus
from neural.api.tasks import compile_model

router = APIRouter(prefix="/compile", tags=["Compilation"])


@router.post("/", response_model=CompileResponse, status_code=status.HTTP_202_ACCEPTED)
async def compile_dsl(
    request: CompileRequest,
    current_user: User = Depends(get_current_active_user)
) -> CompileResponse:
    """
    Compile Neural DSL code to target backend.
    
    This endpoint accepts Neural DSL source code and compiles it asynchronously
    to the specified backend (TensorFlow, PyTorch, or ONNX).
    
    - **dsl_code**: The Neural DSL source code to compile
    - **backend**: Target backend framework
    - **dataset**: Dataset name for the model
    - **auto_flatten_output**: Automatically insert Flatten layer before Dense layers
    - **enable_hpo**: Enable hyperparameter optimization
    
    Returns a job ID that can be used to check compilation status.
    """
    try:
        task = compile_model.apply_async(
            kwargs={
                "dsl_code": request.dsl_code,
                "backend": request.backend.value,
                "dataset": request.dataset,
                "auto_flatten_output": request.auto_flatten_output,
                "enable_hpo": request.enable_hpo
            }
        )
        
        return CompileResponse(
            job_id=task.id,
            status=JobStatus.pending,
            message="Compilation job submitted successfully"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit compilation job: {str(e)}"
        )


@router.post("/sync", response_model=CompileResponse)
async def compile_dsl_sync(
    request: CompileRequest,
    current_user: User = Depends(get_current_active_user)
) -> CompileResponse:
    """
    Compile Neural DSL code synchronously.
    
    This endpoint performs synchronous compilation, returning the result immediately.
    Use this for small models or when you need immediate results.
    """
    try:
        from neural.parser.parser import create_parser, ModelTransformer
        from neural.code_generation import generate_code
        
        parser = create_parser()
        tree = parser.parse(request.dsl_code)
        transformer = ModelTransformer()
        model_dict = transformer.transform(tree)
        
        code = generate_code(
            model_dict,
            backend=request.backend.value,
            dataset=request.dataset,
            auto_flatten_output=request.auto_flatten_output
        )
        
        return CompileResponse(
            job_id="sync",
            status=JobStatus.completed,
            message="Compilation completed successfully",
            compiled_code=code
        )
    except Exception as e:
        return CompileResponse(
            job_id="sync",
            status=JobStatus.failed,
            message="Compilation failed",
            errors=[str(e)]
        )
