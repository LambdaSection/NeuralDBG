"""
Utility functions for platform integrations.

Provides helper functions for common tasks across platforms.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from neural.exceptions import FileOperationError


logger = logging.getLogger(__name__)


def load_credentials_from_file(filepath: Optional[str] = None) -> Dict[str, Any]:
    """
    Load credentials from a JSON file.
    
    Args:
        filepath: Path to credentials file (default: ~/.neural/credentials.json)
        
    Returns:
        Dictionary of credentials
        
    Raises:
        FileOperationError: If file cannot be read
    """
    if filepath is None:
        config_dir = Path.home() / ".neural"
        filepath = str(config_dir / "credentials.json")
    
    try:
        with open(filepath, 'r') as f:
            credentials = json.load(f)
        logger.info(f"Loaded credentials from {filepath}")
        return credentials
    except FileNotFoundError:
        raise FileOperationError(
            operation="read",
            filepath=filepath,
            reason="File not found"
        )
    except json.JSONDecodeError as e:
        raise FileOperationError(
            operation="read",
            filepath=filepath,
            reason=f"Invalid JSON: {e}"
        )
    except Exception as e:
        raise FileOperationError(
            operation="read",
            filepath=filepath,
            reason=str(e)
        )


def save_credentials_to_file(
    credentials: Dict[str, Any],
    filepath: Optional[str] = None
) -> None:
    """
    Save credentials to a JSON file.
    
    Args:
        credentials: Dictionary of credentials
        filepath: Path to save credentials (default: ~/.neural/credentials.json)
        
    Raises:
        FileOperationError: If file cannot be written
    """
    if filepath is None:
        config_dir = Path.home() / ".neural"
        config_dir.mkdir(exist_ok=True)
        filepath = str(config_dir / "credentials.json")
    
    try:
        with open(filepath, 'w') as f:
            json.dump(credentials, f, indent=2)
        
        os.chmod(filepath, 0o600)
        logger.info(f"Saved credentials to {filepath}")
    except Exception as e:
        raise FileOperationError(
            operation="write",
            filepath=filepath,
            reason=str(e)
        )


def format_job_output(result: Any) -> str:
    """
    Format job result for display.
    
    Args:
        result: Job result object
        
    Returns:
        Formatted string
    """
    from .base import JobResult
    
    if not isinstance(result, JobResult):
        return str(result)
    
    lines = [
        f"Job ID: {result.job_id}",
        f"Status: {result.status.value}",
    ]
    
    if result.duration_seconds:
        lines.append(f"Duration: {result.duration_seconds:.2f}s")
    
    if result.error:
        lines.append(f"Error: {result.error}")
    
    if result.metrics:
        lines.append("Metrics:")
        for key, value in result.metrics.items():
            lines.append(f"  {key}: {value}")
    
    if result.logs_url:
        lines.append(f"Logs: {result.logs_url}")
    
    if result.output:
        lines.append(f"\nOutput:\n{result.output[:500]}")
        if len(result.output) > 500:
            lines.append("... (truncated)")
    
    return '\n'.join(lines)
