"""
Neural DSL Platform Integrations Module.

This module provides base integration classes for building ML platform connectors.

Classes
-------
BaseConnector
    Abstract base class for all platform connectors
ResourceConfig
    Configuration for compute resources
JobStatus
    Enumeration of job statuses
JobResult
    Result from a remote job execution

Examples
--------
>>> from neural.integrations import BaseConnector, ResourceConfig, JobStatus
>>> # Implement your own connector by subclassing BaseConnector
"""

from .base import BaseConnector, JobResult, JobStatus, ResourceConfig


__all__ = [
    'BaseConnector',
    'ResourceConfig',
    'JobStatus',
    'JobResult',
]
