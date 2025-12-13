"""
Parallel execution engines for AutoML trials.

Supports sequential, Ray, and Dask execution backends.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class BaseExecutor(ABC):
    """Base class for trial executors."""
    
    def __init__(self, name: str, max_workers: Optional[int] = None):
        self.name = name
        self.max_workers = max_workers
    
    @abstractmethod
    def execute_trials(
        self,
        trial_fn: Callable,
        trial_configs: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Execute multiple trials in parallel."""
        pass
    
    @abstractmethod
    def shutdown(self):
        """Cleanup executor resources."""
        pass


class SequentialExecutor(BaseExecutor):
    """Sequential execution of trials (no parallelism)."""
    
    def __init__(self):
        super().__init__('sequential', max_workers=1)
    
    def execute_trials(
        self,
        trial_fn: Callable,
        trial_configs: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Execute trials sequentially."""
        results = []
        
        for i, config in enumerate(trial_configs):
            try:
                logger.info(f"Executing trial {i+1}/{len(trial_configs)}")
                result = trial_fn(config, trial_id=i, **kwargs)
                results.append({
                    'trial_id': i,
                    'config': config,
                    'result': result,
                    'success': True
                })
            except Exception as e:
                logger.error(f"Trial {i} failed: {e}")
                results.append({
                    'trial_id': i,
                    'config': config,
                    'result': None,
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    def shutdown(self):
        """No cleanup needed for sequential executor."""
        pass


class ThreadPoolExecutorWrapper(BaseExecutor):
    """Thread-based parallel execution."""
    
    def __init__(self, max_workers: Optional[int] = None):
        super().__init__('thread_pool', max_workers)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def execute_trials(
        self,
        trial_fn: Callable,
        trial_configs: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Execute trials using thread pool."""
        results = []
        futures = []
        
        for i, config in enumerate(trial_configs):
            future = self.executor.submit(trial_fn, config, trial_id=i, **kwargs)
            futures.append((i, config, future))
        
        for i, config, future in futures:
            try:
                result = future.result()
                results.append({
                    'trial_id': i,
                    'config': config,
                    'result': result,
                    'success': True
                })
            except Exception as e:
                logger.error(f"Trial {i} failed: {e}")
                results.append({
                    'trial_id': i,
                    'config': config,
                    'result': None,
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    def shutdown(self):
        """Shutdown thread pool."""
        self.executor.shutdown(wait=True)


class ProcessPoolExecutorWrapper(BaseExecutor):
    """Process-based parallel execution."""
    
    def __init__(self, max_workers: Optional[int] = None):
        super().__init__('process_pool', max_workers)
        self.executor = ProcessPoolExecutor(max_workers=max_workers)
    
    def execute_trials(
        self,
        trial_fn: Callable,
        trial_configs: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Execute trials using process pool."""
        results = []
        futures = []
        
        for i, config in enumerate(trial_configs):
            future = self.executor.submit(trial_fn, config, trial_id=i, **kwargs)
            futures.append((i, config, future))
        
        for i, config, future in futures:
            try:
                result = future.result()
                results.append({
                    'trial_id': i,
                    'config': config,
                    'result': result,
                    'success': True
                })
            except Exception as e:
                logger.error(f"Trial {i} failed: {e}")
                results.append({
                    'trial_id': i,
                    'config': config,
                    'result': None,
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    def shutdown(self):
        """Shutdown process pool."""
        self.executor.shutdown(wait=True)


class RayExecutor(BaseExecutor):
    """Ray-based distributed execution."""
    
    def __init__(self, max_workers: Optional[int] = None, address: Optional[str] = None):
        super().__init__('ray', max_workers)
        self.address = address
        self.ray_initialized = False
        self._initialize_ray()
    
    def _initialize_ray(self):
        """Initialize Ray cluster."""
        try:
            import ray
            
            if not ray.is_initialized():
                if self.address:
                    ray.init(address=self.address)
                else:
                    ray.init(num_cpus=self.max_workers, ignore_reinit_error=True)
                self.ray_initialized = True
                logger.info("Ray initialized successfully")
        
        except ImportError:
            logger.warning("Ray not available, falling back to sequential execution")
        except Exception as e:
            logger.error(f"Failed to initialize Ray: {e}")
    
    def execute_trials(
        self,
        trial_fn: Callable,
        trial_configs: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Execute trials using Ray."""
        try:
            import ray
            
            if not self.ray_initialized:
                logger.warning("Ray not initialized, using sequential execution")
                executor = SequentialExecutor()
                return executor.execute_trials(trial_fn, trial_configs, **kwargs)
            
            @ray.remote
            def ray_trial_fn(config, trial_id, **kw):
                return trial_fn(config, trial_id=trial_id, **kw)
            
            futures = []
            for i, config in enumerate(trial_configs):
                future = ray_trial_fn.remote(config, i, **kwargs)
                futures.append((i, config, future))
            
            results = []
            for i, config, future in futures:
                try:
                    result = ray.get(future)
                    results.append({
                        'trial_id': i,
                        'config': config,
                        'result': result,
                        'success': True
                    })
                except Exception as e:
                    logger.error(f"Ray trial {i} failed: {e}")
                    results.append({
                        'trial_id': i,
                        'config': config,
                        'result': None,
                        'success': False,
                        'error': str(e)
                    })
            
            return results
        
        except ImportError:
            logger.warning("Ray not available, using sequential execution")
            executor = SequentialExecutor()
            return executor.execute_trials(trial_fn, trial_configs, **kwargs)
    
    def shutdown(self):
        """Shutdown Ray."""
        if self.ray_initialized:
            try:
                import ray
                ray.shutdown()
                logger.info("Ray shutdown successfully")
            except Exception as e:
                logger.error(f"Error shutting down Ray: {e}")


class DaskExecutor(BaseExecutor):
    """Dask-based distributed execution."""
    
    def __init__(self, max_workers: Optional[int] = None, address: Optional[str] = None):
        super().__init__('dask', max_workers)
        self.address = address
        self.client = None
        self._initialize_dask()
    
    def _initialize_dask(self):
        """Initialize Dask client."""
        try:
            from dask.distributed import Client, LocalCluster
            
            if self.address:
                self.client = Client(self.address)
            else:
                cluster = LocalCluster(n_workers=self.max_workers or 4, threads_per_worker=1)
                self.client = Client(cluster)
            
            logger.info(f"Dask initialized: {self.client}")
        
        except ImportError:
            logger.warning("Dask not available, falling back to sequential execution")
        except Exception as e:
            logger.error(f"Failed to initialize Dask: {e}")
    
    def execute_trials(
        self,
        trial_fn: Callable,
        trial_configs: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Execute trials using Dask."""
        if self.client is None:
            logger.warning("Dask not initialized, using sequential execution")
            executor = SequentialExecutor()
            return executor.execute_trials(trial_fn, trial_configs, **kwargs)
        
        try:
            futures = []
            for i, config in enumerate(trial_configs):
                future = self.client.submit(trial_fn, config, trial_id=i, **kwargs)
                futures.append((i, config, future))
            
            results = []
            for i, config, future in futures:
                try:
                    result = future.result()
                    results.append({
                        'trial_id': i,
                        'config': config,
                        'result': result,
                        'success': True
                    })
                except Exception as e:
                    logger.error(f"Dask trial {i} failed: {e}")
                    results.append({
                        'trial_id': i,
                        'config': config,
                        'result': None,
                        'success': False,
                        'error': str(e)
                    })
            
            return results
        
        except Exception as e:
            logger.error(f"Dask execution failed: {e}")
            executor = SequentialExecutor()
            return executor.execute_trials(trial_fn, trial_configs, **kwargs)
    
    def shutdown(self):
        """Shutdown Dask client."""
        if self.client:
            try:
                self.client.close()
                logger.info("Dask client closed")
            except Exception as e:
                logger.error(f"Error closing Dask client: {e}")


def create_executor(
    executor_type: str = 'sequential',
    max_workers: Optional[int] = None,
    address: Optional[str] = None
) -> BaseExecutor:
    """
    Factory function to create executors.
    
    Args:
        executor_type: Type of executor ('sequential', 'thread', 'process', 'ray', 'dask')
        max_workers: Maximum number of parallel workers
        address: Address for distributed executors (Ray/Dask)
    
    Returns:
        Executor instance
    """
    if executor_type == 'sequential':
        return SequentialExecutor()
    elif executor_type == 'thread':
        return ThreadPoolExecutorWrapper(max_workers)
    elif executor_type == 'process':
        return ProcessPoolExecutorWrapper(max_workers)
    elif executor_type == 'ray':
        return RayExecutor(max_workers, address)
    elif executor_type == 'dask':
        return DaskExecutor(max_workers, address)
    else:
        logger.warning(f"Unknown executor type: {executor_type}, using sequential")
        return SequentialExecutor()
