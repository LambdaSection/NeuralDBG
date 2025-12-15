"""
Main AutoML engine orchestrating the search process.

Integrates search strategies, early stopping, parallel execution, and evaluation.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .architecture_space import ArchitectureSpace
from .early_stopping import EarlyStoppingStrategy, MedianPruner
from .evaluation import ArchitectureEvaluator, MetricTracker, PerformancePredictor
from .executor import BaseExecutor, create_executor
from .search_strategies import RandomSearchStrategy, SearchStrategy


logger = logging.getLogger(__name__)


class AutoMLEngine:
    """
    Main AutoML engine for neural architecture search and hyperparameter optimization.
    
    Integrates:
    - Architecture space definition
    - Multiple search strategies (Grid, Random, Bayesian, Evolutionary)
    - Early stopping mechanisms
    - Parallel execution backends (Sequential, Ray, Dask)
    - HPO module integration
    """
    
    def __init__(
        self,
        search_strategy: str = 'random',
        early_stopping: Optional[str] = 'median',
        executor_type: str = 'sequential',
        max_workers: Optional[int] = None,
        backend: str = 'pytorch',
        device: str = 'auto',
        output_dir: Optional[str] = None
    ):
        """
        Initialize AutoML engine.
        
        Args:
            search_strategy: Search strategy ('grid', 'random', 'bayesian', 'evolutionary')
            early_stopping: Early stopping strategy ('median', 'hyperband', 'asha', 'threshold', None)
            executor_type: Execution backend ('sequential', 'thread', 'process', 'ray', 'dask')
            max_workers: Maximum parallel workers
            backend: ML backend ('pytorch', 'tensorflow')
            device: Device for training ('auto', 'cpu', 'cuda', 'mps')
            output_dir: Directory for saving results
        """
        self.search_strategy_name = search_strategy
        self.early_stopping_name = early_stopping
        self.executor_type = executor_type
        self.max_workers = max_workers
        self.backend = backend
        self.device = device
        self.output_dir = Path(output_dir) if output_dir else Path('./automl_results')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.search_strategy: Optional[SearchStrategy] = None
        self.early_stopping: Optional[EarlyStoppingStrategy] = None
        self.executor: Optional[BaseExecutor] = None
        self.evaluator: ArchitectureEvaluator = ArchitectureEvaluator(
            backend=backend,
            device=device
        )
        self.predictor: PerformancePredictor = PerformancePredictor()
        
        self.trial_history: List[Dict[str, Any]] = []
        self.best_architecture: Optional[Dict[str, Any]] = None
        self.best_metrics: Optional[Dict[str, float]] = None
    
    def search(
        self,
        architecture_space: ArchitectureSpace,
        train_data,
        val_data,
        max_trials: int = 100,
        max_epochs_per_trial: int = 10,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute AutoML search.
        
        Args:
            architecture_space: Architecture search space
            train_data: Training data loader
            val_data: Validation data loader
            max_trials: Maximum number of trials
            max_epochs_per_trial: Maximum epochs per trial
            timeout: Maximum time in seconds (None for no limit)
            **kwargs: Additional arguments for search strategy
        
        Returns:
            Dictionary containing best architecture and search results
        """
        self._initialize_components(architecture_space, **kwargs)
        
        start_time = time.time()
        
        logger.info(f"Starting AutoML search with {max_trials} trials")
        logger.info(f"Search strategy: {self.search_strategy_name}")
        logger.info(f"Early stopping: {self.early_stopping_name}")
        logger.info(f"Executor: {self.executor_type}")
        
        try:
            for trial_id in range(max_trials):
                if timeout and (time.time() - start_time) > timeout:
                    logger.info(f"Timeout reached after {trial_id} trials")
                    break
                
                architecture = self.search_strategy.suggest(architecture_space, trial_id)
                
                logger.info(f"Trial {trial_id + 1}/{max_trials}: Evaluating architecture")
                
                metric_tracker = MetricTracker()
                
                should_stop = False
                for epoch in range(max_epochs_per_trial):
                    metrics = self._evaluate_epoch(
                        architecture,
                        train_data,
                        val_data,
                        epoch,
                        metric_tracker
                    )
                    
                    if self.early_stopping and self.early_stopping.should_stop(
                        trial_id,
                        epoch,
                        metrics,
                        self.trial_history
                    ):
                        logger.info(f"Trial {trial_id} stopped early at epoch {epoch}")
                        should_stop = True
                        break
                
                final_metrics = metric_tracker.get_summary()
                
                trial_result = {
                    'trial_id': trial_id,
                    'architecture': architecture,
                    'metrics': final_metrics,
                    'stopped_early': should_stop,
                    'metric_tracker': metric_tracker,
                    'step_metrics': metric_tracker.step_metrics
                }
                
                self.trial_history.append(trial_result)
                self.search_strategy.update(architecture, final_metrics)
                
                if self._is_better(final_metrics, self.best_metrics):
                    self.best_architecture = architecture
                    self.best_metrics = final_metrics
                    logger.info(f"New best architecture found! Accuracy: {final_metrics.get('val_acc', {}).get('max', 0):.4f}")
                
                self._save_checkpoint(trial_id)
        
        finally:
            self._cleanup()
        
        total_time = time.time() - start_time
        
        results = {
            'best_architecture': self.best_architecture,
            'best_metrics': self.best_metrics,
            'total_trials': len(self.trial_history),
            'total_time': total_time,
            'trial_history': self.trial_history,
            'search_strategy': self.search_strategy_name,
            'early_stopping': self.early_stopping_name
        }
        
        self._save_final_results(results)
        
        logger.info(f"AutoML search completed in {total_time:.2f}s")
        logger.info(f"Best accuracy: {self.best_metrics.get('val_acc', {}).get('max', 0):.4f}")
        
        return results
    
    def search_with_hpo(
        self,
        dsl_config: str,
        train_data,
        val_data,
        n_trials: int = 50,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Search using HPO module integration.
        
        Args:
            dsl_config: DSL configuration with HPO parameters
            train_data: Training data
            val_data: Validation data
            n_trials: Number of trials
            **kwargs: Additional HPO arguments
        
        Returns:
            Best parameters and architecture
        """
        try:
            from neural.hpo.hpo import optimize_and_return
            
            logger.info("Running HPO-integrated search")
            
            best_params = optimize_and_return(
                config=dsl_config,
                n_trials=n_trials,
                backend=self.backend,
                device=self.device,
                **kwargs
            )
            
            architecture_space = ArchitectureSpace.from_dsl(dsl_config)
            
            results = {
                'best_parameters': best_params,
                'architecture_space': architecture_space,
                'n_trials': n_trials
            }
            
            return results
        
        except Exception as e:
            logger.error(f"HPO-integrated search failed: {e}")
            raise
    
    def _initialize_components(self, architecture_space: ArchitectureSpace, **kwargs):
        """Initialize search components."""
        from .search_strategies import (
            BayesianSearchStrategy,
            EvolutionarySearchStrategy,
            GridSearchStrategy,
        )
        try:
            from .search_strategies import RegularizedEvolutionStrategy
        except ImportError:
            RegularizedEvolutionStrategy = None
        
        from .early_stopping import ASHAPruner, HyperbandPruner, ThresholdPruner
        
        if self.search_strategy_name == 'grid':
            self.search_strategy = GridSearchStrategy()
        elif self.search_strategy_name == 'random':
            self.search_strategy = RandomSearchStrategy(seed=kwargs.get('seed'))
        elif self.search_strategy_name == 'bayesian':
            self.search_strategy = BayesianSearchStrategy(
                acquisition_function=kwargs.get('acquisition_function', 'ei'),
                n_initial_random=kwargs.get('n_initial_random', 10)
            )
        elif self.search_strategy_name == 'evolutionary':
            self.search_strategy = EvolutionarySearchStrategy(
                population_size=kwargs.get('population_size', 20),
                mutation_rate=kwargs.get('mutation_rate', 0.2),
                crossover_rate=kwargs.get('crossover_rate', 0.5)
            )
        elif self.search_strategy_name == 'regularized_evolution':
            self.search_strategy = RegularizedEvolutionStrategy(
                population_size=kwargs.get('population_size', 20),
                sample_size=kwargs.get('sample_size', 10)
            )
        else:
            logger.warning(f"Unknown search strategy: {self.search_strategy_name}, using random")
            self.search_strategy = RandomSearchStrategy()
        
        if self.early_stopping_name == 'median':
            self.early_stopping = MedianPruner(
                n_startup_trials=kwargs.get('n_startup_trials', 5),
                n_warmup_steps=kwargs.get('n_warmup_steps', 5)
            )
        elif self.early_stopping_name == 'hyperband':
            self.early_stopping = HyperbandPruner(
                max_resource=kwargs.get('max_resource', 100),
                reduction_factor=kwargs.get('reduction_factor', 3)
            )
        elif self.early_stopping_name == 'asha':
            self.early_stopping = ASHAPruner(
                reduction_factor=kwargs.get('reduction_factor', 4),
                min_resource=kwargs.get('min_resource', 1)
            )
        elif self.early_stopping_name == 'threshold':
            self.early_stopping = ThresholdPruner(
                lower_threshold=kwargs.get('lower_threshold'),
                upper_threshold=kwargs.get('upper_threshold')
            )
        elif self.early_stopping_name is None:
            self.early_stopping = None
        else:
            logger.warning(f"Unknown early stopping: {self.early_stopping_name}, using median")
            self.early_stopping = MedianPruner()
        
        self.executor = create_executor(
            executor_type=self.executor_type,
            max_workers=self.max_workers
        )
    
    def _evaluate_epoch(
        self,
        architecture: Dict[str, Any],
        train_data,
        val_data,
        epoch: int,
        metric_tracker: MetricTracker
    ) -> Dict[str, float]:
        """Evaluate one epoch of training."""
        try:
            model = self.evaluator._build_model(architecture)
            optimizer = self.evaluator._create_optimizer(model, architecture)
            
            train_metrics = self.evaluator._train_epoch(model, optimizer, train_data, epoch)
            val_metrics = self.evaluator._validate(model, val_data, epoch)
            
            metrics = {
                'train_loss': train_metrics['loss'],
                'train_acc': train_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                'accuracy': val_metrics['accuracy'],
                'loss': val_metrics['loss']
            }
            
            metric_tracker.log(epoch, **metrics)
            
            return metrics
        
        except Exception as e:
            logger.error(f"Epoch evaluation failed: {e}")
            return {
                'train_loss': float('inf'),
                'train_acc': 0.0,
                'val_loss': float('inf'),
                'val_acc': 0.0,
                'accuracy': 0.0,
                'loss': float('inf')
            }
    
    def _is_better(
        self,
        metrics1: Optional[Dict[str, Any]],
        metrics2: Optional[Dict[str, Any]]
    ) -> bool:
        """Check if metrics1 is better than metrics2."""
        if metrics2 is None:
            return True
        if metrics1 is None:
            return False
        
        acc1 = metrics1.get('val_acc', {}).get('max', 0) if isinstance(metrics1.get('val_acc'), dict) else 0
        acc2 = metrics2.get('val_acc', {}).get('max', 0) if isinstance(metrics2.get('val_acc'), dict) else 0
        
        return acc1 > acc2
    
    def _save_checkpoint(self, trial_id: int):
        """Save search checkpoint."""
        checkpoint = {
            'trial_id': trial_id,
            'best_architecture': self.best_architecture,
            'best_metrics': self.best_metrics,
            'num_trials': len(self.trial_history)
        }
        
        checkpoint_path = self.output_dir / f'checkpoint_{trial_id}.json'
        
        try:
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
    
    def _save_final_results(self, results: Dict[str, Any]):
        """Save final search results."""
        results_path = self.output_dir / 'final_results.json'
        
        try:
            serializable_results = {
                'best_architecture': results['best_architecture'],
                'best_metrics': results['best_metrics'],
                'total_trials': results['total_trials'],
                'total_time': results['total_time'],
                'search_strategy': results['search_strategy'],
                'early_stopping': results['early_stopping']
            }
            
            with open(results_path, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            logger.info(f"Results saved to {results_path}")
        
        except Exception as e:
            logger.warning(f"Failed to save final results: {e}")
    
    def _cleanup(self):
        """Cleanup resources."""
        if self.executor:
            self.executor.shutdown()
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load a search checkpoint."""
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            
            self.best_architecture = checkpoint['best_architecture']
            self.best_metrics = checkpoint['best_metrics']
            
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            return checkpoint
        
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def get_search_summary(self) -> Dict[str, Any]:
        """Get summary of the search process."""
        if not self.trial_history:
            return {}
        
        accuracies = []
        for trial in self.trial_history:
            metrics = trial.get('metrics', {})
            val_acc = metrics.get('val_acc', {})
            if isinstance(val_acc, dict):
                acc = val_acc.get('max', 0)
            else:
                acc = val_acc if val_acc else 0
            accuracies.append(acc)
        
        summary = {
            'num_trials': len(self.trial_history),
            'best_accuracy': max(accuracies) if accuracies else 0,
            'mean_accuracy': np.mean(accuracies) if accuracies else 0,
            'std_accuracy': np.std(accuracies) if accuracies else 0,
            'improvement': max(accuracies) - accuracies[0] if len(accuracies) > 0 else 0,
            'search_strategy': self.search_strategy_name,
            'early_stopping': self.early_stopping_name
        }
        
        return summary
