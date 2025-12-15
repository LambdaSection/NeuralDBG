"""
Utility functions for HPO operations.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


def extract_best_params(study, normalize: bool = True) -> Dict[str, Any]:
    """
    Extract best parameters from an Optuna study.
    
    Args:
        study: Optuna study object
        normalize: Whether to normalize parameter names
        
    Returns:
        Dictionary of best parameters
    """
    if not study or not study.best_trials:
        return {}
    
    best_trial = study.best_trials[0]
    params = best_trial.params.copy()
    
    if normalize:
        # Normalize parameter names
        normalized = {}
        for key, value in params.items():
            # Remove layer type prefix if present
            if '_' in key:
                parts = key.split('_')
                if len(parts) > 1:
                    normalized['_'.join(parts[1:])] = value
                else:
                    normalized[key] = value
            else:
                normalized[key] = value
        return normalized
    
    return params


def trials_to_dataframe(trials: List[Dict[str, Any]]) -> Any:
    """
    Convert trials list to pandas DataFrame for analysis.
    
    Args:
        trials: List of trial dictionaries
        
    Returns:
        pandas DataFrame or None if pandas not available
    """
    try:
        import pandas as pd
        
        if not trials:
            return pd.DataFrame()
        
        # Extract parameters and metrics
        records = []
        for trial in trials:
            record = {}
            
            # Add trial number
            record['trial_number'] = trial.get('trial_number', None)
            
            # Add parameters
            params = trial.get('parameters', {})
            for param, value in params.items():
                record[f'param_{param}'] = value
            
            # Add metrics
            for key in ['loss', 'accuracy', 'precision', 'recall', 'score']:
                if key in trial:
                    record[key] = trial[key]
            
            records.append(record)
        
        return pd.DataFrame(records)
    
    except ImportError:
        logger.warning("pandas not available. Install with: pip install pandas")
        return None


def save_trials(trials: List[Dict[str, Any]], filepath: str) -> None:
    """
    Save trials to a JSON file.
    
    Args:
        trials: List of trial dictionaries
        filepath: Path to save file
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(trials, f, indent=2, default=str)
        logger.info(f"Saved {len(trials)} trials to {filepath}")
    except Exception as e:
        logger.error(f"Error saving trials: {e}")


def load_trials(filepath: str) -> List[Dict[str, Any]]:
    """
    Load trials from a JSON file.
    
    Args:
        filepath: Path to trials file
        
    Returns:
        List of trial dictionaries
    """
    try:
        with open(filepath, 'r') as f:
            trials = json.load(f)
        logger.info(f"Loaded {len(trials)} trials from {filepath}")
        return trials
    except Exception as e:
        logger.error(f"Error loading trials: {e}")
        return []


def compute_pareto_front(objectives: np.ndarray, 
                        directions: List[str]) -> np.ndarray:
    """
    Compute Pareto front from multi-objective results.
    
    Args:
        objectives: Array of shape (n_trials, n_objectives)
        directions: List of 'minimize' or 'maximize' for each objective
        
    Returns:
        Boolean mask indicating Pareto optimal points
    """
    n_points = objectives.shape[0]
    pareto_mask = np.ones(n_points, dtype=bool)
    
    # Adjust signs based on optimization direction
    adjusted_obj = objectives.copy()
    for i, direction in enumerate(directions):
        if direction == 'maximize':
            adjusted_obj[:, i] = -adjusted_obj[:, i]
    
    # Find Pareto optimal points
    for i in range(n_points):
        for j in range(n_points):
            if i != j and pareto_mask[i]:
                # Check if point j dominates point i
                if np.all(adjusted_obj[j] <= adjusted_obj[i]) and \
                   np.any(adjusted_obj[j] < adjusted_obj[i]):
                    pareto_mask[i] = False
                    break
    
    return pareto_mask


def compute_hypervolume(pareto_front: np.ndarray, 
                       reference_point: np.ndarray) -> float:
    """
    Compute hypervolume indicator for Pareto front.
    
    Args:
        pareto_front: Array of Pareto optimal points
        reference_point: Reference point for hypervolume calculation
        
    Returns:
        Hypervolume value
    """
    try:
        # Sort points by first objective
        sorted_front = pareto_front[pareto_front[:, 0].argsort()]
        
        # Compute hypervolume using 2D method
        if sorted_front.shape[1] == 2:
            hv = 0.0
            for i in range(len(sorted_front)):
                if i == 0:
                    width = reference_point[0] - sorted_front[i, 0]
                else:
                    width = sorted_front[i-1, 0] - sorted_front[i, 0]
                height = reference_point[1] - sorted_front[i, 1]
                hv += width * height
            return hv
        else:
            logger.warning("Hypervolume calculation only supported for 2 objectives")
            return 0.0
    except Exception as e:
        logger.error(f"Error computing hypervolume: {e}")
        return 0.0


def suggest_best_compromise(pareto_front: List[Dict[str, Any]], 
                           weights: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    Suggest best compromise solution from Pareto front using weighted sum.
    
    Args:
        pareto_front: List of Pareto optimal solutions
        weights: Weights for each objective (default: equal weights)
        
    Returns:
        Best compromise solution
    """
    if not pareto_front:
        return {}
    
    if len(pareto_front) == 1:
        return pareto_front[0]
    
    # Extract objective names and values
    obj_names = [k for k in pareto_front[0].keys() if k != 'parameters']
    n_objectives = len(obj_names)
    
    if weights is None:
        weights = [1.0 / n_objectives] * n_objectives
    
    # Normalize objectives to [0, 1]
    obj_values = np.array([[sol[obj] for obj in obj_names] for sol in pareto_front])
    obj_min = obj_values.min(axis=0)
    obj_max = obj_values.max(axis=0)
    obj_normalized = (obj_values - obj_min) / (obj_max - obj_min + 1e-8)
    
    # Compute weighted sum
    weighted_scores = obj_normalized @ np.array(weights)
    
    # Return solution with highest weighted score
    best_idx = np.argmax(weighted_scores)
    return pareto_front[best_idx]


def estimate_remaining_time(trials: List[Dict[str, Any]], 
                           n_total_trials: int) -> float:
    """
    Estimate remaining time for optimization.
    
    Args:
        trials: Completed trials
        n_total_trials: Total number of trials
        
    Returns:
        Estimated remaining time in seconds
    """
    if not trials:
        return 0.0
    
    # Estimate average time per trial
    # This is a simple heuristic; actual implementation would track timestamps
    avg_time_per_trial = 60.0  # Placeholder: 60 seconds
    
    n_completed = len(trials)
    n_remaining = max(0, n_total_trials - n_completed)
    
    return avg_time_per_trial * n_remaining


def get_parameter_ranges(trials: List[Dict[str, Any]]) -> Dict[str, Tuple[Any, Any]]:
    """
    Extract parameter ranges from trials.
    
    Args:
        trials: List of trial dictionaries
        
    Returns:
        Dictionary mapping parameter names to (min, max) tuples
    """
    ranges = {}
    
    for trial in trials:
        params = trial.get('parameters', {})
        for param, value in params.items():
            if isinstance(value, (int, float)):
                if param not in ranges:
                    ranges[param] = (value, value)
                else:
                    current_min, current_max = ranges[param]
                    ranges[param] = (min(current_min, value), max(current_max, value))
    
    return ranges


def filter_trials_by_metric(trials: List[Dict[str, Any]], 
                           metric: str,
                           threshold: float,
                           mode: str = 'greater') -> List[Dict[str, Any]]:
    """
    Filter trials by metric threshold.
    
    Args:
        trials: List of trial dictionaries
        metric: Metric name to filter by
        threshold: Threshold value
        mode: 'greater' or 'less'
        
    Returns:
        Filtered list of trials
    """
    filtered = []
    
    for trial in trials:
        value = trial.get(metric, None)
        if value is None:
            continue
        
        if mode == 'greater' and value > threshold:
            filtered.append(trial)
        elif mode == 'less' and value < threshold:
            filtered.append(trial)
    
    return filtered


def summarize_trials(trials: List[Dict[str, Any]], 
                    metrics: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Summarize trial statistics.
    
    Args:
        trials: List of trial dictionaries
        metrics: List of metrics to summarize (default: all numeric metrics)
        
    Returns:
        Summary statistics dictionary
    """
    if not trials:
        return {}
    
    summary = {
        'n_trials': len(trials),
        'metrics': {}
    }
    
    # Auto-detect numeric metrics if not specified
    if metrics is None:
        metrics = []
        for key in trials[0].keys():
            if key != 'parameters' and isinstance(trials[0].get(key), (int, float)):
                metrics.append(key)
    
    # Compute statistics for each metric
    for metric in metrics:
        values = [t.get(metric) for t in trials if t.get(metric) is not None]
        
        if values:
            summary['metrics'][metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
    
    return summary


def create_search_space_from_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create search space dictionary from configuration.
    
    Args:
        config_dict: Configuration dictionary with HPO specifications
        
    Returns:
        Search space dictionary for Ray Tune or similar
    """
    search_space = {}
    
    # This is a placeholder implementation
    # Actual implementation would parse the config_dict structure
    # to extract HPO parameter definitions
    
    return search_space
