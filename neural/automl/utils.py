"""
Utility functions for AutoML.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def hash_architecture(architecture: Dict[str, Any]) -> str:
    """
    Generate a hash for an architecture.
    
    Args:
        architecture: Architecture configuration
    
    Returns:
        SHA256 hash string
    """
    arch_str = json.dumps(architecture, sort_keys=True)
    return hashlib.sha256(arch_str.encode()).hexdigest()


def compare_architectures(arch1: Dict[str, Any], arch2: Dict[str, Any]) -> float:
    """
    Calculate similarity score between two architectures.
    
    Args:
        arch1: First architecture
        arch2: Second architecture
    
    Returns:
        Similarity score between 0 and 1
    """
    layers1 = arch1.get('layers', [])
    layers2 = arch2.get('layers', [])
    
    if len(layers1) != len(layers2):
        return 0.0
    
    matches = 0
    for l1, l2 in zip(layers1, layers2):
        if l1.get('type') == l2.get('type'):
            matches += 1
            
            params1 = l1.get('params', {})
            params2 = l2.get('params', {})
            
            if params1 == params2:
                matches += 0.5
    
    max_score = len(layers1) * 1.5
    return matches / max_score if max_score > 0 else 0.0


def estimate_training_time(
    architecture: Dict[str, Any],
    dataset_size: int,
    batch_size: int = 32,
    epochs: int = 10
) -> float:
    """
    Estimate training time for an architecture.
    
    Args:
        architecture: Architecture configuration
        dataset_size: Number of training samples
        batch_size: Batch size
        epochs: Number of epochs
    
    Returns:
        Estimated training time in seconds
    """
    from neural.automl.nas_operations import estimate_model_size, compute_flops
    
    num_params = estimate_model_size(architecture)
    
    input_shape = architecture.get('input', {}).get('shape', (224, 224, 3))
    flops = compute_flops(architecture, input_shape)
    
    num_batches = (dataset_size + batch_size - 1) // batch_size
    
    time_per_sample = (flops / 1e9) * 0.001
    
    time_per_epoch = time_per_sample * dataset_size
    
    total_time = time_per_epoch * epochs
    
    return total_time


def filter_duplicate_architectures(
    architectures: List[Dict[str, Any]],
    threshold: float = 0.95
) -> List[Dict[str, Any]]:
    """
    Filter out duplicate or very similar architectures.
    
    Args:
        architectures: List of architectures
        threshold: Similarity threshold for considering duplicates
    
    Returns:
        Filtered list of unique architectures
    """
    if not architectures:
        return []
    
    unique = [architectures[0]]
    
    for arch in architectures[1:]:
        is_duplicate = False
        
        for unique_arch in unique:
            similarity = compare_architectures(arch, unique_arch)
            if similarity >= threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique.append(arch)
    
    return unique


def validate_architecture(architecture: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate an architecture configuration.
    
    Args:
        architecture: Architecture to validate
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(architecture, dict):
        return False, "Architecture must be a dictionary"
    
    if 'layers' not in architecture:
        return False, "Architecture must have 'layers' key"
    
    layers = architecture['layers']
    if not isinstance(layers, list):
        return False, "Layers must be a list"
    
    if len(layers) == 0:
        return False, "Architecture must have at least one layer"
    
    for i, layer in enumerate(layers):
        if not isinstance(layer, dict):
            return False, f"Layer {i} must be a dictionary"
        
        if 'type' not in layer:
            return False, f"Layer {i} must have 'type' key"
    
    return True, None


def normalize_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize metrics to [0, 1] range.
    
    Args:
        metrics: Dictionary of metrics
    
    Returns:
        Normalized metrics
    """
    normalized = {}
    
    for key, value in metrics.items():
        if 'accuracy' in key.lower() or 'acc' in key.lower():
            normalized[key] = np.clip(value, 0, 1)
        elif 'loss' in key.lower():
            normalized[key] = 1.0 / (1.0 + value)
        elif 'time' in key.lower():
            normalized[key] = 1.0 / (1.0 + np.log1p(value))
        else:
            normalized[key] = value
    
    return normalized


def merge_search_results(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge results from multiple search runs.
    
    Args:
        results_list: List of result dictionaries
    
    Returns:
        Merged results
    """
    if not results_list:
        return {}
    
    all_trials = []
    best_architecture = None
    best_metrics = None
    
    for results in results_list:
        all_trials.extend(results.get('trial_history', []))
        
        if best_metrics is None or (
            results.get('best_metrics', {}).get('accuracy', 0) >
            best_metrics.get('accuracy', 0)
        ):
            best_architecture = results.get('best_architecture')
            best_metrics = results.get('best_metrics')
    
    merged = {
        'best_architecture': best_architecture,
        'best_metrics': best_metrics,
        'total_trials': len(all_trials),
        'trial_history': all_trials,
        'num_runs': len(results_list)
    }
    
    return merged


def create_architecture_summary(architecture: Dict[str, Any]) -> str:
    """
    Create a human-readable summary of an architecture.
    
    Args:
        architecture: Architecture configuration
    
    Returns:
        Summary string
    """
    layers = architecture.get('layers', [])
    
    summary_parts = [
        f"Architecture with {len(layers)} layers:",
        ""
    ]
    
    for i, layer in enumerate(layers):
        layer_type = layer.get('type', 'Unknown')
        params = layer.get('params', {})
        
        param_str = ', '.join(f"{k}={v}" for k, v in params.items())
        summary_parts.append(f"  {i+1}. {layer_type}({param_str})")
    
    optimizer = architecture.get('optimizer', {})
    if optimizer:
        opt_type = optimizer.get('type', 'Unknown')
        opt_params = optimizer.get('params', {})
        lr = opt_params.get('learning_rate', 'N/A')
        summary_parts.append(f"\nOptimizer: {opt_type}(lr={lr})")
    
    training_config = architecture.get('training_config', {})
    if training_config:
        batch_size = training_config.get('batch_size', 'N/A')
        epochs = training_config.get('epochs', 'N/A')
        summary_parts.append(f"Training: batch_size={batch_size}, epochs={epochs}")
    
    return '\n'.join(summary_parts)


def export_architecture_to_dsl(architecture: Dict[str, Any]) -> str:
    """
    Export an architecture back to DSL format.
    
    Args:
        architecture: Architecture configuration
    
    Returns:
        DSL string
    """
    dsl_parts = ["network ExportedArchitecture {"]
    
    input_shape = architecture.get('input', {}).get('shape', (224, 224, 3))
    dsl_parts.append(f"    input: {input_shape}")
    dsl_parts.append("")
    
    for layer in architecture.get('layers', []):
        layer_type = layer.get('type', 'Unknown')
        params = layer.get('params', {})
        
        if params:
            param_str = ', '.join(f"{k}: {v}" for k, v in params.items())
            dsl_parts.append(f"    {layer_type}({param_str})")
        else:
            dsl_parts.append(f"    {layer_type}()")
    
    optimizer = architecture.get('optimizer', {})
    if optimizer:
        opt_type = optimizer.get('type', 'adam').lower()
        opt_params = optimizer.get('params', {})
        lr = opt_params.get('learning_rate', 0.001)
        dsl_parts.append("")
        dsl_parts.append(f"    optimizer: {opt_type}(learning_rate: {lr})")
    
    training_config = architecture.get('training_config', {})
    if training_config:
        dsl_parts.append("    training: {")
        for key, value in training_config.items():
            dsl_parts.append(f"        {key}: {value}")
        dsl_parts.append("    }")
    
    dsl_parts.append("}")
    
    return '\n'.join(dsl_parts)


class ArchitectureRegistry:
    """Registry for tracking evaluated architectures."""
    
    def __init__(self):
        self.architectures: Dict[str, Dict[str, Any]] = {}
    
    def register(self, architecture: Dict[str, Any], metrics: Dict[str, float]):
        """Register an architecture with its metrics."""
        arch_hash = hash_architecture(architecture)
        
        if arch_hash in self.architectures:
            existing_metrics = self.architectures[arch_hash]['metrics']
            if metrics.get('accuracy', 0) > existing_metrics.get('accuracy', 0):
                self.architectures[arch_hash]['metrics'] = metrics
        else:
            self.architectures[arch_hash] = {
                'architecture': architecture,
                'metrics': metrics,
                'hash': arch_hash
            }
    
    def get(self, architecture: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached results for an architecture."""
        arch_hash = hash_architecture(architecture)
        return self.architectures.get(arch_hash)
    
    def get_top_k(self, k: int = 10, metric: str = 'accuracy') -> List[Dict[str, Any]]:
        """Get top k architectures by a metric."""
        sorted_archs = sorted(
            self.architectures.values(),
            key=lambda x: x['metrics'].get(metric, 0),
            reverse=True
        )
        return sorted_archs[:k]
    
    def size(self) -> int:
        """Get number of registered architectures."""
        return len(self.architectures)
