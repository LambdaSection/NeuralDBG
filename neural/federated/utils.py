from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def split_data_iid(
    data: Tuple[np.ndarray, np.ndarray],
    num_clients: int,
    shuffle: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    X, y = data
    
    if shuffle:
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
    
    client_data = []
    samples_per_client = len(X) // num_clients
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client if i < num_clients - 1 else len(X)
        
        X_client = X[start_idx:end_idx]
        y_client = y[start_idx:end_idx]
        
        client_data.append((X_client, y_client))
    
    return client_data


def split_data_non_iid(
    data: Tuple[np.ndarray, np.ndarray],
    num_clients: int,
    alpha: float = 0.5,
    num_classes: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    X, y = data
    
    if num_classes is None:
        num_classes = len(np.unique(y))
    
    class_indices = [np.where(y == i)[0] for i in range(num_classes)]
    
    proportions = np.random.dirichlet([alpha] * num_clients, num_classes)
    
    client_indices = [[] for _ in range(num_clients)]
    
    for class_id, indices in enumerate(class_indices):
        np.random.shuffle(indices)
        
        class_proportions = proportions[class_id]
        splits = (np.cumsum(class_proportions) * len(indices)).astype(int)
        
        start_idx = 0
        for client_id, split_idx in enumerate(splits[:-1]):
            client_indices[client_id].extend(indices[start_idx:split_idx])
            start_idx = split_idx
        client_indices[-1].extend(indices[start_idx:])
    
    client_data = []
    for indices in client_indices:
        if len(indices) == 0:
            continue
        
        indices = np.array(indices)
        np.random.shuffle(indices)
        
        X_client = X[indices]
        y_client = y[indices]
        
        client_data.append((X_client, y_client))
    
    return client_data


def compute_data_statistics(
    client_data: List[Tuple[np.ndarray, np.ndarray]]
) -> Dict[str, Any]:
    stats = {
        'num_clients': len(client_data),
        'total_samples': 0,
        'samples_per_client': [],
        'class_distribution': [],
    }
    
    for X, y in client_data:
        num_samples = len(X)
        stats['total_samples'] += num_samples
        stats['samples_per_client'].append(num_samples)
        
        unique, counts = np.unique(y, return_counts=True)
        class_dist = dict(zip(unique.tolist(), counts.tolist()))
        stats['class_distribution'].append(class_dist)
    
    stats['avg_samples_per_client'] = np.mean(stats['samples_per_client'])
    stats['std_samples_per_client'] = np.std(stats['samples_per_client'])
    stats['min_samples_per_client'] = np.min(stats['samples_per_client'])
    stats['max_samples_per_client'] = np.max(stats['samples_per_client'])
    
    return stats


def compute_model_size(weights: List[np.ndarray]) -> int:
    return sum([w.nbytes for w in weights])


def compute_gradient_norm(weights: List[np.ndarray]) -> float:
    total_norm = 0.0
    for w in weights:
        total_norm += np.sum(np.square(w))
    return float(np.sqrt(total_norm))


def clip_weights(
    weights: List[np.ndarray],
    clip_norm: float,
) -> List[np.ndarray]:
    total_norm = compute_gradient_norm(weights)
    
    if total_norm <= clip_norm:
        return weights
    
    clip_coef = clip_norm / total_norm
    return [w * clip_coef for w in weights]


def cosine_similarity(
    weights1: List[np.ndarray],
    weights2: List[np.ndarray],
) -> float:
    if len(weights1) != len(weights2):
        raise ValueError("Weight lists must have the same length")
    
    dot_product = 0.0
    norm1 = 0.0
    norm2 = 0.0
    
    for w1, w2 in zip(weights1, weights2):
        if w1.shape != w2.shape:
            continue
        
        dot_product += np.sum(w1 * w2)
        norm1 += np.sum(w1 ** 2)
        norm2 += np.sum(w2 ** 2)
    
    norm1 = np.sqrt(norm1)
    norm2 = np.sqrt(norm2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


def compute_weight_divergence(
    client_weights: List[List[np.ndarray]],
    global_weights: List[np.ndarray],
) -> Dict[str, float]:
    divergences = []
    
    for weights in client_weights:
        similarity = cosine_similarity(weights, global_weights)
        divergence = 1 - similarity
        divergences.append(divergence)
    
    return {
        'mean_divergence': float(np.mean(divergences)),
        'std_divergence': float(np.std(divergences)),
        'max_divergence': float(np.max(divergences)),
        'min_divergence': float(np.min(divergences)),
    }


def detect_byzantine_clients(
    client_weights: List[List[np.ndarray]],
    global_weights: List[np.ndarray],
    threshold: float = 0.5,
) -> List[int]:
    byzantine_indices = []
    
    for i, weights in enumerate(client_weights):
        similarity = cosine_similarity(weights, global_weights)
        
        if similarity < threshold:
            byzantine_indices.append(i)
            logger.warning(f"Detected potential Byzantine client at index {i}")
    
    return byzantine_indices


def filter_byzantine_clients(
    client_results: List[Dict[str, Any]],
    global_weights: List[np.ndarray],
    threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    client_weights = [r['weights'] for r in client_results]
    byzantine_indices = detect_byzantine_clients(client_weights, global_weights, threshold)
    
    filtered_results = [
        r for i, r in enumerate(client_results)
        if i not in byzantine_indices
    ]
    
    logger.info(f"Filtered {len(byzantine_indices)} Byzantine clients")
    return filtered_results


def compute_communication_cost(
    weights: List[np.ndarray],
    compression_ratio: float = 1.0,
) -> Dict[str, float]:
    original_size = compute_model_size(weights)
    compressed_size = original_size * compression_ratio
    
    return {
        'original_size_bytes': original_size,
        'compressed_size_bytes': compressed_size,
        'original_size_mb': original_size / (1024 ** 2),
        'compressed_size_mb': compressed_size / (1024 ** 2),
        'compression_ratio': compression_ratio,
        'savings_ratio': 1 - compression_ratio,
    }


def estimate_training_time(
    num_clients: int,
    num_rounds: int,
    avg_client_time: float,
    parallel_clients: int = 1,
) -> Dict[str, float]:
    sequential_time = num_clients * num_rounds * avg_client_time
    parallel_time = (num_clients / parallel_clients) * num_rounds * avg_client_time
    
    return {
        'sequential_time_seconds': sequential_time,
        'parallel_time_seconds': parallel_time,
        'sequential_time_hours': sequential_time / 3600,
        'parallel_time_hours': parallel_time / 3600,
        'speedup': sequential_time / parallel_time if parallel_time > 0 else 0,
    }


def create_random_client_data(
    num_clients: int,
    num_samples_per_client: int,
    input_shape: Tuple[int, ...],
    num_classes: int,
    heterogeneous: bool = False,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    client_data = []
    
    for i in range(num_clients):
        if heterogeneous:
            actual_samples = np.random.randint(
                num_samples_per_client // 2,
                num_samples_per_client * 2
            )
        else:
            actual_samples = num_samples_per_client
        
        X = np.random.randn(actual_samples, *input_shape).astype(np.float32)
        y = np.random.randint(0, num_classes, actual_samples)
        
        client_data.append((X, y))
    
    return client_data


def aggregate_metrics(
    client_metrics: List[Dict[str, float]],
    num_samples_list: List[int],
    weighted: bool = True,
) -> Dict[str, float]:
    if not client_metrics:
        return {}
    
    aggregated = {}
    metric_keys = client_metrics[0].keys()
    
    total_samples = sum(num_samples_list)
    
    for key in metric_keys:
        if weighted and total_samples > 0:
            weighted_sum = sum([
                metrics[key] * num_samples
                for metrics, num_samples in zip(client_metrics, num_samples_list)
            ])
            aggregated[key] = weighted_sum / total_samples
        else:
            aggregated[key] = np.mean([m[key] for m in client_metrics])
    
    return aggregated


def save_federated_checkpoint(
    filepath: str,
    global_weights: List[np.ndarray],
    round_num: int,
    metrics: Dict[str, Any],
):
    import pickle
    
    checkpoint = {
        'round': round_num,
        'global_weights': global_weights,
        'metrics': metrics,
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    logger.info(f"Checkpoint saved to {filepath}")


def load_federated_checkpoint(filepath: str) -> Dict[str, Any]:
    import pickle
    
    with open(filepath, 'rb') as f:
        checkpoint = pickle.load(f)
    
    logger.info(f"Checkpoint loaded from {filepath}")
    return checkpoint
