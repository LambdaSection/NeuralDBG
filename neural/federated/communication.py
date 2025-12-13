from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CompressionStrategy(ABC):
    def __init__(self, compression_ratio: float = 0.5):
        self.compression_ratio = compression_ratio
        self.compression_stats = {
            'original_size': 0,
            'compressed_size': 0,
            'total_compressions': 0,
        }
    
    @abstractmethod
    def compress(self, weights: List[np.ndarray]) -> Tuple[List[np.ndarray], Dict]:
        pass
    
    @abstractmethod
    def decompress(self, compressed_weights: List[np.ndarray], metadata: Dict) -> List[np.ndarray]:
        pass
    
    def compute_size(self, weights: List[np.ndarray]) -> int:
        return sum([w.nbytes for w in weights])
    
    def get_compression_ratio(self) -> float:
        if self.compression_stats['original_size'] == 0:
            return 0.0
        return self.compression_stats['compressed_size'] / self.compression_stats['original_size']


class QuantizationCompressor(CompressionStrategy):
    def __init__(
        self,
        num_bits: int = 8,
        stochastic: bool = False,
    ):
        super().__init__()
        self.num_bits = num_bits
        self.stochastic = stochastic
        self.levels = 2 ** num_bits
    
    def compress(self, weights: List[np.ndarray]) -> Tuple[List[np.ndarray], Dict]:
        compressed_weights = []
        metadata = {
            'mins': [],
            'maxs': [],
            'shapes': [],
            'dtypes': [],
        }
        
        original_size = self.compute_size(weights)
        
        for w in weights:
            w_min = w.min()
            w_max = w.max()
            
            if w_max == w_min:
                quantized = np.zeros_like(w, dtype=np.uint8)
            else:
                normalized = (w - w_min) / (w_max - w_min)
                
                if self.stochastic:
                    scaled = normalized * (self.levels - 1)
                    floored = np.floor(scaled)
                    prob = scaled - floored
                    random_vals = np.random.rand(*w.shape)
                    quantized = (floored + (random_vals < prob)).astype(np.uint8)
                else:
                    quantized = np.round(normalized * (self.levels - 1)).astype(np.uint8)
            
            compressed_weights.append(quantized)
            metadata['mins'].append(float(w_min))
            metadata['maxs'].append(float(w_max))
            metadata['shapes'].append(w.shape)
            metadata['dtypes'].append(str(w.dtype))
        
        compressed_size = self.compute_size(compressed_weights)
        self.compression_stats['original_size'] += original_size
        self.compression_stats['compressed_size'] += compressed_size
        self.compression_stats['total_compressions'] += 1
        
        return compressed_weights, metadata
    
    def decompress(self, compressed_weights: List[np.ndarray], metadata: Dict) -> List[np.ndarray]:
        decompressed_weights = []
        
        for quantized, w_min, w_max, shape, dtype in zip(
            compressed_weights,
            metadata['mins'],
            metadata['maxs'],
            metadata['shapes'],
            metadata['dtypes'],
        ):
            if w_max == w_min:
                w = np.full(shape, w_min, dtype=dtype)
            else:
                normalized = quantized.astype(np.float32) / (self.levels - 1)
                w = normalized * (w_max - w_min) + w_min
                w = w.astype(dtype)
            
            decompressed_weights.append(w)
        
        return decompressed_weights


class SparsificationCompressor(CompressionStrategy):
    def __init__(
        self,
        sparsity: float = 0.9,
        method: str = 'topk',
    ):
        super().__init__(compression_ratio=1 - sparsity)
        self.sparsity = sparsity
        self.method = method
    
    def compress(self, weights: List[np.ndarray]) -> Tuple[List[np.ndarray], Dict]:
        compressed_weights = []
        metadata = {
            'indices': [],
            'shapes': [],
            'dtypes': [],
        }
        
        original_size = self.compute_size(weights)
        
        for w in weights:
            if self.method == 'topk':
                k = max(1, int(w.size * (1 - self.sparsity)))
                flat_w = w.flatten()
                threshold_indices = np.argpartition(np.abs(flat_w), -k)[-k:]
                
                sparse_values = flat_w[threshold_indices]
                compressed_weights.append(sparse_values)
                metadata['indices'].append(threshold_indices)
            
            elif self.method == 'threshold':
                flat_w = w.flatten()
                threshold = np.percentile(np.abs(flat_w), self.sparsity * 100)
                mask = np.abs(flat_w) >= threshold
                
                sparse_values = flat_w[mask]
                compressed_weights.append(sparse_values)
                metadata['indices'].append(np.where(mask)[0])
            
            elif self.method == 'random':
                k = max(1, int(w.size * (1 - self.sparsity)))
                flat_w = w.flatten()
                indices = np.random.choice(w.size, k, replace=False)
                
                sparse_values = flat_w[indices]
                compressed_weights.append(sparse_values)
                metadata['indices'].append(indices)
            
            metadata['shapes'].append(w.shape)
            metadata['dtypes'].append(str(w.dtype))
        
        compressed_size = sum([w.nbytes + idx.nbytes for w, idx in zip(compressed_weights, metadata['indices'])])
        self.compression_stats['original_size'] += original_size
        self.compression_stats['compressed_size'] += compressed_size
        self.compression_stats['total_compressions'] += 1
        
        return compressed_weights, metadata
    
    def decompress(self, compressed_weights: List[np.ndarray], metadata: Dict) -> List[np.ndarray]:
        decompressed_weights = []
        
        for sparse_values, indices, shape, dtype in zip(
            compressed_weights,
            metadata['indices'],
            metadata['shapes'],
            metadata['dtypes'],
        ):
            w = np.zeros(np.prod(shape), dtype=dtype)
            w[indices] = sparse_values
            w = w.reshape(shape)
            decompressed_weights.append(w)
        
        return decompressed_weights


class AdaptiveCompressor(CompressionStrategy):
    def __init__(
        self,
        target_compression: float = 0.5,
        quantization_bits: int = 8,
        sparsity: float = 0.0,
    ):
        super().__init__(compression_ratio=target_compression)
        self.target_compression = target_compression
        self.quantizer = QuantizationCompressor(num_bits=quantization_bits)
        self.sparsifier = SparsificationCompressor(sparsity=sparsity) if sparsity > 0 else None
    
    def compress(self, weights: List[np.ndarray]) -> Tuple[List[np.ndarray], Dict]:
        if self.sparsifier:
            sparse_weights, sparse_meta = self.sparsifier.compress(weights)
            quantized_weights, quant_meta = self.quantizer.compress(sparse_weights)
            
            metadata = {
                'sparse': sparse_meta,
                'quantized': quant_meta,
                'use_sparsifier': True,
            }
            return quantized_weights, metadata
        else:
            quantized_weights, quant_meta = self.quantizer.compress(weights)
            metadata = {
                'quantized': quant_meta,
                'use_sparsifier': False,
            }
            return quantized_weights, metadata
    
    def decompress(self, compressed_weights: List[np.ndarray], metadata: Dict) -> List[np.ndarray]:
        quantized_weights = self.quantizer.decompress(
            compressed_weights,
            metadata['quantized'],
        )
        
        if metadata.get('use_sparsifier', False):
            weights = self.sparsifier.decompress(
                quantized_weights,
                metadata['sparse'],
            )
            return weights
        else:
            return quantized_weights


class GradientCompression:
    def __init__(
        self,
        compression_strategy: CompressionStrategy,
        error_feedback: bool = True,
    ):
        self.compression_strategy = compression_strategy
        self.error_feedback = error_feedback
        self.error_memory = None
    
    def compress_gradients(
        self,
        gradients: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], Dict]:
        if self.error_feedback and self.error_memory is not None:
            gradients_with_error = [
                g + e for g, e in zip(gradients, self.error_memory)
            ]
        else:
            gradients_with_error = gradients
        
        compressed, metadata = self.compression_strategy.compress(gradients_with_error)
        
        if self.error_feedback:
            decompressed = self.compression_strategy.decompress(compressed, metadata)
            self.error_memory = [
                g - d for g, d in zip(gradients_with_error, decompressed)
            ]
        
        return compressed, metadata
    
    def decompress_gradients(
        self,
        compressed: List[np.ndarray],
        metadata: Dict,
    ) -> List[np.ndarray]:
        return self.compression_strategy.decompress(compressed, metadata)


class SketchCompression:
    def __init__(self, sketch_size: int = 1000):
        self.sketch_size = sketch_size
    
    def compress(self, weights: List[np.ndarray]) -> Tuple[List[np.ndarray], Dict]:
        compressed = []
        metadata = {'shapes': [], 'sketch_indices': []}
        
        for w in weights:
            flat_w = w.flatten()
            size = min(self.sketch_size, flat_w.size)
            
            indices = np.random.choice(flat_w.size, size, replace=False)
            sketch = flat_w[indices]
            
            compressed.append(sketch)
            metadata['shapes'].append(w.shape)
            metadata['sketch_indices'].append(indices)
        
        return compressed, metadata
    
    def decompress(self, compressed: List[np.ndarray], metadata: Dict) -> List[np.ndarray]:
        decompressed = []
        
        for sketch, shape, indices in zip(
            compressed,
            metadata['shapes'],
            metadata['sketch_indices'],
        ):
            w = np.zeros(np.prod(shape))
            w[indices] = sketch
            w = w.reshape(shape)
            decompressed.append(w)
        
        return decompressed


class CommunicationScheduler:
    def __init__(
        self,
        initial_interval: int = 1,
        max_interval: int = 10,
        adaptation_rate: float = 0.1,
    ):
        self.initial_interval = initial_interval
        self.max_interval = max_interval
        self.adaptation_rate = adaptation_rate
        self.current_interval = initial_interval
        self.round_count = 0
        self.performance_history = []
    
    def should_communicate(self) -> bool:
        self.round_count += 1
        return self.round_count % self.current_interval == 0
    
    def update_schedule(self, performance_improvement: float):
        self.performance_history.append(performance_improvement)
        
        if len(self.performance_history) < 5:
            return
        
        recent_improvement = np.mean(self.performance_history[-5:])
        
        if recent_improvement < 0.01:
            self.current_interval = min(
                self.max_interval,
                int(self.current_interval * (1 + self.adaptation_rate))
            )
        elif recent_improvement > 0.05:
            self.current_interval = max(
                self.initial_interval,
                int(self.current_interval * (1 - self.adaptation_rate))
            )
        
        logger.info(f"Communication interval updated to {self.current_interval}")
    
    def reset(self):
        self.current_interval = self.initial_interval
        self.round_count = 0
        self.performance_history = []
