"""
Prediction logging and analysis for production monitoring.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class PredictionRecord:
    """Single prediction record."""
    
    timestamp: float
    prediction_id: str
    input_features: Dict[str, Any]
    prediction: Any
    prediction_proba: Optional[Dict[str, float]] = None
    ground_truth: Optional[Any] = None
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'prediction_id': self.prediction_id,
            'input_features': self.input_features,
            'prediction': self.prediction,
            'prediction_proba': self.prediction_proba,
            'ground_truth': self.ground_truth,
            'latency_ms': self.latency_ms,
            'metadata': self.metadata,
        }


class PredictionLogger:
    """Logger for model predictions in production."""
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        batch_size: int = 100,
        enable_sampling: bool = False,
        sampling_rate: float = 1.0
    ):
        """
        Initialize prediction logger.
        
        Parameters
        ----------
        storage_path : str, optional
            Path to store prediction logs
        batch_size : int
            Number of predictions to buffer before writing
        enable_sampling : bool
            Whether to sample predictions
        sampling_rate : float
            Rate at which to sample predictions (0-1)
        """
        self.storage_path = Path(storage_path) if storage_path else Path("monitoring_data/predictions")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.enable_sampling = enable_sampling
        self.sampling_rate = sampling_rate
        
        self.prediction_buffer: List[PredictionRecord] = []
        self.total_predictions = 0
        
    def log_prediction(
        self,
        prediction_id: str,
        input_features: Dict[str, Any],
        prediction: Any,
        prediction_proba: Optional[Dict[str, float]] = None,
        ground_truth: Optional[Any] = None,
        latency_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log a prediction.
        
        Parameters
        ----------
        prediction_id : str
            Unique identifier for prediction
        input_features : dict
            Input features
        prediction : any
            Model prediction
        prediction_proba : dict, optional
            Prediction probabilities
        ground_truth : any, optional
            Ground truth label
        latency_ms : float
            Prediction latency in milliseconds
        metadata : dict, optional
            Additional metadata
        """
        # Sample if enabled
        if self.enable_sampling and np.random.random() > self.sampling_rate:
            return
        
        record = PredictionRecord(
            timestamp=time.time(),
            prediction_id=prediction_id,
            input_features=input_features,
            prediction=prediction,
            prediction_proba=prediction_proba,
            ground_truth=ground_truth,
            latency_ms=latency_ms,
            metadata=metadata or {}
        )
        
        self.prediction_buffer.append(record)
        self.total_predictions += 1
        
        # Flush if buffer is full
        if len(self.prediction_buffer) >= self.batch_size:
            self.flush()
    
    def flush(self):
        """Flush prediction buffer to disk."""
        if not self.prediction_buffer:
            return
        
        timestamp = int(time.time())
        log_file = self.storage_path / f"predictions_{timestamp}.jsonl"
        
        with open(log_file, 'w') as f:
            for record in self.prediction_buffer:
                f.write(json.dumps(record.to_dict()) + '\n')
        
        self.prediction_buffer.clear()
    
    def get_recent_predictions(self, n: int = 100) -> List[PredictionRecord]:
        """Get recent predictions."""
        # Load from most recent log files
        log_files = sorted(self.storage_path.glob("predictions_*.jsonl"), reverse=True)
        
        predictions = []
        for log_file in log_files:
            with open(log_file, 'r') as f:
                for line in f:
                    if len(predictions) >= n:
                        break
                    data = json.loads(line)
                    predictions.append(PredictionRecord(**data))
            
            if len(predictions) >= n:
                break
        
        return predictions[:n]


class PredictionAnalyzer:
    """Analyzer for prediction logs."""
    
    def __init__(self, logger: PredictionLogger):
        """
        Initialize prediction analyzer.
        
        Parameters
        ----------
        logger : PredictionLogger
            Prediction logger instance
        """
        self.logger = logger
    
    def analyze_predictions(
        self,
        window: int = 1000,
        time_window: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Analyze recent predictions.
        
        Parameters
        ----------
        window : int
            Number of recent predictions to analyze
        time_window : float, optional
            Time window in seconds (overrides window if provided)
            
        Returns
        -------
        dict
            Analysis results
        """
        # Load predictions
        if time_window:
            predictions = self._load_predictions_by_time(time_window)
        else:
            predictions = self.logger.get_recent_predictions(window)
        
        if not predictions:
            return {
                'status': 'no_data',
                'message': 'No predictions available'
            }
        
        analysis = {
            'status': 'ok',
            'total_predictions': len(predictions),
            'time_range': {
                'start': min(p.timestamp for p in predictions),
                'end': max(p.timestamp for p in predictions)
            }
        }
        
        # Latency statistics
        latencies = [p.latency_ms for p in predictions if p.latency_ms > 0]
        if latencies:
            analysis['latency'] = {
                'mean': np.mean(latencies),
                'median': np.median(latencies),
                'p50': np.percentile(latencies, 50),
                'p95': np.percentile(latencies, 95),
                'p99': np.percentile(latencies, 99),
                'min': np.min(latencies),
                'max': np.max(latencies),
            }
        
        # Prediction distribution
        predictions_list = [p.prediction for p in predictions]
        unique_predictions, counts = np.unique(predictions_list, return_counts=True)
        analysis['prediction_distribution'] = {
            str(pred): int(count) for pred, count in zip(unique_predictions, counts)
        }
        
        # Accuracy if ground truth available
        with_ground_truth = [p for p in predictions if p.ground_truth is not None]
        if with_ground_truth:
            correct = sum(p.prediction == p.ground_truth for p in with_ground_truth)
            analysis['accuracy'] = {
                'total_labeled': len(with_ground_truth),
                'correct': correct,
                'accuracy': correct / len(with_ground_truth)
            }
        
        # Confidence statistics (if probabilities available)
        predictions_with_proba = [p for p in predictions if p.prediction_proba is not None]
        if predictions_with_proba:
            max_probas = [max(p.prediction_proba.values()) for p in predictions_with_proba]
            analysis['confidence'] = {
                'mean': np.mean(max_probas),
                'median': np.median(max_probas),
                'min': np.min(max_probas),
                'max': np.max(max_probas),
                'low_confidence_rate': sum(p < 0.5 for p in max_probas) / len(max_probas)
            }
        
        return analysis
    
    def _load_predictions_by_time(self, time_window: float) -> List[PredictionRecord]:
        """Load predictions within time window."""
        current_time = time.time()
        start_time = current_time - time_window
        
        predictions = []
        log_files = sorted(self.logger.storage_path.glob("predictions_*.jsonl"), reverse=True)
        
        for log_file in log_files:
            # Check if file is in time range
            file_timestamp = int(log_file.stem.split('_')[1])
            if file_timestamp < start_time:
                break
            
            with open(log_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    if data['timestamp'] >= start_time:
                        predictions.append(PredictionRecord(**data))
        
        return sorted(predictions, key=lambda p: p.timestamp)
    
    def calculate_performance_metrics(
        self,
        window: int = 1000
    ) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Parameters
        ----------
        window : int
            Number of recent predictions to analyze
            
        Returns
        -------
        dict
            Performance metrics
        """
        predictions = self.logger.get_recent_predictions(window)
        
        # Filter predictions with ground truth
        labeled_predictions = [p for p in predictions if p.ground_truth is not None]
        
        if not labeled_predictions:
            return {}
        
        metrics = {}
        
        # Accuracy
        correct = sum(p.prediction == p.ground_truth for p in labeled_predictions)
        metrics['accuracy'] = correct / len(labeled_predictions)
        
        # For binary classification
        predictions_array = np.array([p.prediction for p in labeled_predictions])
        ground_truth_array = np.array([p.ground_truth for p in labeled_predictions])
        
        if set(ground_truth_array) == {0, 1} or set(ground_truth_array) == {0.0, 1.0}:
            # Calculate precision, recall, F1
            tp = np.sum((predictions_array == 1) & (ground_truth_array == 1))
            fp = np.sum((predictions_array == 1) & (ground_truth_array == 0))
            fn = np.sum((predictions_array == 0) & (ground_truth_array == 1))
            tn = np.sum((predictions_array == 0) & (ground_truth_array == 0))
            
            metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            if metrics['precision'] + metrics['recall'] > 0:
                metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])
            else:
                metrics['f1'] = 0.0
        
        return metrics
    
    def detect_anomalous_predictions(
        self,
        window: int = 1000,
        threshold: float = 3.0
    ) -> List[PredictionRecord]:
        """
        Detect anomalous predictions.
        
        Parameters
        ----------
        window : int
            Number of recent predictions to analyze
        threshold : float
            Z-score threshold for anomaly detection
            
        Returns
        -------
        list
            List of anomalous predictions
        """
        predictions = self.logger.get_recent_predictions(window)
        
        if not predictions:
            return []
        
        # Detect based on confidence
        predictions_with_proba = [p for p in predictions if p.prediction_proba is not None]
        
        if not predictions_with_proba:
            return []
        
        confidences = np.array([max(p.prediction_proba.values()) for p in predictions_with_proba])
        mean_confidence = np.mean(confidences)
        std_confidence = np.std(confidences)
        
        if std_confidence == 0:
            return []
        
        z_scores = np.abs((confidences - mean_confidence) / std_confidence)
        anomalous_indices = np.where(z_scores > threshold)[0]
        
        return [predictions_with_proba[i] for i in anomalous_indices]
