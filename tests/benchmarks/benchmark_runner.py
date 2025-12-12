"""
Benchmark runner that executes and measures performance metrics.
"""
from __future__ import annotations

import time
import psutil
import os
import tempfile
import json
from typing import Dict, Any, Optional, List
from pathlib import Path


class BenchmarkRunner:
    """Runs benchmarks and collects performance metrics."""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or tempfile.mkdtemp(prefix='neural_benchmark_')
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.results = {}
        
    def benchmark_neural_dsl(
        self,
        model_name: str,
        dsl_code: str,
        backend: str,
        dataset: Any,
        epochs: int = 5
    ) -> Dict[str, Any]:
        """Benchmark Neural DSL model compilation and training."""
        from neural.parser.parser import create_parser, ModelTransformer
        from neural.code_generation.code_generator import generate_code
        
        metrics = {
            'model_name': model_name,
            'backend': backend,
            'framework': 'neural_dsl'
        }
        
        start_parse = time.time()
        parser = create_parser('network')
        tree = parser.parse(dsl_code)
        model_data = ModelTransformer().transform(tree)
        parse_time = time.time() - start_parse
        metrics['parse_time'] = parse_time
        
        start_codegen = time.time()
        code = generate_code(model_data, backend)
        codegen_time = time.time() - start_codegen
        metrics['codegen_time'] = codegen_time
        
        code_file = Path(self.output_dir) / f'{model_name}_{backend}_neural.py'
        code_file.write_text(code)
        
        train_metrics = self._execute_training(
            str(code_file),
            backend,
            dataset,
            epochs
        )
        
        metrics.update(train_metrics)
        metrics['total_overhead'] = parse_time + codegen_time
        
        return metrics
    
    def benchmark_native(
        self,
        model_name: str,
        native_code: str,
        backend: str,
        dataset: Any,
        epochs: int = 5
    ) -> Dict[str, Any]:
        """Benchmark native TensorFlow/PyTorch model training."""
        metrics = {
            'model_name': model_name,
            'backend': backend,
            'framework': 'native'
        }
        
        code_file = Path(self.output_dir) / f'{model_name}_{backend}_native.py'
        code_file.write_text(native_code)
        
        train_metrics = self._execute_training(
            str(code_file),
            backend,
            dataset,
            epochs,
            is_native=True
        )
        
        metrics.update(train_metrics)
        metrics['total_overhead'] = 0.0
        
        return metrics
    
    def _execute_training(
        self,
        code_file: str,
        backend: str,
        dataset: Any,
        epochs: int,
        is_native: bool = False
    ) -> Dict[str, Any]:
        """Execute training and collect metrics."""
        metrics = {}
        
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024
        
        start_train = time.time()
        
        if backend == 'tensorflow':
            train_results = self._train_tensorflow(code_file, dataset, epochs, is_native)
        elif backend == 'pytorch':
            train_results = self._train_pytorch(code_file, dataset, epochs, is_native)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        
        train_time = time.time() - start_train
        mem_after = process.memory_info().rss / 1024 / 1024
        
        metrics['training_time'] = train_time
        metrics['memory_used_mb'] = mem_after - mem_before
        metrics['peak_memory_mb'] = mem_after
        metrics.update(train_results)
        
        return metrics
    
    def _train_tensorflow(
        self,
        code_file: str,
        dataset: Any,
        epochs: int,
        is_native: bool
    ) -> Dict[str, Any]:
        """Train TensorFlow model and collect metrics."""
        import tensorflow as tf
        import numpy as np
        
        namespace = {}
        with open(code_file, 'r') as f:
            code = f.read()
        
        exec(code, namespace)
        
        if is_native:
            model = namespace['create_model']()
        else:
            model = namespace.get('model')
        
        if model is None:
            raise ValueError("Model not found in generated code")
        
        (x_train, y_train), (x_test, y_test) = dataset
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        if len(x_train.shape) == 3:
            x_train = np.expand_dims(x_train, -1)
            x_test = np.expand_dims(x_test, -1)
        
        start = time.time()
        history = model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=128,
            validation_split=0.2,
            verbose=0
        )
        training_time = time.time() - start
        
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        
        return {
            'final_loss': float(loss),
            'final_accuracy': float(accuracy),
            'training_time_pure': training_time,
            'epochs_completed': epochs
        }
    
    def _train_pytorch(
        self,
        code_file: str,
        dataset: Any,
        epochs: int,
        is_native: bool
    ) -> Dict[str, Any]:
        """Train PyTorch model and collect metrics."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        import numpy as np
        
        namespace = {}
        with open(code_file, 'r') as f:
            code = f.read()
        
        exec(code, namespace)
        
        if is_native:
            model = namespace['create_model']()
        else:
            model = namespace.get('model')
        
        if model is None:
            raise ValueError("Model not found in generated code")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        (x_train, y_train), (x_test, y_test) = dataset
        x_train = torch.FloatTensor(x_train).to(device) / 255.0
        y_train = torch.LongTensor(y_train).to(device)
        x_test = torch.FloatTensor(x_test).to(device) / 255.0
        y_test = torch.LongTensor(y_test).to(device)
        
        if len(x_train.shape) == 3:
            x_train = x_train.unsqueeze(-1)
            x_test = x_test.unsqueeze(-1)
        
        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        start = time.time()
        model.train()
        for epoch in range(epochs):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        training_time = time.time() - start
        
        model.eval()
        with torch.no_grad():
            outputs = model(x_test)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y_test).sum().item() / len(y_test)
            loss = criterion(outputs, y_test).item()
        
        return {
            'final_loss': float(loss),
            'final_accuracy': float(accuracy),
            'training_time_pure': training_time,
            'epochs_completed': epochs
        }
    
    def save_results(self, results: Dict[str, Any], filename: str = 'benchmark_results.json'):
        """Save benchmark results to JSON file."""
        output_path = Path(self.output_dir) / filename
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        return str(output_path)
    
    def compare_results(
        self,
        neural_results: Dict[str, Any],
        native_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare Neural DSL vs native implementation results."""
        comparison = {
            'model_name': neural_results['model_name'],
            'backend': neural_results['backend']
        }
        
        comparison['overhead'] = {
            'parse_time': neural_results.get('parse_time', 0),
            'codegen_time': neural_results.get('codegen_time', 0),
            'total_overhead': neural_results.get('total_overhead', 0)
        }
        
        comparison['training_time'] = {
            'neural_dsl': neural_results['training_time'],
            'native': native_results['training_time'],
            'difference': neural_results['training_time'] - native_results['training_time'],
            'percentage': ((neural_results['training_time'] - native_results['training_time']) / 
                          native_results['training_time'] * 100)
        }
        
        comparison['memory'] = {
            'neural_dsl_mb': neural_results['memory_used_mb'],
            'native_mb': native_results['memory_used_mb'],
            'difference_mb': neural_results['memory_used_mb'] - native_results['memory_used_mb'],
            'percentage': ((neural_results['memory_used_mb'] - native_results['memory_used_mb']) / 
                          native_results['memory_used_mb'] * 100) if native_results['memory_used_mb'] > 0 else 0
        }
        
        comparison['accuracy'] = {
            'neural_dsl': neural_results['final_accuracy'],
            'native': native_results['final_accuracy'],
            'difference': neural_results['final_accuracy'] - native_results['final_accuracy']
        }
        
        comparison['loss'] = {
            'neural_dsl': neural_results['final_loss'],
            'native': native_results['final_loss'],
            'difference': neural_results['final_loss'] - native_results['final_loss']
        }
        
        return comparison
