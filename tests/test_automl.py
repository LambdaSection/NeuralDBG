"""
Tests for Neural AutoML module.
"""
import pytest
import numpy as np

from neural.automl import (
    AutoMLEngine,
    ArchitectureSpace,
    LayerChoice,
    ArchitectureBuilder,
    GridSearchStrategy,
    RandomSearchStrategy,
    BayesianSearchStrategy,
    EvolutionarySearchStrategy,
    MedianPruner,
    HyperbandPruner,
    ASHAPruner,
    ThresholdPruner,
    SequentialExecutor,
    MetricTracker,
    PerformancePredictor,
)
from neural.automl.nas_operations import (
    get_nas_primitives,
    create_nas_cell,
    estimate_model_size,
    compute_flops,
)
from neural.automl.utils import (
    hash_architecture,
    compare_architectures,
    validate_architecture,
    ArchitectureRegistry,
    create_architecture_summary,
)


class TestArchitectureSpace:
    """Test ArchitectureSpace class."""
    
    def test_create_architecture_space(self):
        """Test basic architecture space creation."""
        space = ArchitectureSpace()
        assert space.input_shape is None
        assert len(space.layer_choices) == 0
        assert len(space.fixed_layers) == 0
    
    def test_add_layer_choice(self):
        """Test adding layer choices."""
        space = ArchitectureSpace()
        choices = [
            {'type': 'Dense', 'params': {'units': 64}},
            {'type': 'Dense', 'params': {'units': 128}}
        ]
        space.add_layer_choice('dense_layer', choices)
        assert len(space.layer_choices) == 1
        assert space.layer_choices[0].num_choices == 2
    
    def test_sample_architecture(self):
        """Test architecture sampling."""
        space = ArchitectureSpace()
        space.input_shape = (28, 28, 1)
        choices = [
            {'type': 'Dense', 'params': {'units': 64}},
            {'type': 'Dense', 'params': {'units': 128}}
        ]
        space.add_layer_choice('dense', choices)
        
        arch = space.sample_architecture()
        assert 'input' in arch
        assert 'layers' in arch
        assert len(arch['layers']) >= 1
    
    def test_search_space_size(self):
        """Test search space size calculation."""
        space = ArchitectureSpace()
        choices = [
            {'type': 'Dense', 'params': {'units': 64}},
            {'type': 'Dense', 'params': {'units': 128}}
        ]
        space.add_layer_choice('dense', choices)
        assert space.get_search_space_size() == 2


class TestSearchStrategies:
    """Test search strategies."""
    
    def test_random_search(self):
        """Test random search strategy."""
        strategy = RandomSearchStrategy(seed=42)
        assert strategy.name == 'random_search'
        
        space = ArchitectureSpace()
        space.input_shape = (28, 28, 1)
        choices = [
            {'type': 'Dense', 'params': {'units': 64}},
            {'type': 'Dense', 'params': {'units': 128}}
        ]
        space.add_layer_choice('dense', choices)
        
        arch = strategy.suggest(space, 0)
        assert arch is not None
    
    def test_grid_search(self):
        """Test grid search strategy."""
        strategy = GridSearchStrategy()
        assert strategy.name == 'grid_search'
        
        space = ArchitectureSpace()
        space.input_shape = (28, 28, 1)
        choices = [
            {'type': 'Dense', 'params': {'units': 64}},
            {'type': 'Dense', 'params': {'units': 128}}
        ]
        space.add_layer_choice('dense', choices)
        
        arch = strategy.suggest(space, 0)
        assert arch is not None
    
    def test_evolutionary_search(self):
        """Test evolutionary search strategy."""
        strategy = EvolutionarySearchStrategy(population_size=5)
        assert strategy.name == 'evolutionary_search'
        
        space = ArchitectureSpace()
        space.input_shape = (28, 28, 1)
        choices = [
            {'type': 'Dense', 'params': {'units': 64}},
            {'type': 'Dense', 'params': {'units': 128}}
        ]
        space.add_layer_choice('dense', choices)
        
        arch = strategy.suggest(space, 0)
        assert arch is not None


class TestEarlyStopping:
    """Test early stopping strategies."""
    
    def test_median_pruner(self):
        """Test median pruner."""
        pruner = MedianPruner(n_startup_trials=2, n_warmup_steps=2)
        assert pruner.name == 'median_pruner'
        
        trial_history = []
        metrics = {'accuracy': 0.5}
        should_stop = pruner.should_stop(0, 5, metrics, trial_history)
        assert isinstance(should_stop, bool)
    
    def test_asha_pruner(self):
        """Test ASHA pruner."""
        pruner = ASHAPruner(reduction_factor=4, min_resource=1)
        assert pruner.name == 'asha_pruner'
        
        trial_history = []
        metrics = {'accuracy': 0.5}
        should_stop = pruner.should_stop(0, 1, metrics, trial_history)
        assert isinstance(should_stop, bool)
    
    def test_threshold_pruner(self):
        """Test threshold pruner."""
        pruner = ThresholdPruner(lower_threshold=0.3, metric_name='accuracy')
        assert pruner.name == 'threshold_pruner'
        
        trial_history = []
        metrics = {'accuracy': 0.2}
        should_stop = pruner.should_stop(0, 5, metrics, trial_history)
        assert should_stop is True


class TestExecutors:
    """Test execution backends."""
    
    def test_sequential_executor(self):
        """Test sequential executor."""
        executor = SequentialExecutor()
        assert executor.name == 'sequential'
        assert executor.max_workers == 1
        
        def dummy_fn(config, trial_id):
            return {'result': trial_id}
        
        configs = [{'param': i} for i in range(3)]
        results = executor.execute_trials(dummy_fn, configs)
        
        assert len(results) == 3
        assert all(r['success'] for r in results)
        
        executor.shutdown()


class TestNASOperations:
    """Test NAS operations."""
    
    def test_get_nas_primitives(self):
        """Test getting NAS primitives."""
        ops = get_nas_primitives()
        assert len(ops) > 0
        assert all(hasattr(op, 'name') for op in ops)
    
    def test_create_nas_cell(self):
        """Test creating NAS cell."""
        ops = get_nas_primitives()[:3]
        cell = create_nas_cell(ops, num_nodes=2, in_channels=32, out_channels=32)
        
        assert 'type' in cell
        assert 'operations' in cell
        assert len(cell['operations']) > 0
    
    def test_estimate_model_size(self):
        """Test model size estimation."""
        arch = {
            'layers': [
                {'type': 'Conv2D', 'params': {'filters': 64, 'kernel_size': 3}},
                {'type': 'Dense', 'params': {'units': 128}}
            ]
        }
        size = estimate_model_size(arch)
        assert size > 0
    
    def test_compute_flops(self):
        """Test FLOPS computation."""
        arch = {
            'layers': [
                {'type': 'Conv2D', 'params': {'filters': 64, 'kernel_size': 3}},
                {'type': 'Dense', 'params': {'units': 128}}
            ]
        }
        flops = compute_flops(arch, (32, 32, 3))
        assert flops > 0


class TestEvaluation:
    """Test evaluation components."""
    
    def test_metric_tracker(self):
        """Test metric tracker."""
        tracker = MetricTracker()
        
        tracker.log(0, loss=0.5, accuracy=0.7)
        tracker.log(1, loss=0.4, accuracy=0.75)
        
        assert tracker.get_metric('loss', 0) == 0.5
        assert tracker.get_metric('accuracy', 1) == 0.75
        
        best_step, best_acc = tracker.get_best('accuracy', mode='max')
        assert best_acc == 0.75
    
    def test_performance_predictor(self):
        """Test performance predictor."""
        predictor = PerformancePredictor(predictor_type='zero_cost')
        
        arch = {
            'layers': [
                {'type': 'Dense', 'params': {'units': 128}},
                {'type': 'Dense', 'params': {'units': 10}}
            ]
        }
        
        pred = predictor.predict(arch)
        assert 0 <= pred <= 1


class TestUtils:
    """Test utility functions."""
    
    def test_hash_architecture(self):
        """Test architecture hashing."""
        arch1 = {'layers': [{'type': 'Dense', 'params': {'units': 64}}]}
        arch2 = {'layers': [{'type': 'Dense', 'params': {'units': 64}}]}
        arch3 = {'layers': [{'type': 'Dense', 'params': {'units': 128}}]}
        
        hash1 = hash_architecture(arch1)
        hash2 = hash_architecture(arch2)
        hash3 = hash_architecture(arch3)
        
        assert hash1 == hash2
        assert hash1 != hash3
    
    def test_compare_architectures(self):
        """Test architecture comparison."""
        arch1 = {'layers': [{'type': 'Dense', 'params': {'units': 64}}]}
        arch2 = {'layers': [{'type': 'Dense', 'params': {'units': 64}}]}
        arch3 = {'layers': [{'type': 'Conv2D', 'params': {'filters': 32}}]}
        
        sim12 = compare_architectures(arch1, arch2)
        sim13 = compare_architectures(arch1, arch3)
        
        assert sim12 > sim13
    
    def test_validate_architecture(self):
        """Test architecture validation."""
        valid_arch = {'layers': [{'type': 'Dense', 'params': {}}]}
        invalid_arch = {'no_layers': []}
        
        is_valid, error = validate_architecture(valid_arch)
        assert is_valid
        assert error is None
        
        is_valid, error = validate_architecture(invalid_arch)
        assert not is_valid
        assert error is not None
    
    def test_architecture_registry(self):
        """Test architecture registry."""
        registry = ArchitectureRegistry()
        
        arch = {'layers': [{'type': 'Dense', 'params': {'units': 64}}]}
        metrics = {'accuracy': 0.9, 'loss': 0.3}
        
        registry.register(arch, metrics)
        assert registry.size() == 1
        
        cached = registry.get(arch)
        assert cached is not None
        assert cached['metrics']['accuracy'] == 0.9
    
    def test_create_architecture_summary(self):
        """Test architecture summary creation."""
        arch = {
            'layers': [
                {'type': 'Dense', 'params': {'units': 64}},
                {'type': 'Dense', 'params': {'units': 10}}
            ],
            'optimizer': {'type': 'Adam', 'params': {'learning_rate': 0.001}}
        }
        
        summary = create_architecture_summary(arch)
        assert isinstance(summary, str)
        assert 'Dense' in summary
        assert 'Adam' in summary


class TestAutoMLEngine:
    """Test AutoML engine (integration tests)."""
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        engine = AutoMLEngine(
            search_strategy='random',
            early_stopping='median',
            executor_type='sequential',
            backend='pytorch',
            device='cpu'
        )
        
        assert engine.search_strategy_name == 'random'
        assert engine.early_stopping_name == 'median'
        assert engine.executor_type == 'sequential'
        assert engine.backend == 'pytorch'
    
    def test_get_search_summary(self):
        """Test search summary generation."""
        engine = AutoMLEngine()
        
        engine.trial_history = [
            {
                'trial_id': 0,
                'metrics': {'val_acc': {'max': 0.8}}
            },
            {
                'trial_id': 1,
                'metrics': {'val_acc': {'max': 0.85}}
            }
        ]
        
        summary = engine.get_search_summary()
        assert 'num_trials' in summary
        assert 'best_accuracy' in summary
        assert summary['num_trials'] == 2
        assert summary['best_accuracy'] == 0.85


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
