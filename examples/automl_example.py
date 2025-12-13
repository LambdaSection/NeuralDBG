"""
Comprehensive examples for Neural AutoML.

This script demonstrates:
- Basic AutoML search
- Neural Architecture Search (NAS)
- Distributed execution with Ray
- HPO integration
- Custom search strategies and operations
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor

from neural.automl import (
    AutoMLEngine,
    ArchitectureSpace,
    LayerChoice,
    GridSearchStrategy,
    RandomSearchStrategy,
    BayesianSearchStrategy,
    EvolutionarySearchStrategy,
    MedianPruner,
    HyperbandPruner,
    ASHAPruner
)
from neural.automl.nas_operations import get_nas_primitives, create_nas_cell
from neural.automl.utils import (
    create_architecture_summary,
    export_architecture_to_dsl,
    ArchitectureRegistry
)


def get_data_loaders(dataset_name='MNIST', batch_size=32):
    """Get train and validation data loaders."""
    if dataset_name == 'MNIST':
        train_dataset = MNIST(root='./data', train=True, transform=ToTensor(), download=True)
        val_dataset = MNIST(root='./data', train=False, transform=ToTensor(), download=True)
    elif dataset_name == 'CIFAR10':
        train_dataset = CIFAR10(root='./data', train=True, transform=ToTensor(), download=True)
        val_dataset = CIFAR10(root='./data', train=False, transform=ToTensor(), download=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, val_loader


def example_basic_automl():
    """Example 1: Basic AutoML search with DSL."""
    print("\n" + "="*70)
    print("Example 1: Basic AutoML Search")
    print("="*70)
    
    dsl_config = """
    network AutoMLNet {
        input: (28, 28, 1)
        
        Dense(units: hpo(categorical: [64, 128, 256])) -> relu
        Dropout(rate: hpo(range: [0.3, 0.5, step=0.1]))
        Dense(units: 10) -> softmax
        
        optimizer: adam(learning_rate: hpo(log_range: [1e-4, 1e-2]))
        training: {
            batch_size: hpo(categorical: [16, 32, 64])
        }
    }
    """
    
    print("\nCreating architecture space from DSL...")
    space = ArchitectureSpace.from_dsl(dsl_config)
    print(f"Search space size: {space.get_search_space_size()}")
    
    print("\nInitializing AutoML engine with Random Search...")
    engine = AutoMLEngine(
        search_strategy='random',
        early_stopping='median',
        executor_type='sequential',
        backend='pytorch',
        device='cpu'
    )
    
    print("\nGetting data loaders...")
    train_loader, val_loader = get_data_loaders('MNIST', batch_size=32)
    
    print("\nRunning search (5 trials for demo)...")
    results = engine.search(
        architecture_space=space,
        train_data=train_loader,
        val_data=val_loader,
        max_trials=5,
        max_epochs_per_trial=3
    )
    
    print("\nSearch completed!")
    print(f"Total trials: {results['total_trials']}")
    print(f"Total time: {results['total_time']:.2f}s")
    
    if results['best_metrics']:
        best_acc = results['best_metrics'].get('val_acc', {})
        if isinstance(best_acc, dict):
            print(f"Best accuracy: {best_acc.get('max', 0):.4f}")
        else:
            print(f"Best accuracy: {best_acc:.4f}")
    
    print("\nBest architecture summary:")
    print(create_architecture_summary(results['best_architecture']))
    
    return results


def example_nas_search():
    """Example 2: Neural Architecture Search with custom operations."""
    print("\n" + "="*70)
    print("Example 2: Neural Architecture Search")
    print("="*70)
    
    print("\nCreating NAS search space...")
    space = ArchitectureSpace()
    space.input_shape = (28, 28, 1)
    
    space.add_layer_choice('conv_block', [
        {'type': 'Conv2D', 'params': {'filters': 32, 'kernel_size': 3}},
        {'type': 'Conv2D', 'params': {'filters': 64, 'kernel_size': 3}},
        {'type': 'SeparableConv2D', 'params': {'filters': 32, 'kernel_size': 3}}
    ])
    
    space.add_layer_choice('pooling', [
        {'type': 'MaxPooling2D', 'params': {'pool_size': 2}},
        {'type': 'AveragePooling2D', 'params': {'pool_size': 2}}
    ])
    
    space.add_fixed_layer({'type': 'Flatten', 'params': {}})
    space.add_fixed_layer({'type': 'Dense', 'params': {'units': 128}})
    space.add_fixed_layer({'type': 'Dense', 'params': {'units': 10}})
    
    space.add_hyperparameter('optimizer', {
        'type': 'Adam',
        'params': {'learning_rate': 0.001}
    })
    
    print(f"Search space size: {space.get_search_space_size()}")
    
    print("\nInitializing engine with Evolutionary Search...")
    engine = AutoMLEngine(
        search_strategy='evolutionary',
        early_stopping='asha',
        executor_type='sequential',
        backend='pytorch',
        device='cpu'
    )
    
    train_loader, val_loader = get_data_loaders('MNIST', batch_size=64)
    
    print("\nRunning NAS (5 trials for demo)...")
    results = engine.search(
        architecture_space=space,
        train_data=train_loader,
        val_data=val_loader,
        max_trials=5,
        max_epochs_per_trial=3,
        population_size=3,
        mutation_rate=0.3
    )
    
    print("\nNAS completed!")
    print(f"Best architecture found:")
    print(create_architecture_summary(results['best_architecture']))
    
    print("\nExporting to DSL...")
    dsl_export = export_architecture_to_dsl(results['best_architecture'])
    print(dsl_export)
    
    return results


def example_bayesian_search():
    """Example 3: Bayesian Optimization."""
    print("\n" + "="*70)
    print("Example 3: Bayesian Optimization")
    print("="*70)
    
    dsl_config = """
    network BayesianNet {
        input: (28, 28, 1)
        
        Dense(units: hpo(categorical: [64, 128])) -> relu
        Dropout(rate: hpo(range: [0.3, 0.5, step=0.1]))
        Dense(units: 10) -> softmax
        
        optimizer: adam(learning_rate: hpo(log_range: [1e-3, 1e-2]))
        training: {
            batch_size: 32
        }
    }
    """
    
    space = ArchitectureSpace.from_dsl(dsl_config)
    
    print("\nInitializing with Bayesian Optimization (TPE)...")
    engine = AutoMLEngine(
        search_strategy='bayesian',
        early_stopping='median',
        executor_type='sequential',
        backend='pytorch',
        device='cpu'
    )
    
    train_loader, val_loader = get_data_loaders('MNIST', batch_size=32)
    
    print("\nRunning Bayesian search (8 trials)...")
    results = engine.search(
        architecture_space=space,
        train_data=train_loader,
        val_data=val_loader,
        max_trials=8,
        max_epochs_per_trial=3,
        acquisition_function='ei',
        n_initial_random=3
    )
    
    print("\nBayesian search completed!")
    summary = engine.get_search_summary()
    print(f"Best accuracy: {summary['best_accuracy']:.4f}")
    print(f"Mean accuracy: {summary['mean_accuracy']:.4f}")
    print(f"Improvement: {summary['improvement']:.4f}")
    
    return results


def example_hpo_integration():
    """Example 4: Integration with HPO module."""
    print("\n" + "="*70)
    print("Example 4: HPO Module Integration")
    print("="*70)
    
    dsl_config = """
    network HPOIntegratedNet {
        input: (28, 28, 1)
        
        Dense(units: hpo(categorical: [64, 128])) -> relu
        Dropout(rate: hpo(range: [0.3, 0.5, step=0.1]))
        Dense(units: 10) -> softmax
        
        optimizer: adam(learning_rate: hpo(log_range: [1e-3, 1e-2]))
        training: {
            batch_size: hpo(categorical: [32, 64])
        }
    }
    """
    
    print("\nInitializing AutoML engine...")
    engine = AutoMLEngine(
        backend='pytorch',
        device='cpu'
    )
    
    train_loader, val_loader = get_data_loaders('MNIST', batch_size=32)
    
    print("\nRunning HPO-integrated search...")
    try:
        results = engine.search_with_hpo(
            dsl_config=dsl_config,
            train_data=train_loader,
            val_data=val_loader,
            n_trials=5,
            dataset_name='MNIST'
        )
        
        print("\nHPO search completed!")
        print(f"Best parameters: {results['best_parameters']}")
    
    except Exception as e:
        print(f"HPO integration example skipped: {e}")
    
    return None


def example_architecture_registry():
    """Example 5: Using Architecture Registry."""
    print("\n" + "="*70)
    print("Example 5: Architecture Registry")
    print("="*70)
    
    print("\nCreating architecture registry...")
    registry = ArchitectureRegistry()
    
    arch1 = {
        'input': {'shape': (28, 28, 1)},
        'layers': [
            {'type': 'Dense', 'params': {'units': 128}},
            {'type': 'Dense', 'params': {'units': 10}}
        ]
    }
    metrics1 = {'accuracy': 0.92, 'loss': 0.25}
    
    arch2 = {
        'input': {'shape': (28, 28, 1)},
        'layers': [
            {'type': 'Dense', 'params': {'units': 256}},
            {'type': 'Dropout', 'params': {'rate': 0.3}},
            {'type': 'Dense', 'params': {'units': 10}}
        ]
    }
    metrics2 = {'accuracy': 0.95, 'loss': 0.18}
    
    print("\nRegistering architectures...")
    registry.register(arch1, metrics1)
    registry.register(arch2, metrics2)
    
    print(f"Registry size: {registry.size()}")
    
    print("\nGetting cached results...")
    cached = registry.get(arch1)
    if cached:
        print(f"Cached metrics: {cached['metrics']}")
    
    print("\nTop architectures:")
    top_archs = registry.get_top_k(k=2, metric='accuracy')
    for i, entry in enumerate(top_archs):
        print(f"{i+1}. Accuracy: {entry['metrics']['accuracy']:.4f}")


def example_custom_nas_operations():
    """Example 6: Custom NAS operations."""
    print("\n" + "="*70)
    print("Example 6: Custom NAS Operations")
    print("="*70)
    
    print("\nGetting standard NAS primitives...")
    nas_ops = get_nas_primitives()
    print(f"Available operations: {[op.name for op in nas_ops]}")
    
    print("\nCreating NAS cell...")
    cell = create_nas_cell(
        operations=nas_ops[:5],
        num_nodes=3,
        in_channels=32,
        out_channels=32
    )
    
    print(f"Cell configuration: {len(cell['operations'])} operations")
    
    from neural.automl.nas_operations import estimate_model_size, compute_flops
    
    test_arch = {
        'input': {'shape': (32, 32, 3)},
        'layers': [
            {'type': 'Conv2D', 'params': {'filters': 64, 'kernel_size': 3}},
            {'type': 'BatchNormalization', 'params': {}},
            {'type': 'MaxPooling2D', 'params': {'pool_size': 2}},
            {'type': 'Flatten', 'params': {}},
            {'type': 'Dense', 'params': {'units': 128}},
            {'type': 'Dense', 'params': {'units': 10}}
        ]
    }
    
    print("\nEstimating model complexity...")
    num_params = estimate_model_size(test_arch)
    flops = compute_flops(test_arch, (32, 32, 3))
    
    print(f"Estimated parameters: {num_params:,}")
    print(f"Estimated FLOPs: {flops:,}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("Neural AutoML - Comprehensive Examples")
    print("="*70)
    print("\nThese examples demonstrate the AutoML capabilities of Neural DSL.")
    print("Note: Using small trial counts for demonstration purposes.")
    
    try:
        example_basic_automl()
        
        example_nas_search()
        
        example_bayesian_search()
        
        example_hpo_integration()
        
        example_architecture_registry()
        
        example_custom_nas_operations()
    
    except Exception as e:
        print(f"\nExample failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("Examples completed!")
    print("="*70)
    print("\nFor more information, see: neural/automl/README.md")


if __name__ == '__main__':
    main()
