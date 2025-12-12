"""
Advanced HPO Examples for Neural DSL

This script demonstrates the enhanced HPO capabilities including:
- Bayesian optimization with different samplers
- Multi-objective optimization
- Distributed HPO with Ray Tune
- Parameter importance analysis
- Rich visualizations
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neural.hpo import (
    optimize_and_return,
    MultiObjectiveOptimizer,
    DistributedHPO,
    ParameterImportanceAnalyzer,
    BayesianParameterImportance,
    plot_optimization_history,
    plot_param_importance,
    plot_parallel_coordinates,
    plot_multi_objective_pareto,
    create_optimization_report
)


# Example 1: Basic Bayesian Optimization with TPE
def example_bayesian_optimization():
    """Demonstrate Bayesian optimization with TPE sampler."""
    print("\n" + "="*70)
    print("Example 1: Bayesian Optimization with TPE Sampler")
    print("="*70)
    
    config = """
    network SimpleMLP {
        input: (28, 28, 1)
        
        Dense(units: hpo(categorical: [64, 128, 256])) -> relu
        Dropout(rate: hpo(range: [0.3, 0.7, step=0.1]))
        Dense(units: 10) -> softmax
        
        optimizer: adam(learning_rate: hpo(log_range: [1e-4, 1e-2]))
        training: {
            batch_size: hpo(categorical: [16, 32, 64])
        }
    }
    """
    
    print("\nRunning Bayesian optimization with TPE sampler...")
    print("This uses Tree-structured Parzen Estimator for efficient search")
    
    results = optimize_and_return(
        config=config,
        n_trials=10,  # Use 10 trials for demo (increase for real use)
        dataset_name='MNIST',
        backend='pytorch',
        device='cpu',  # Use CPU for demo
        sampler='tpe',  # Bayesian optimization
        enable_pruning=True
    )
    
    print("\nOptimization Results:")
    print(f"Best batch_size: {results.get('batch_size', 'N/A')}")
    print(f"Best learning_rate: {results.get('learning_rate', 'N/A')}")
    print(f"Best dense_units: {results.get('dense_units', 'N/A')}")
    print(f"Best dropout_rate: {results.get('dropout_rate', 'N/A')}")
    
    return results


# Example 2: Multi-Objective Optimization
def example_multi_objective():
    """Demonstrate multi-objective optimization."""
    print("\n" + "="*70)
    print("Example 2: Multi-Objective Optimization")
    print("="*70)
    
    config = """
    network MultiObjectiveMLP {
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
    
    print("\nRunning multi-objective optimization...")
    print("Optimizing: loss (minimize), accuracy (maximize)")
    
    results = optimize_and_return(
        config=config,
        n_trials=10,
        dataset_name='MNIST',
        backend='pytorch',
        device='cpu',
        objectives=['loss', 'accuracy'],
        sampler='nsgaii'  # NSGA-II for multi-objective
    )
    
    print("\nMulti-objective results obtained!")
    if 'pareto_front' in results:
        print(f"Number of Pareto-optimal solutions: {len(results['pareto_front'])}")
    
    return results


# Example 3: Parameter Importance Analysis
def example_parameter_importance(trials):
    """Demonstrate parameter importance analysis."""
    print("\n" + "="*70)
    print("Example 3: Parameter Importance Analysis")
    print("="*70)
    
    if not trials:
        print("No trials available for analysis")
        return
    
    print("\nAnalyzing parameter importance using Random Forest...")
    
    # Standard Random Forest importance
    analyzer = ParameterImportanceAnalyzer(method='random_forest')
    importance = analyzer.analyze(trials, target_metric='accuracy')
    
    print("\nParameter Importances:")
    for param, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {param}: {score:.4f}")
    
    # Create visualizations
    print("\nGenerating importance plots...")
    
    try:
        # Importance plot
        fig1 = analyzer.plot_importance(importance)
        fig1.savefig('importance_plot.png')
        print("  - Saved importance_plot.png")
        
        # Importance with uncertainty (bootstrap)
        if len(trials) >= 10:
            fig2 = analyzer.plot_importance_with_std(
                trials, 
                'accuracy', 
                n_iterations=10
            )
            fig2.savefig('importance_uncertainty.png')
            print("  - Saved importance_uncertainty.png")
        
        # Interaction heatmap
        if len(trials) >= 10:
            fig3 = analyzer.plot_interaction_heatmap(trials, 'accuracy')
            fig3.savefig('interaction_heatmap.png')
            print("  - Saved interaction_heatmap.png")
        
        # Marginal effects
        if len(trials) >= 5:
            fig4 = analyzer.plot_marginal_effects(trials, 'accuracy')
            fig4.savefig('marginal_effects.png')
            print("  - Saved marginal_effects.png")
    
    except Exception as e:
        print(f"  Warning: Could not create all plots: {e}")
    
    return importance


# Example 4: Comprehensive Visualization Suite
def example_visualizations(trials):
    """Demonstrate the visualization suite."""
    print("\n" + "="*70)
    print("Example 4: Comprehensive Visualization Suite")
    print("="*70)
    
    if not trials:
        print("No trials available for visualization")
        return
    
    print("\nGenerating comprehensive visualizations...")
    
    try:
        # Optimization history
        fig1 = plot_optimization_history(trials, metric='accuracy')
        fig1.savefig('optimization_history.png')
        print("  - Saved optimization_history.png")
        
        # Parallel coordinates plot
        if len(trials) >= 5:
            fig2 = plot_parallel_coordinates(trials, metric='accuracy', top_n=5)
            fig2.savefig('parallel_coordinates.png')
            print("  - Saved parallel_coordinates.png")
        
        # Generate HTML report
        report_path = create_optimization_report(
            trials,
            metric='accuracy',
            output_path='hpo_report.html'
        )
        if report_path:
            print(f"  - Saved comprehensive HTML report: {report_path}")
    
    except Exception as e:
        print(f"  Warning: Could not create all visualizations: {e}")


# Example 5: Comparing Different Samplers
def example_sampler_comparison():
    """Compare different optimization samplers."""
    print("\n" + "="*70)
    print("Example 5: Comparing Different Samplers")
    print("="*70)
    
    config = """
    network ComparisonMLP {
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
    
    samplers = ['tpe', 'random', 'cmaes']
    results_dict = {}
    
    for sampler in samplers:
        print(f"\nRunning optimization with {sampler.upper()} sampler...")
        
        try:
            results = optimize_and_return(
                config=config,
                n_trials=5,  # Small number for demo
                dataset_name='MNIST',
                backend='pytorch',
                device='cpu',
                sampler=sampler
            )
            results_dict[sampler] = results
            print(f"  {sampler.upper()} completed!")
        except Exception as e:
            print(f"  {sampler.upper()} failed: {e}")
    
    return results_dict


# Example 6: Bayesian Parameter Importance with Gaussian Process
def example_bayesian_importance(trials):
    """Demonstrate Bayesian parameter importance analysis."""
    print("\n" + "="*70)
    print("Example 6: Bayesian Parameter Importance (Gaussian Process)")
    print("="*70)
    
    if not trials or len(trials) < 5:
        print("Need at least 5 trials for GP analysis")
        return
    
    print("\nAnalyzing parameter importance using Gaussian Process...")
    
    try:
        bayesian_analyzer = BayesianParameterImportance()
        gp_importance = bayesian_analyzer.analyze_with_gp(trials, 'accuracy')
        
        if gp_importance:
            print("\nGP-based Parameter Importances:")
            for param, score in sorted(gp_importance.items(), key=lambda x: x[1], reverse=True):
                print(f"  {param}: {score:.4f}")
            
            # Create visualization
            fig = bayesian_analyzer.plot_importance_with_uncertainty(trials, 'accuracy')
            if fig:
                fig.savefig('gp_importance.png')
                print("\n  - Saved gp_importance.png")
        else:
            print("Could not compute GP importance")
    
    except Exception as e:
        print(f"GP analysis failed: {e}")


# Main execution
def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("Neural DSL - Advanced HPO Examples")
    print("="*70)
    print("\nThis script demonstrates enhanced hyperparameter optimization features.")
    print("Note: Examples use small trial counts for demonstration purposes.")
    print("For production use, increase n_trials (e.g., 50-100).")
    
    # Run examples
    try:
        # Example 1: Basic Bayesian optimization
        results1 = example_bayesian_optimization()
        trials1 = results1.get('_trials_history', [])
        
        # Example 3: Parameter importance (requires trials from example 1)
        if trials1:
            example_parameter_importance(trials1)
        
        # Example 4: Visualizations (requires trials from example 1)
        if trials1:
            example_visualizations(trials1)
        
        # Example 6: Bayesian importance (requires trials from example 1)
        if trials1:
            example_bayesian_importance(trials1)
        
        # Example 5: Sampler comparison (optional, can be time-consuming)
        # Uncomment to run:
        # example_sampler_comparison()
        
        # Example 2: Multi-objective (optional, can be time-consuming)
        # Uncomment to run:
        # example_multi_objective()
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("Examples completed!")
    print("="*70)
    print("\nGenerated files:")
    print("  - importance_plot.png")
    print("  - importance_uncertainty.png (if enough trials)")
    print("  - interaction_heatmap.png (if enough trials)")
    print("  - marginal_effects.png (if enough trials)")
    print("  - optimization_history.png")
    print("  - parallel_coordinates.png (if enough trials)")
    print("  - hpo_report.html")
    print("  - gp_importance.png (if enough trials)")
    print("\nFor more information, see: neural/hpo/README.md")


if __name__ == '__main__':
    main()
