from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import click

logger = logging.getLogger(__name__)


@click.group()
def federated():
    """Federated learning commands."""
    pass


@federated.command()
@click.option('--config', type=click.Path(exists=True), required=True, help='Federated training config file')
@click.option('--data', type=click.Path(exists=True), required=True, help='Training data path')
@click.option('--test-data', type=click.Path(exists=True), help='Test data path')
@click.option('--backend', type=click.Choice(['tensorflow', 'pytorch']), default='tensorflow', help='Backend framework')
@click.option('--output', type=click.Path(), default='federated_model', help='Output model path')
def train(config: str, data: str, test_data: Optional[str], backend: str, output: str):
    """Train a model using federated learning."""
    try:
        from neural.federated.integration import federated_train_from_config
        import numpy as np
        
        logger.info(f"Loading data from {data}")
        data_dict = np.load(data)
        X_train = data_dict['X_train']
        y_train = data_dict['y_train']
        
        test_data_tuple = None
        if test_data:
            test_dict = np.load(test_data)
            X_test = test_dict['X_test']
            y_test = test_dict['y_test']
            test_data_tuple = (X_test, y_test)
        
        logger.info(f"Starting federated training with config: {config}")
        results = federated_train_from_config(
            config_path=config,
            data=(X_train, y_train),
            test_data=test_data_tuple,
        )
        
        logger.info("Training completed successfully")
        logger.info(f"Final accuracy: {results['summary'].get('final_eval_accuracy', 'N/A'):.4f}")
        
        if backend == 'tensorflow':
            results['global_model'].save(output)
        elif backend == 'pytorch':
            import torch
            torch.save(results['global_model'].state_dict(), f"{output}.pt")
        
        logger.info(f"Model saved to {output}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise click.ClickException(str(e))


@federated.command()
@click.option('--num-devices', type=int, default=100, help='Number of devices')
@click.option('--devices-per-round', type=int, default=10, help='Devices per round')
@click.option('--data', type=click.Path(exists=True), required=True, help='Training data path')
@click.option('--model', type=click.Path(exists=True), required=True, help='Model definition file')
@click.option('--num-rounds', type=int, default=100, help='Number of training rounds')
@click.option('--backend', type=click.Choice(['tensorflow', 'pytorch']), default='tensorflow')
@click.option('--output', type=click.Path(), default='federated_model')
def quick_start(
    num_devices: int,
    devices_per_round: int,
    data: str,
    model: str,
    num_rounds: int,
    backend: str,
    output: str,
):
    """Quick start federated training with minimal configuration."""
    try:
        from neural.federated import FederatedOrchestrator, CrossDeviceScenario
        import numpy as np
        
        logger.info("Loading data...")
        data_dict = np.load(data)
        X_train = data_dict['X_train']
        y_train = data_dict['y_train']
        
        logger.info("Loading model...")
        if backend == 'tensorflow':
            import tensorflow as tf
            model_instance = tf.keras.models.load_model(model)
            
            def model_fn():
                return tf.keras.models.clone_model(model_instance)
        elif backend == 'pytorch':
            import torch
            model_instance = torch.load(model)
            
            import copy
            def model_fn():
                return copy.deepcopy(model_instance)
        
        logger.info("Setting up federated orchestrator...")
        orchestrator = FederatedOrchestrator(
            model=model_instance,
            backend=backend,
            scenario=CrossDeviceScenario(
                num_devices=num_devices,
                devices_per_round=devices_per_round,
            ),
            aggregation_strategy='fedavg',
        )
        
        orchestrator.setup_clients(
            model_fn=model_fn,
            data=(X_train, y_train),
            local_epochs=1,
        )
        
        logger.info(f"Starting training for {num_rounds} rounds...")
        history = orchestrator.train(num_rounds=num_rounds)
        
        logger.info("Training completed")
        orchestrator.server.save_model(output)
        
    except Exception as e:
        logger.error(f"Quick start failed: {e}")
        raise click.ClickException(str(e))


@federated.command()
@click.option('--data', type=click.Path(exists=True), required=True, help='Data path')
@click.option('--num-clients', type=int, default=10, help='Number of clients')
@click.option('--distribution', type=click.Choice(['iid', 'non_iid']), default='iid')
@click.option('--alpha', type=float, default=0.5, help='Non-IID alpha parameter')
@click.option('--output', type=click.Path(), default='client_data', help='Output directory')
def prepare_data(data: str, num_clients: int, distribution: str, alpha: float, output: str):
    """Prepare and split data for federated learning."""
    try:
        from neural.federated.utils import split_data_iid, split_data_non_iid, compute_data_statistics
        import numpy as np
        from pathlib import Path
        
        logger.info(f"Loading data from {data}")
        data_dict = np.load(data)
        X = data_dict['X_train']
        y = data_dict['y_train']
        
        logger.info(f"Splitting data for {num_clients} clients ({distribution})...")
        if distribution == 'iid':
            client_data = split_data_iid((X, y), num_clients)
        else:
            client_data = split_data_non_iid((X, y), num_clients, alpha=alpha)
        
        stats = compute_data_statistics(client_data)
        logger.info(f"\nData Statistics:")
        logger.info(f"  Total samples: {stats['total_samples']}")
        logger.info(f"  Avg samples per client: {stats['avg_samples_per_client']:.1f}")
        logger.info(f"  Std samples per client: {stats['std_samples_per_client']:.1f}")
        
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, (X_client, y_client) in enumerate(client_data):
            client_file = output_dir / f'client_{i}.npz'
            np.savez(client_file, X=X_client, y=y_client)
            logger.info(f"Saved client {i} data to {client_file}")
        
        logger.info(f"\nData preparation completed. Clients: {len(client_data)}")
        
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        raise click.ClickException(str(e))


@federated.command()
@click.option('--metrics', type=click.Path(exists=True), required=True, help='Metrics JSON file')
@click.option('--output', type=click.Path(), default='federated_report.txt')
def report(metrics: str, output: str):
    """Generate a federated training report."""
    try:
        import json
        
        with open(metrics, 'r') as f:
            data = json.load(f)
        
        history = data.get('history', {})
        summary = data.get('summary', {})
        config = data.get('config', {})
        
        report_lines = [
            "=" * 60,
            "Federated Learning Training Report",
            "=" * 60,
            "",
            "Configuration:",
            f"  Scenario: {config.get('scenario', 'N/A')}",
            f"  Aggregation: {config.get('aggregation', 'N/A')}",
            f"  Privacy: {config.get('privacy', 'N/A')}",
            f"  Compression: {config.get('compression', 'N/A')}",
            "",
            "Summary:",
            f"  Total rounds: {summary.get('total_rounds', 'N/A')}",
            f"  Final train loss: {summary.get('final_train_loss', 'N/A'):.4f}",
            f"  Final train accuracy: {summary.get('final_train_accuracy', 'N/A'):.4f}",
        ]
        
        if 'final_eval_accuracy' in summary:
            report_lines.extend([
                f"  Final eval loss: {summary.get('final_eval_loss', 'N/A'):.4f}",
                f"  Final eval accuracy: {summary.get('final_eval_accuracy', 'N/A'):.4f}",
                f"  Best eval accuracy: {summary.get('best_eval_accuracy', 'N/A'):.4f}",
            ])
        
        if 'avg_computation_time' in summary:
            report_lines.append(f"  Avg computation time: {summary['avg_computation_time']:.2f}s")
        
        if 'avg_communication_cost' in summary:
            report_lines.append(f"  Avg communication cost: {summary['avg_communication_cost']:.4f}")
        
        if 'privacy_budget_spent' in summary:
            report_lines.append(f"  Privacy budget spent: {summary['privacy_budget_spent']:.4f}")
        
        report_lines.extend([
            "",
            "=" * 60,
        ])
        
        report_text = "\n".join(report_lines)
        
        with open(output, 'w') as f:
            f.write(report_text)
        
        logger.info(report_text)
        logger.info(f"\nReport saved to {output}")
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise click.ClickException(str(e))


@federated.command()
@click.option('--scenario', type=click.Choice(['cross_device', 'cross_silo', 'hybrid']), default='cross_device')
@click.option('--num-clients', type=int, default=10)
@click.option('--output', type=click.Path(), default='federated_config.yaml')
def generate_config(scenario: str, num_clients: int, output: str):
    """Generate a federated learning configuration template."""
    try:
        import yaml
        
        if scenario == 'cross_device':
            config = {
                'backend': 'tensorflow',
                'model': {
                    'model_data': {
                        'input': {'shape': [784]},
                        'layers': [
                            {'type': 'dense', 'units': 128, 'activation': 'relu'},
                            {'type': 'dropout', 'rate': 0.2},
                            {'type': 'output', 'units': 10, 'activation': 'softmax'},
                        ],
                    },
                },
                'training': {
                    'batch_size': 32,
                    'learning_rate': 0.01,
                },
                'federated': {
                    'enabled': True,
                    'scenario': 'cross_device',
                    'num_devices': num_clients,
                    'devices_per_round': max(2, num_clients // 10),
                    'num_rounds': 100,
                    'local_epochs': 1,
                    'aggregation': 'fedavg',
                    'privacy': {
                        'enabled': False,
                        'mechanism': 'gaussian',
                        'epsilon': 1.0,
                        'delta': 1e-5,
                    },
                    'compression': {
                        'enabled': False,
                        'type': 'quantization',
                        'num_bits': 8,
                    },
                },
            }
        elif scenario == 'cross_silo':
            config = {
                'backend': 'tensorflow',
                'federated': {
                    'scenario': 'cross_silo',
                    'num_silos': num_clients,
                    'silos_per_round': max(2, num_clients // 2),
                    'num_rounds': 50,
                    'local_epochs': 5,
                    'aggregation': 'fedavg',
                },
            }
        else:
            config = {
                'backend': 'tensorflow',
                'federated': {
                    'scenario': 'hybrid',
                    'num_silos': max(2, num_clients // 10),
                    'devices_per_silo': 10,
                    'num_rounds': 100,
                    'aggregation': 'fedavg',
                },
            }
        
        with open(output, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Configuration template generated: {output}")
        
    except Exception as e:
        logger.error(f"Config generation failed: {e}")
        raise click.ClickException(str(e))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    federated()
