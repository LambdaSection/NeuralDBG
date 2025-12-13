from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from neural.federated.aggregation import (
    AdaptiveAggregation,
    FedAdam,
    FedAvg,
    FedMA,
    FedProx,
    FedYogi,
    SecureAggregator,
)
from neural.federated.client import FederatedClient
from neural.federated.communication import (
    AdaptiveCompressor,
    CommunicationScheduler,
    GradientCompression,
    QuantizationCompressor,
    SparsificationCompressor,
)
from neural.federated.privacy import (
    AdaptivePrivacy,
    GaussianDP,
    LaplacianDP,
    PrivacyAccountant,
    ShuffleDP,
)
from neural.federated.scenarios import (
    CrossDeviceScenario,
    CrossSiloScenario,
    HybridScenario,
)
from neural.federated.server import FederatedServer

logger = logging.getLogger(__name__)


class FederatedOrchestrator:
    def __init__(
        self,
        model: Any,
        backend: str = 'tensorflow',
        scenario: Optional[Any] = None,
        aggregation_strategy: str = 'fedavg',
        privacy_mechanism: Optional[str] = None,
        compression_strategy: Optional[str] = None,
        **kwargs
    ):
        self.model = model
        self.backend = backend
        self.scenario = scenario or CrossDeviceScenario()
        
        self.aggregation_strategy = self._create_aggregation_strategy(
            aggregation_strategy,
            kwargs,
        )
        
        self.privacy_mechanism = self._create_privacy_mechanism(
            privacy_mechanism,
            kwargs,
        )
        
        self.compression_strategy = self._create_compression_strategy(
            compression_strategy,
            kwargs,
        )
        
        self.server = FederatedServer(
            model=model,
            aggregation_strategy=self.aggregation_strategy,
            backend=backend,
            min_available_clients=kwargs.get('min_available_clients', 2),
            fraction_fit=kwargs.get('fraction_fit', 1.0),
            fraction_evaluate=kwargs.get('fraction_evaluate', 1.0),
        )
        
        self.clients = []
        self.communication_scheduler = CommunicationScheduler(
            initial_interval=kwargs.get('communication_interval', 1),
            max_interval=kwargs.get('max_communication_interval', 10),
        )
        
        self.gradient_compressor = None
        if self.compression_strategy:
            self.gradient_compressor = GradientCompression(
                compression_strategy=self.compression_strategy,
                error_feedback=kwargs.get('error_feedback', True),
            )
        
        self.config = kwargs
        self.metrics_history = {
            'rounds': [],
            'train_loss': [],
            'train_accuracy': [],
            'eval_loss': [],
            'eval_accuracy': [],
            'communication_cost': [],
            'computation_time': [],
            'privacy_budget_spent': [],
        }
    
    def _create_aggregation_strategy(self, strategy_name: str, config: Dict) -> Any:
        if strategy_name == 'fedavg':
            return FedAvg()
        elif strategy_name == 'fedprox':
            return FedProx(mu=config.get('proximal_mu', 0.01))
        elif strategy_name == 'fedadam':
            return FedAdam(
                learning_rate=config.get('server_lr', 0.01),
                beta_1=config.get('beta_1', 0.9),
                beta_2=config.get('beta_2', 0.999),
            )
        elif strategy_name == 'fedyogi':
            return FedYogi(
                learning_rate=config.get('server_lr', 0.01),
                beta_1=config.get('beta_1', 0.9),
                beta_2=config.get('beta_2', 0.999),
            )
        elif strategy_name == 'fedma':
            return FedMA(sigma=config.get('fedma_sigma', 1.0))
        elif strategy_name == 'adaptive':
            return AdaptiveAggregation(alpha=config.get('adaptive_alpha', 0.5))
        else:
            logger.warning(f"Unknown aggregation strategy: {strategy_name}, using FedAvg")
            return FedAvg()
    
    def _create_privacy_mechanism(self, mechanism_name: Optional[str], config: Dict) -> Any:
        if mechanism_name is None:
            return None
        
        epsilon = config.get('epsilon', 1.0)
        delta = config.get('delta', 1e-5)
        
        if mechanism_name == 'gaussian':
            return GaussianDP(
                epsilon=epsilon,
                delta=delta,
                clip_norm=config.get('clip_norm', 1.0),
            )
        elif mechanism_name == 'laplacian':
            return LaplacianDP(
                epsilon=epsilon,
                delta=delta,
                clip_norm=config.get('clip_norm', 1.0),
            )
        elif mechanism_name == 'shuffle':
            return ShuffleDP(epsilon=epsilon, delta=delta)
        elif mechanism_name == 'adaptive':
            return AdaptivePrivacy(
                epsilon_total=epsilon,
                delta_total=delta,
                num_rounds=config.get('num_rounds', 100),
            )
        else:
            logger.warning(f"Unknown privacy mechanism: {mechanism_name}")
            return None
    
    def _create_compression_strategy(self, strategy_name: Optional[str], config: Dict) -> Any:
        if strategy_name is None:
            return None
        
        if strategy_name == 'quantization':
            return QuantizationCompressor(
                num_bits=config.get('quantization_bits', 8),
                stochastic=config.get('stochastic_quantization', False),
            )
        elif strategy_name == 'sparsification':
            return SparsificationCompressor(
                sparsity=config.get('sparsity', 0.9),
                method=config.get('sparsification_method', 'topk'),
            )
        elif strategy_name == 'adaptive':
            return AdaptiveCompressor(
                target_compression=config.get('target_compression', 0.5),
                quantization_bits=config.get('quantization_bits', 8),
                sparsity=config.get('sparsity', 0.5),
            )
        else:
            logger.warning(f"Unknown compression strategy: {strategy_name}")
            return None
    
    def setup_clients(
        self,
        model_fn: Any,
        data: Tuple[np.ndarray, np.ndarray],
        **kwargs
    ) -> List[FederatedClient]:
        merged_config = {**self.config, **kwargs}
        self.clients = self.scenario.create_clients(
            model_fn=model_fn,
            data=data,
            backend=self.backend,
            **merged_config
        )
        return self.clients
    
    def train(
        self,
        num_rounds: int,
        evaluate_every: int = 1,
        test_data: Optional[Tuple] = None,
        save_model_path: Optional[str] = None,
    ) -> Dict[str, List]:
        if not self.clients:
            raise ValueError("No clients available. Call setup_clients() first.")
        
        logger.info(
            f"Starting federated training with {len(self.clients)} clients "
            f"for {num_rounds} rounds"
        )
        
        use_secure_aggregation = self.config.get('use_secure_aggregation', False)
        proximal_mu = self.config.get('proximal_mu', 0.0)
        
        privacy_accountant = None
        if self.privacy_mechanism and hasattr(self.privacy_mechanism, 'accountant'):
            privacy_accountant = PrivacyAccountant(
                epsilon_total=self.config.get('epsilon', 1.0),
                delta_total=self.config.get('delta', 1e-5),
            )
        
        for round_num in range(num_rounds):
            round_start_time = time.time()
            self.server.current_round = round_num + 1
            
            available_clients = self.clients
            if hasattr(self.scenario, 'simulate_availability'):
                available_clients = self.scenario.simulate_availability(self.clients)
            
            if len(available_clients) < self.server.min_available_clients:
                logger.warning(
                    f"Not enough clients available ({len(available_clients)}), "
                    f"skipping round {round_num + 1}"
                )
                continue
            
            aggregated_weights, train_metrics = self.server.fit_round(
                available_clients,
                use_secure_aggregation=use_secure_aggregation,
                proximal_mu=proximal_mu,
            )
            
            if self.privacy_mechanism:
                sensitivity = self.config.get('sensitivity', 1.0)
                aggregated_weights = self.privacy_mechanism.add_noise(
                    aggregated_weights,
                    sensitivity,
                )
                
                if privacy_accountant:
                    privacy_accountant.spend_privacy_budget(
                        epsilon=self.config.get('epsilon', 1.0) / num_rounds,
                        delta=self.config.get('delta', 1e-5) / num_rounds,
                        operation=f'round_{round_num + 1}',
                    )
            
            self.server.global_weights = aggregated_weights
            self.server.set_weights(aggregated_weights)
            
            round_time = time.time() - round_start_time
            
            self.metrics_history['rounds'].append(round_num + 1)
            self.metrics_history['train_loss'].append(train_metrics.get('loss', 0.0))
            self.metrics_history['train_accuracy'].append(train_metrics.get('accuracy', 0.0))
            self.metrics_history['computation_time'].append(round_time)
            
            if self.compression_strategy:
                compression_ratio = self.compression_strategy.get_compression_ratio()
                self.metrics_history['communication_cost'].append(compression_ratio)
            
            if privacy_accountant:
                remaining_epsilon, _ = privacy_accountant.get_remaining_budget()
                self.metrics_history['privacy_budget_spent'].append(
                    self.config.get('epsilon', 1.0) - remaining_epsilon
                )
            
            logger.info(
                f"Round {round_num + 1}/{num_rounds} - "
                f"Loss: {train_metrics.get('loss', 0.0):.4f}, "
                f"Accuracy: {train_metrics.get('accuracy', 0.0):.4f}, "
                f"Time: {round_time:.2f}s"
            )
            
            if evaluate_every > 0 and (round_num + 1) % evaluate_every == 0:
                eval_metrics = self.server.evaluate_round(
                    available_clients,
                    test_data=test_data,
                )
                self.metrics_history['eval_loss'].append(eval_metrics.get('loss', 0.0))
                self.metrics_history['eval_accuracy'].append(eval_metrics.get('accuracy', 0.0))
                
                logger.info(
                    f"Evaluation - Loss: {eval_metrics.get('loss', 0.0):.4f}, "
                    f"Accuracy: {eval_metrics.get('accuracy', 0.0):.4f}"
                )
                
                if len(self.metrics_history['eval_accuracy']) > 1:
                    improvement = (
                        self.metrics_history['eval_accuracy'][-1] -
                        self.metrics_history['eval_accuracy'][-2]
                    )
                    self.communication_scheduler.update_schedule(improvement)
        
        logger.info("Federated training completed")
        
        if save_model_path:
            self.server.save_model(save_model_path)
        
        if privacy_accountant:
            logger.info(
                f"Privacy budget: "
                f"ε spent = {privacy_accountant.epsilon_spent:.4f}/"
                f"{privacy_accountant.epsilon_total:.4f}"
            )
        
        return self.metrics_history
    
    def get_global_model(self) -> Any:
        return self.server.model
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        if not self.metrics_history['rounds']:
            return {}
        
        summary = {
            'total_rounds': len(self.metrics_history['rounds']),
            'final_train_loss': self.metrics_history['train_loss'][-1] if self.metrics_history['train_loss'] else 0,
            'final_train_accuracy': self.metrics_history['train_accuracy'][-1] if self.metrics_history['train_accuracy'] else 0,
            'avg_computation_time': np.mean(self.metrics_history['computation_time']) if self.metrics_history['computation_time'] else 0,
        }
        
        if self.metrics_history['eval_loss']:
            summary['final_eval_loss'] = self.metrics_history['eval_loss'][-1]
            summary['final_eval_accuracy'] = self.metrics_history['eval_accuracy'][-1]
            summary['best_eval_accuracy'] = max(self.metrics_history['eval_accuracy'])
        
        if self.metrics_history['communication_cost']:
            summary['avg_communication_cost'] = np.mean(self.metrics_history['communication_cost'])
        
        if self.metrics_history['privacy_budget_spent']:
            summary['privacy_budget_spent'] = self.metrics_history['privacy_budget_spent'][-1]
        
        return summary
    
    def save_metrics(self, filepath: str):
        import json
        
        metrics_dict = {
            'history': self.metrics_history,
            'summary': self.get_metrics_summary(),
            'config': {
                'scenario': self.scenario.name,
                'aggregation': type(self.aggregation_strategy).__name__,
                'privacy': type(self.privacy_mechanism).__name__ if self.privacy_mechanism else None,
                'compression': type(self.compression_strategy).__name__ if self.compression_strategy else None,
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        logger.info(f"Metrics saved to {filepath}")


def create_federated_orchestrator(
    model: Any,
    scenario_type: str = 'cross_device',
    backend: str = 'tensorflow',
    **kwargs
) -> FederatedOrchestrator:
    if scenario_type == 'cross_device':
        scenario = CrossDeviceScenario(
            num_devices=kwargs.get('num_devices', 100),
            devices_per_round=kwargs.get('devices_per_round', 10),
            data_heterogeneity=kwargs.get('data_heterogeneity', 0.5),
            device_availability=kwargs.get('device_availability', 0.8),
        )
    elif scenario_type == 'cross_silo':
        scenario = CrossSiloScenario(
            num_silos=kwargs.get('num_silos', 10),
            silos_per_round=kwargs.get('silos_per_round', 5),
            data_heterogeneity=kwargs.get('data_heterogeneity', 0.3),
        )
    elif scenario_type == 'hybrid':
        scenario = HybridScenario(
            num_silos=kwargs.get('num_silos', 5),
            devices_per_silo=kwargs.get('devices_per_silo', 20),
            silos_per_round=kwargs.get('silos_per_round', 3),
            devices_per_silo_per_round=kwargs.get('devices_per_silo_per_round', 5),
        )
    else:
        logger.warning(f"Unknown scenario type: {scenario_type}, using cross_device")
        scenario = CrossDeviceScenario()
    
    return FederatedOrchestrator(
        model=model,
        backend=backend,
        scenario=scenario,
        aggregation_strategy=kwargs.get('aggregation_strategy', 'fedavg'),
        privacy_mechanism=kwargs.get('privacy_mechanism', None),
        compression_strategy=kwargs.get('compression_strategy', None),
        **kwargs
    )
