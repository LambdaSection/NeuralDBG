from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from neural.federated.client import FederatedClient
from neural.federated.communication import CompressionStrategy

logger = logging.getLogger(__name__)


class FederatedScenario(ABC):
    def __init__(
        self,
        name: str,
        min_clients: int = 2,
        max_clients: int = 100,
    ):
        self.name = name
        self.min_clients = min_clients
        self.max_clients = max_clients
        self.clients = []
    
    @abstractmethod
    def create_clients(self, **kwargs) -> List[FederatedClient]:
        pass
    
    @abstractmethod
    def get_client_selection_strategy(self) -> str:
        pass
    
    @abstractmethod
    def get_aggregation_config(self) -> Dict[str, Any]:
        pass


class CrossDeviceScenario(FederatedScenario):
    def __init__(
        self,
        num_devices: int = 100,
        devices_per_round: int = 10,
        data_heterogeneity: float = 0.5,
        device_availability: float = 0.8,
    ):
        super().__init__(
            name='cross_device',
            min_clients=max(2, devices_per_round // 2),
            max_clients=num_devices,
        )
        self.num_devices = num_devices
        self.devices_per_round = devices_per_round
        self.data_heterogeneity = data_heterogeneity
        self.device_availability = device_availability
    
    def create_clients(
        self,
        model_fn: Any,
        data: Tuple[np.ndarray, np.ndarray],
        backend: str = 'tensorflow',
        **kwargs
    ) -> List[FederatedClient]:
        X, y = data
        
        clients = []
        num_samples = len(X)
        
        alpha = self.data_heterogeneity * 10
        if alpha > 0:
            proportions = np.random.dirichlet([alpha] * self.num_devices)
        else:
            proportions = np.ones(self.num_devices) / self.num_devices
        
        start_idx = 0
        for i in range(self.num_devices):
            num_client_samples = int(proportions[i] * num_samples)
            end_idx = min(start_idx + num_client_samples, num_samples)
            
            if start_idx >= end_idx:
                continue
            
            X_client = X[start_idx:end_idx]
            y_client = y[start_idx:end_idx]
            
            if len(X_client) == 0:
                continue
            
            compute_capability = np.random.uniform(0.5, 1.0)
            bandwidth = np.random.uniform(0.3, 1.0)
            
            client = FederatedClient(
                client_id=f'device_{i}',
                model=model_fn(),
                local_data=(X_client, y_client),
                backend=backend,
                local_epochs=kwargs.get('local_epochs', 1),
                batch_size=min(32, len(X_client)),
                learning_rate=kwargs.get('learning_rate', 0.01),
                compute_capability=compute_capability,
                bandwidth=bandwidth,
            )
            clients.append(client)
            
            start_idx = end_idx
        
        logger.info(f"Created {len(clients)} cross-device clients")
        self.clients = clients
        return clients
    
    def get_client_selection_strategy(self) -> str:
        return 'random'
    
    def get_aggregation_config(self) -> Dict[str, Any]:
        return {
            'strategy': 'fedavg',
            'use_secure_aggregation': True,
            'compression': {
                'enabled': True,
                'type': 'quantization',
                'num_bits': 4,
            },
        }
    
    def simulate_availability(self, clients: List[FederatedClient]) -> List[FederatedClient]:
        available_clients = []
        for client in clients:
            if np.random.rand() < self.device_availability:
                available_clients.append(client)
        return available_clients


class CrossSiloScenario(FederatedScenario):
    def __init__(
        self,
        num_silos: int = 10,
        silos_per_round: int = 5,
        data_heterogeneity: float = 0.3,
    ):
        super().__init__(
            name='cross_silo',
            min_clients=2,
            max_clients=num_silos,
        )
        self.num_silos = num_silos
        self.silos_per_round = silos_per_round
        self.data_heterogeneity = data_heterogeneity
    
    def create_clients(
        self,
        model_fn: Any,
        data: Tuple[np.ndarray, np.ndarray],
        backend: str = 'tensorflow',
        **kwargs
    ) -> List[FederatedClient]:
        X, y = data
        
        clients = []
        num_samples = len(X)
        
        alpha = self.data_heterogeneity * 10
        if alpha > 0:
            proportions = np.random.dirichlet([alpha] * self.num_silos)
        else:
            proportions = np.ones(self.num_silos) / self.num_silos
        
        start_idx = 0
        for i in range(self.num_silos):
            num_client_samples = int(proportions[i] * num_samples)
            end_idx = min(start_idx + num_client_samples, num_samples)
            
            if start_idx >= end_idx:
                continue
            
            X_client = X[start_idx:end_idx]
            y_client = y[start_idx:end_idx]
            
            if len(X_client) == 0:
                continue
            
            compute_capability = np.random.uniform(0.8, 1.0)
            bandwidth = np.random.uniform(0.7, 1.0)
            
            client = FederatedClient(
                client_id=f'silo_{i}',
                model=model_fn(),
                local_data=(X_client, y_client),
                backend=backend,
                local_epochs=kwargs.get('local_epochs', 5),
                batch_size=min(128, len(X_client)),
                learning_rate=kwargs.get('learning_rate', 0.01),
                compute_capability=compute_capability,
                bandwidth=bandwidth,
            )
            clients.append(client)
            
            start_idx = end_idx
        
        logger.info(f"Created {len(clients)} cross-silo clients")
        self.clients = clients
        return clients
    
    def get_client_selection_strategy(self) -> str:
        return 'resource_aware'
    
    def get_aggregation_config(self) -> Dict[str, Any]:
        return {
            'strategy': 'fedavg',
            'use_secure_aggregation': False,
            'compression': {
                'enabled': False,
            },
        }


class HybridScenario(FederatedScenario):
    def __init__(
        self,
        num_silos: int = 5,
        devices_per_silo: int = 20,
        silos_per_round: int = 3,
        devices_per_silo_per_round: int = 5,
    ):
        super().__init__(
            name='hybrid',
            min_clients=2,
            max_clients=num_silos * devices_per_silo,
        )
        self.num_silos = num_silos
        self.devices_per_silo = devices_per_silo
        self.silos_per_round = silos_per_round
        self.devices_per_silo_per_round = devices_per_silo_per_round
        self.silo_clients = {}
    
    def create_clients(
        self,
        model_fn: Any,
        data: Tuple[np.ndarray, np.ndarray],
        backend: str = 'tensorflow',
        **kwargs
    ) -> List[FederatedClient]:
        X, y = data
        
        clients = []
        num_samples = len(X)
        
        silo_proportions = np.random.dirichlet([1.0] * self.num_silos)
        
        start_idx = 0
        for silo_id in range(self.num_silos):
            silo_samples = int(silo_proportions[silo_id] * num_samples)
            silo_end_idx = min(start_idx + silo_samples, num_samples)
            
            if start_idx >= silo_end_idx:
                continue
            
            X_silo = X[start_idx:silo_end_idx]
            y_silo = y[start_idx:silo_end_idx]
            
            device_proportions = np.random.dirichlet([1.0] * self.devices_per_silo)
            
            silo_clients = []
            device_start = 0
            for device_id in range(self.devices_per_silo):
                device_samples = int(device_proportions[device_id] * len(X_silo))
                device_end = min(device_start + device_samples, len(X_silo))
                
                if device_start >= device_end:
                    continue
                
                X_device = X_silo[device_start:device_end]
                y_device = y_silo[device_start:device_end]
                
                if len(X_device) == 0:
                    continue
                
                client = FederatedClient(
                    client_id=f'silo_{silo_id}_device_{device_id}',
                    model=model_fn(),
                    local_data=(X_device, y_device),
                    backend=backend,
                    local_epochs=kwargs.get('local_epochs', 2),
                    batch_size=min(32, len(X_device)),
                    learning_rate=kwargs.get('learning_rate', 0.01),
                    compute_capability=np.random.uniform(0.5, 1.0),
                    bandwidth=np.random.uniform(0.3, 1.0),
                )
                silo_clients.append(client)
                clients.append(client)
                
                device_start = device_end
            
            self.silo_clients[silo_id] = silo_clients
            start_idx = silo_end_idx
        
        logger.info(f"Created {len(clients)} clients in {self.num_silos} silos (hybrid)")
        self.clients = clients
        return clients
    
    def get_client_selection_strategy(self) -> str:
        return 'hierarchical'
    
    def get_aggregation_config(self) -> Dict[str, Any]:
        return {
            'strategy': 'hierarchical_fedavg',
            'use_secure_aggregation': True,
            'compression': {
                'enabled': True,
                'type': 'adaptive',
            },
            'hierarchical': {
                'silo_aggregation': 'fedavg',
                'global_aggregation': 'fedavg',
            },
        }
    
    def select_hierarchical_clients(self) -> List[FederatedClient]:
        selected_silos = np.random.choice(
            self.num_silos,
            self.silos_per_round,
            replace=False
        )
        
        selected_clients = []
        for silo_id in selected_silos:
            if silo_id not in self.silo_clients:
                continue
            
            silo_clients = self.silo_clients[silo_id]
            num_to_select = min(self.devices_per_silo_per_round, len(silo_clients))
            
            selected_indices = np.random.choice(
                len(silo_clients),
                num_to_select,
                replace=False
            )
            
            for idx in selected_indices:
                selected_clients.append(silo_clients[idx])
        
        return selected_clients


class VerticalFederatedScenario(FederatedScenario):
    def __init__(
        self,
        num_parties: int = 3,
    ):
        super().__init__(
            name='vertical_federated',
            min_clients=2,
            max_clients=num_parties,
        )
        self.num_parties = num_parties
    
    def create_clients(
        self,
        model_fn: Any,
        data: Tuple[np.ndarray, np.ndarray],
        backend: str = 'tensorflow',
        **kwargs
    ) -> List[FederatedClient]:
        X, y = data
        
        if len(X.shape) < 2:
            raise ValueError("Data must be at least 2-dimensional for vertical FL")
        
        num_features = X.shape[1] if len(X.shape) == 2 else np.prod(X.shape[1:])
        X_flat = X.reshape(len(X), -1)
        
        feature_splits = np.array_split(range(num_features), self.num_parties)
        
        clients = []
        for i, feature_indices in enumerate(feature_splits):
            X_party = X_flat[:, feature_indices]
            
            if i == 0:
                local_data = (X_party, y)
            else:
                local_data = (X_party, np.zeros(len(y)))
            
            client = FederatedClient(
                client_id=f'party_{i}',
                model=model_fn(),
                local_data=local_data,
                backend=backend,
                local_epochs=kwargs.get('local_epochs', 1),
                batch_size=kwargs.get('batch_size', 32),
                learning_rate=kwargs.get('learning_rate', 0.01),
            )
            clients.append(client)
        
        logger.info(f"Created {len(clients)} parties for vertical FL")
        self.clients = clients
        return clients
    
    def get_client_selection_strategy(self) -> str:
        return 'all'
    
    def get_aggregation_config(self) -> Dict[str, Any]:
        return {
            'strategy': 'vertical_aggregation',
            'use_secure_aggregation': True,
            'privacy': {
                'enabled': True,
                'mechanism': 'secure_multiparty_computation',
            },
        }
