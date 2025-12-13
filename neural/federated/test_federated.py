from __future__ import annotations

import unittest

import numpy as np


class TestFederatedClient(unittest.TestCase):
    def test_client_creation(self):
        from neural.federated.client import FederatedClient
        
        X = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randint(0, 2, 100)
        
        client = FederatedClient(
            client_id='test_client',
            model=None,
            local_data=(X, y),
            backend='tensorflow',
            local_epochs=1,
        )
        
        self.assertEqual(client.client_id, 'test_client')
        self.assertEqual(client.num_samples, 100)


class TestFederatedServer(unittest.TestCase):
    def test_server_creation(self):
        from neural.federated.server import FederatedServer
        
        server = FederatedServer(
            model=None,
            backend='tensorflow',
        )
        
        self.assertIsNotNone(server)
        self.assertEqual(server.current_round, 0)


class TestAggregation(unittest.TestCase):
    def test_fedavg(self):
        from neural.federated.aggregation import FedAvg
        
        weights1 = [np.array([[1.0, 2.0], [3.0, 4.0]])]
        weights2 = [np.array([[2.0, 3.0], [4.0, 5.0]])]
        weights_list = [weights1, weights2]
        num_samples_list = [100, 100]
        
        aggregator = FedAvg()
        result = aggregator.aggregate(weights_list, num_samples_list)
        
        expected = np.array([[1.5, 2.5], [3.5, 4.5]])
        np.testing.assert_array_almost_equal(result[0], expected)
    
    def test_weighted_fedavg(self):
        from neural.federated.aggregation import FedAvg
        
        weights1 = [np.array([[1.0, 2.0]])]
        weights2 = [np.array([[3.0, 4.0]])]
        weights_list = [weights1, weights2]
        num_samples_list = [300, 100]
        
        aggregator = FedAvg()
        result = aggregator.aggregate(weights_list, num_samples_list)
        
        expected = np.array([[1.5, 2.5]])
        np.testing.assert_array_almost_equal(result[0], expected)


class TestPrivacy(unittest.TestCase):
    def test_gaussian_dp(self):
        from neural.federated.privacy import GaussianDP
        
        dp = GaussianDP(epsilon=1.0, delta=1e-5, clip_norm=1.0)
        
        weights = [np.array([[1.0, 2.0], [3.0, 4.0]])]
        noisy_weights = dp.add_noise(weights, sensitivity=1.0)
        
        self.assertEqual(len(noisy_weights), 1)
        self.assertEqual(noisy_weights[0].shape, (2, 2))
    
    def test_privacy_accountant(self):
        from neural.federated.privacy import PrivacyAccountant
        
        accountant = PrivacyAccountant(epsilon_total=1.0, delta_total=1e-5)
        
        success = accountant.spend_privacy_budget(0.1, 1e-6, 'test')
        self.assertTrue(success)
        
        remaining_eps, remaining_delta = accountant.get_remaining_budget()
        self.assertAlmostEqual(remaining_eps, 0.9)


class TestCompression(unittest.TestCase):
    def test_quantization(self):
        from neural.federated.communication import QuantizationCompressor
        
        compressor = QuantizationCompressor(num_bits=8)
        
        weights = [np.random.randn(10, 10).astype(np.float32)]
        compressed, metadata = compressor.compress(weights)
        decompressed = compressor.decompress(compressed, metadata)
        
        self.assertEqual(len(decompressed), 1)
        self.assertEqual(decompressed[0].shape, (10, 10))
    
    def test_sparsification(self):
        from neural.federated.communication import SparsificationCompressor
        
        compressor = SparsificationCompressor(sparsity=0.9, method='topk')
        
        weights = [np.random.randn(100).astype(np.float32)]
        compressed, metadata = compressor.compress(weights)
        
        self.assertLessEqual(len(compressed[0]), 10)


class TestScenarios(unittest.TestCase):
    def test_cross_device_scenario(self):
        from neural.federated.scenarios import CrossDeviceScenario
        
        scenario = CrossDeviceScenario(
            num_devices=10,
            devices_per_round=5,
        )
        
        self.assertEqual(scenario.num_devices, 10)
        self.assertEqual(scenario.name, 'cross_device')
    
    def test_cross_silo_scenario(self):
        from neural.federated.scenarios import CrossSiloScenario
        
        scenario = CrossSiloScenario(
            num_silos=5,
            silos_per_round=3,
        )
        
        self.assertEqual(scenario.num_silos, 5)
        self.assertEqual(scenario.name, 'cross_silo')


class TestUtils(unittest.TestCase):
    def test_split_data_iid(self):
        from neural.federated.utils import split_data_iid
        
        X = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randint(0, 2, 100)
        
        client_data = split_data_iid((X, y), num_clients=5)
        
        self.assertEqual(len(client_data), 5)
        self.assertEqual(len(client_data[0][0]), 20)
    
    def test_split_data_non_iid(self):
        from neural.federated.utils import split_data_non_iid
        
        X = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randint(0, 2, 100)
        
        client_data = split_data_non_iid((X, y), num_clients=5, alpha=0.5)
        
        self.assertGreater(len(client_data), 0)
    
    def test_compute_gradient_norm(self):
        from neural.federated.utils import compute_gradient_norm
        
        weights = [np.array([[3.0, 4.0]])]
        norm = compute_gradient_norm(weights)
        
        self.assertAlmostEqual(norm, 5.0)
    
    def test_cosine_similarity(self):
        from neural.federated.utils import cosine_similarity
        
        weights1 = [np.array([[1.0, 0.0]])]
        weights2 = [np.array([[0.0, 1.0]])]
        
        similarity = cosine_similarity(weights1, weights2)
        self.assertAlmostEqual(similarity, 0.0)


if __name__ == '__main__':
    unittest.main()
