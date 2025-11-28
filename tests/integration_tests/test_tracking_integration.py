import os
import shutil
import sys
import pytest
import json
from neural.parser.parser import create_parser, ModelTransformer
from neural.code_generation.code_generator import generate_code

class TestTrackingIntegration:
    def setup_method(self):
        # Clean up experiments directory
        if os.path.exists("neural_experiments"):
            shutil.rmtree("neural_experiments")
        
        # Create model data manually to avoid parser syntax issues
        self.model_data = {
            "input": {"shape": [28, 28, 1]},
            "layers": [
                {"type": "Flatten"},
                {"type": "Dense", "params": {"units": 10, "activation": "relu"}},
                {"type": "Output", "params": {"units": 10, "activation": "softmax"}}
            ],
            "optimizer": {"type": "Adam"},
            "training_config": {"epochs": 1, "batch_size": 32}
        }

    def teardown_method(self):
        # Clean up generated files
        if os.path.exists("test_model_tf.py"):
            os.remove("test_model_tf.py")
        if os.path.exists("test_model_pt.py"):
            os.remove("test_model_pt.py")
        if os.path.exists("neural_experiments"):
            shutil.rmtree("neural_experiments")

    def test_tensorflow_tracking(self):
        # Generate TensorFlow code
        code = generate_code(self.model_data, 'tensorflow')
        
        # Validate that tracking imports are present
        assert "from neural.tracking.experiment_tracker import ExperimentManager" in code
        assert "experiment_manager = ExperimentManager()" in code
        assert "experiment = experiment_manager.create_experiment()" in code
        
        # Validate that hyperparameters are logged
        assert "experiment.log_hyperparameters" in code
        assert "'optimizer': 'Adam'" in code
        assert "'backend': 'tensorflow'" in code
        
        # Validate that the callback is defined
        assert "class NeuralTrackingCallback" in code
        assert "def on_epoch_end(self, epoch, logs=None):" in code
        assert "self.experiment.log_metrics(logs, step=epoch)" in code
        
        # Validate that the callback is used in model.fit
        assert "callbacks=[NeuralTrackingCallback(experiment)]" in code

    def test_pytorch_tracking(self):
        # Generate PyTorch code
        code = generate_code(self.model_data, 'pytorch')
        
        # Validate that tracking imports are present
        assert "from neural.tracking.experiment_tracker import ExperimentManager" in code
        assert "experiment_manager = ExperimentManager()" in code
        assert "experiment = experiment_manager.create_experiment()" in code
        
        # Validate that hyperparameters are logged
        assert "experiment.log_hyperparameters" in code
        assert "'optimizer': 'Adam'" in code
        assert "'backend': 'pytorch'" in code
        
        # Validate that metrics are logged after training
        assert "experiment.log_metrics" in code
        assert "'loss': avg_loss" in code
        assert "'accuracy': accuracy" in code
