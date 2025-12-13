"""
Architecture space definition and construction for NAS.

Provides DSL-based architecture space definition and dynamic architecture building.
"""
from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class LayerChoice:
    """Represents a choice among multiple layer configurations."""
    
    def __init__(self, name: str, choices: List[Dict[str, Any]]):
        self.name = name
        self.choices = choices
        self.num_choices = len(choices)
    
    def sample(self, idx: Optional[int] = None) -> Dict[str, Any]:
        """Sample a layer configuration."""
        if idx is None:
            idx = np.random.randint(self.num_choices)
        return copy.deepcopy(self.choices[idx % self.num_choices])
    
    def get_all_choices(self) -> List[Dict[str, Any]]:
        """Get all possible layer configurations."""
        return [copy.deepcopy(c) for c in self.choices]


class ArchitectureSpace:
    """Defines the search space for neural architecture search."""
    
    def __init__(self):
        self.input_shape = None
        self.output_shape = None
        self.layer_choices: List[LayerChoice] = []
        self.fixed_layers: List[Dict[str, Any]] = []
        self.hyperparameters: Dict[str, Any] = {}
        self.constraints: Dict[str, Any] = {}
        
    def add_layer_choice(self, name: str, choices: List[Dict[str, Any]]):
        """Add a layer choice to the architecture space."""
        self.layer_choices.append(LayerChoice(name, choices))
    
    def add_fixed_layer(self, layer_config: Dict[str, Any]):
        """Add a fixed layer (no choices) to the architecture."""
        self.fixed_layers.append(layer_config)
    
    def add_hyperparameter(self, name: str, param_space: Dict[str, Any]):
        """Add a hyperparameter search space."""
        self.hyperparameters[name] = param_space
    
    def add_constraint(self, name: str, constraint: Dict[str, Any]):
        """Add an architecture constraint."""
        self.constraints[name] = constraint
    
    def sample_architecture(self, choices: Optional[List[int]] = None) -> Dict[str, Any]:
        """Sample a complete architecture from the search space."""
        layers = []
        choice_idx = 0
        
        for layer_choice in self.layer_choices:
            if choices and choice_idx < len(choices):
                layer = layer_choice.sample(choices[choice_idx])
            else:
                layer = layer_choice.sample()
            layers.append(layer)
            choice_idx += 1
        
        layers.extend(self.fixed_layers)
        
        architecture = {
            'input': {'shape': self.input_shape},
            'layers': layers,
            'optimizer': self.hyperparameters.get('optimizer', {'type': 'Adam', 'params': {'learning_rate': 0.001}}),
            'training_config': self.hyperparameters.get('training_config', {'batch_size': 32, 'epochs': 10})
        }
        
        return architecture
    
    def get_search_space_size(self) -> int:
        """Calculate the total size of the search space."""
        size = 1
        for layer_choice in self.layer_choices:
            size *= layer_choice.num_choices
        
        for param_name, param_space in self.hyperparameters.items():
            if param_space.get('type') == 'categorical':
                size *= len(param_space['values'])
        
        return size
    
    @classmethod
    def from_dsl(cls, dsl_config: str) -> ArchitectureSpace:
        """
        Parse architecture space from DSL configuration.
        
        The DSL supports special syntax for NAS:
        - LayerChoice[Layer1, Layer2, Layer3] - choice among layers
        - Layer(param=search_range(min, max)) - parameter search
        - Layer(param=search_choice([val1, val2, val3])) - categorical choice
        """
        space = cls()
        
        try:
            from neural.parser.parser import ModelTransformer
            transformer = ModelTransformer()
            
            if 'search_space' in dsl_config or 'LayerChoice' in dsl_config:
                space = cls._parse_nas_dsl(dsl_config, transformer)
            else:
                model_dict, hpo_params = transformer.parse_network_with_hpo(dsl_config)
                space = cls._convert_from_hpo(model_dict, hpo_params)
        
        except Exception as e:
            logger.error(f"Failed to parse DSL config: {e}")
            raise ValueError(f"Invalid DSL configuration: {e}")
        
        return space
    
    @classmethod
    def _parse_nas_dsl(cls, dsl_config: str, transformer) -> ArchitectureSpace:
        """Parse NAS-specific DSL syntax."""
        space = cls()
        
        lines = dsl_config.strip().split('\n')
        for line in lines:
            line = line.strip()
            
            if line.startswith('input:'):
                shape_str = line.split('input:')[1].strip()
                space.input_shape = eval(shape_str)
            
            elif 'LayerChoice[' in line:
                name = line.split('LayerChoice[')[0].strip()
                choices_str = line.split('LayerChoice[')[1].split(']')[0]
                choices_list = [c.strip() for c in choices_str.split(',')]
                
                choices = []
                for choice_str in choices_list:
                    layer_config = cls._parse_layer_string(choice_str)
                    choices.append(layer_config)
                
                space.add_layer_choice(name, choices)
            
            elif 'search_range(' in line or 'search_choice(' in line:
                param_name = line.split('=')[0].strip().split('(')[1].strip()
                if 'search_range(' in line:
                    range_str = line.split('search_range(')[1].split(')')[0]
                    min_val, max_val = [float(x.strip()) for x in range_str.split(',')]
                    space.add_hyperparameter(param_name, {
                        'type': 'range',
                        'min': min_val,
                        'max': max_val
                    })
                elif 'search_choice(' in line:
                    choices_str = line.split('search_choice([')[1].split('])')[0]
                    choices = [eval(x.strip()) for x in choices_str.split(',')]
                    space.add_hyperparameter(param_name, {
                        'type': 'categorical',
                        'values': choices
                    })
        
        return space
    
    @classmethod
    def _convert_from_hpo(cls, model_dict: Dict[str, Any], hpo_params: List[Dict[str, Any]]) -> ArchitectureSpace:
        """Convert HPO configuration to architecture space."""
        space = cls()
        space.input_shape = model_dict['input']['shape']
        
        for layer in model_dict['layers']:
            if layer.get('params') and any(
                isinstance(v, dict) and 'hpo' in v 
                for v in layer['params'].values()
            ):
                choices = cls._generate_layer_choices_from_hpo(layer)
                space.add_layer_choice(layer['type'], choices)
            else:
                space.add_fixed_layer(layer)
        
        if model_dict.get('optimizer'):
            space.add_hyperparameter('optimizer', model_dict['optimizer'])
        
        if model_dict.get('training_config'):
            space.add_hyperparameter('training_config', model_dict['training_config'])
        
        return space
    
    @staticmethod
    def _parse_layer_string(layer_str: str) -> Dict[str, Any]:
        """Parse a layer string into a configuration dict."""
        layer_type = layer_str.split('(')[0].strip()
        params = {}
        
        if '(' in layer_str:
            params_str = layer_str.split('(')[1].split(')')[0]
            for param in params_str.split(','):
                if '=' in param:
                    key, value = param.split('=')
                    params[key.strip()] = eval(value.strip())
        
        return {'type': layer_type, 'params': params}
    
    @staticmethod
    def _generate_layer_choices_from_hpo(layer: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate layer choices from HPO parameters."""
        choices = []
        
        hpo_params = {
            k: v for k, v in layer['params'].items()
            if isinstance(v, dict) and 'hpo' in v
        }
        
        if not hpo_params:
            return [layer]
        
        param_name = list(hpo_params.keys())[0]
        hpo_spec = hpo_params[param_name]['hpo']
        
        if hpo_spec['type'] == 'categorical':
            for value in hpo_spec['values']:
                new_layer = copy.deepcopy(layer)
                new_layer['params'][param_name] = value
                choices.append(new_layer)
        elif hpo_spec['type'] == 'range':
            start, end = hpo_spec['start'], hpo_spec['end']
            step = hpo_spec.get('step', (end - start) / 5)
            for value in np.arange(start, end + step, step):
                new_layer = copy.deepcopy(layer)
                new_layer['params'][param_name] = float(value)
                choices.append(new_layer)
        
        return choices if choices else [layer]


class ArchitectureBuilder:
    """Builds actual model instances from architecture specifications."""
    
    def __init__(self, backend: str = 'pytorch'):
        self.backend = backend
    
    def build(self, architecture: Dict[str, Any], trial=None):
        """Build a model from an architecture specification."""
        if self.backend == 'pytorch':
            return self._build_pytorch_model(architecture, trial)
        elif self.backend == 'tensorflow':
            return self._build_tensorflow_model(architecture, trial)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def _build_pytorch_model(self, architecture: Dict[str, Any], trial=None):
        """Build a PyTorch model."""
        try:
            from neural.hpo.hpo import DynamicPTModel
            
            hpo_params = []
            if trial:
                return DynamicPTModel(architecture, trial, hpo_params)
            else:
                import optuna
                study = optuna.create_study()
                trial = study.ask()
                return DynamicPTModel(architecture, trial, hpo_params)
        
        except ImportError as e:
            logger.error(f"Failed to import PyTorch dependencies: {e}")
            raise
    
    def _build_tensorflow_model(self, architecture: Dict[str, Any], trial=None):
        """Build a TensorFlow model."""
        try:
            from neural.hpo.hpo import DynamicTFModel
            
            hpo_params = []
            if trial:
                return DynamicTFModel(architecture, trial, hpo_params)
            else:
                import optuna
                study = optuna.create_study()
                trial = study.ask()
                return DynamicTFModel(architecture, trial, hpo_params)
        
        except ImportError as e:
            logger.error(f"Failed to import TensorFlow dependencies: {e}")
            raise
