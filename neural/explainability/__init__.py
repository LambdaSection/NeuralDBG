"""
Neural Explainability Module.

This module provides comprehensive model interpretability and explainability tools:
- SHAP and LIME integration for feature importance
- Attention visualization for transformer models
- Gradient-based saliency maps
- Feature importance ranking
- Counterfactual explanations
- Model card generation for documentation
"""

from .explainer import ModelExplainer
from .shap_explainer import SHAPExplainer
from .lime_explainer import LIMEExplainer
from .attention_visualizer import AttentionVisualizer
from .saliency_maps import SaliencyMapGenerator
from .feature_importance import FeatureImportanceRanker
from .counterfactual import CounterfactualGenerator
from .model_card import ModelCardGenerator

__all__ = [
    'ModelExplainer',
    'SHAPExplainer',
    'LIMEExplainer',
    'AttentionVisualizer',
    'SaliencyMapGenerator',
    'FeatureImportanceRanker',
    'CounterfactualGenerator',
    'ModelCardGenerator',
]
