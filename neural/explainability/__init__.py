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

from .attention_visualizer import AttentionVisualizer
from .counterfactual import CounterfactualGenerator
from .explainer import ModelExplainer
from .feature_importance import FeatureImportanceRanker
from .lime_explainer import LIMEExplainer
from .model_card import ModelCardGenerator
from .saliency_maps import SaliencyMapGenerator
from .shap_explainer import SHAPExplainer


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
