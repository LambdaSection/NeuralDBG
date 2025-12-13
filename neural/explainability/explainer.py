"""
Main explainer interface that orchestrates various explainability methods.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ModelExplainer:
    """
    Unified interface for model explainability and interpretability.
    
    This class provides a high-level API to access various explainability methods:
    - SHAP (SHapley Additive exPlanations)
    - LIME (Local Interpretable Model-agnostic Explanations)
    - Attention visualization
    - Gradient-based saliency maps
    - Feature importance ranking
    - Counterfactual explanations
    """
    
    def __init__(
        self,
        model: Any,
        backend: str = 'tensorflow',
        task_type: str = 'classification',
        input_shape: Optional[tuple] = None,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize the ModelExplainer.
        
        Args:
            model: The trained model to explain
            backend: ML framework ('tensorflow', 'pytorch', 'onnx')
            task_type: Type of task ('classification', 'regression', 'segmentation')
            input_shape: Shape of input data (excluding batch dimension)
            feature_names: Names of input features
            class_names: Names of output classes (for classification)
        """
        self.model = model
        self.backend = backend.lower()
        self.task_type = task_type
        self.input_shape = input_shape
        self.feature_names = feature_names
        self.class_names = class_names
        
        self._shap_explainer = None
        self._lime_explainer = None
        self._attention_visualizer = None
        self._saliency_generator = None
        self._feature_ranker = None
        self._counterfactual_generator = None
        
        logger.info(f"Initialized ModelExplainer for {backend} model with task: {task_type}")
    
    def explain_prediction(
        self,
        input_data: np.ndarray,
        method: str = 'shap',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Explain a single prediction using the specified method.
        
        Args:
            input_data: Input sample to explain
            method: Explainability method ('shap', 'lime', 'saliency', 'all')
            **kwargs: Additional arguments for the specific method
            
        Returns:
            Dictionary containing explanation results
        """
        if method == 'shap':
            return self._explain_with_shap(input_data, **kwargs)
        elif method == 'lime':
            return self._explain_with_lime(input_data, **kwargs)
        elif method == 'saliency':
            return self._explain_with_saliency(input_data, **kwargs)
        elif method == 'all':
            return {
                'shap': self._explain_with_shap(input_data, **kwargs),
                'lime': self._explain_with_lime(input_data, **kwargs),
                'saliency': self._explain_with_saliency(input_data, **kwargs)
            }
        else:
            raise ValueError(f"Unknown method: {method}. Choose from 'shap', 'lime', 'saliency', 'all'")
    
    def _explain_with_shap(self, input_data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Explain using SHAP."""
        if self._shap_explainer is None:
            from .shap_explainer import SHAPExplainer
            self._shap_explainer = SHAPExplainer(
                self.model,
                backend=self.backend,
                task_type=self.task_type
            )
        
        return self._shap_explainer.explain(input_data, **kwargs)
    
    def _explain_with_lime(self, input_data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Explain using LIME."""
        if self._lime_explainer is None:
            from .lime_explainer import LIMEExplainer
            self._lime_explainer = LIMEExplainer(
                self.model,
                backend=self.backend,
                task_type=self.task_type,
                feature_names=self.feature_names,
                class_names=self.class_names
            )
        
        return self._lime_explainer.explain(input_data, **kwargs)
    
    def _explain_with_saliency(self, input_data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Explain using gradient-based saliency maps."""
        if self._saliency_generator is None:
            from .saliency_maps import SaliencyMapGenerator
            self._saliency_generator = SaliencyMapGenerator(
                self.model,
                backend=self.backend
            )
        
        return self._saliency_generator.generate(input_data, **kwargs)
    
    def visualize_attention(
        self,
        input_data: np.ndarray,
        layer_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Visualize attention weights for transformer models.
        
        Args:
            input_data: Input sequence data
            layer_name: Name of attention layer to visualize
            **kwargs: Additional visualization parameters
            
        Returns:
            Dictionary containing attention maps and visualizations
        """
        if self._attention_visualizer is None:
            from .attention_visualizer import AttentionVisualizer
            self._attention_visualizer = AttentionVisualizer(
                self.model,
                backend=self.backend
            )
        
        return self._attention_visualizer.visualize(input_data, layer_name, **kwargs)
    
    def rank_features(
        self,
        input_data: np.ndarray,
        method: str = 'permutation',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Rank feature importance.
        
        Args:
            input_data: Input data for feature importance calculation
            method: Ranking method ('permutation', 'gradient', 'integrated_gradient')
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing feature rankings and scores
        """
        if self._feature_ranker is None:
            from .feature_importance import FeatureImportanceRanker
            self._feature_ranker = FeatureImportanceRanker(
                self.model,
                backend=self.backend,
                feature_names=self.feature_names
            )
        
        return self._feature_ranker.rank(input_data, method=method, **kwargs)
    
    def generate_counterfactuals(
        self,
        input_data: np.ndarray,
        target_class: Optional[int] = None,
        num_samples: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate counterfactual explanations.
        
        Args:
            input_data: Input sample to generate counterfactuals for
            target_class: Target class for counterfactuals (if None, uses opposite prediction)
            num_samples: Number of counterfactual samples to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing counterfactual samples and explanations
        """
        if self._counterfactual_generator is None:
            from .counterfactual import CounterfactualGenerator
            self._counterfactual_generator = CounterfactualGenerator(
                self.model,
                backend=self.backend,
                task_type=self.task_type
            )
        
        return self._counterfactual_generator.generate(
            input_data,
            target_class=target_class,
            num_samples=num_samples,
            **kwargs
        )
    
    def generate_model_card(
        self,
        model_info: Dict[str, Any],
        output_path: str = 'model_card.md'
    ) -> str:
        """
        Generate a model card for documentation.
        
        Args:
            model_info: Dictionary containing model information
            output_path: Path to save the model card
            
        Returns:
            Path to generated model card
        """
        from .model_card import ModelCardGenerator
        generator = ModelCardGenerator()
        return generator.generate(model_info, output_path)
    
    def explain_dataset(
        self,
        dataset: np.ndarray,
        labels: Optional[np.ndarray] = None,
        num_samples: int = 100,
        methods: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate global explanations for a dataset.
        
        Args:
            dataset: Input dataset
            labels: True labels (optional)
            num_samples: Number of samples to analyze
            methods: List of methods to use (default: ['shap', 'feature_importance'])
            
        Returns:
            Dictionary containing global explanations
        """
        if methods is None:
            methods = ['shap', 'feature_importance']
        
        results = {}
        
        sample_indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        sample_data = dataset[sample_indices]
        
        if 'shap' in methods:
            logger.info("Computing SHAP values for dataset...")
            if self._shap_explainer is None:
                from .shap_explainer import SHAPExplainer
                self._shap_explainer = SHAPExplainer(
                    self.model,
                    backend=self.backend,
                    task_type=self.task_type
                )
            results['shap'] = self._shap_explainer.explain_dataset(sample_data)
        
        if 'feature_importance' in methods:
            logger.info("Computing feature importance for dataset...")
            results['feature_importance'] = self.rank_features(sample_data)
        
        return results
