"""
LIME (Local Interpretable Model-agnostic Explanations) integration.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np


logger = logging.getLogger(__name__)


class LIMEExplainer:
    """
    LIME-based explainer for neural network models.
    
    LIME explains individual predictions by learning an interpretable model
    locally around the prediction.
    """
    
    def __init__(
        self,
        model: Any,
        backend: str = 'tensorflow',
        task_type: str = 'classification',
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        mode: str = 'auto'
    ):
        """
        Initialize LIME explainer.
        
        Args:
            model: The model to explain
            backend: ML framework ('tensorflow', 'pytorch', 'onnx')
            task_type: Type of task ('classification', 'regression')
            feature_names: Names of input features
            class_names: Names of output classes
            mode: Data mode ('auto', 'tabular', 'image', 'text')
        """
        self.model = model
        self.backend = backend.lower()
        self.task_type = task_type
        self.feature_names = feature_names
        self.class_names = class_names
        self.mode = mode
        self.explainer = None
        
        logger.info(f"Initialized LIME explainer for {backend} model")
    
    def _create_predict_function(self):
        """Create a prediction function for LIME."""
        if self.backend == 'tensorflow':
            def predict_fn(x):
                preds = self.model(x).numpy()
                if self.task_type == 'classification' and len(preds.shape) == 2:
                    return preds
                return preds
        elif self.backend == 'pytorch':
            import torch
            def predict_fn(x):
                if not isinstance(x, torch.Tensor):
                    x = torch.FloatTensor(x)
                with torch.no_grad():
                    preds = self.model(x)
                    if isinstance(preds, torch.Tensor):
                        return preds.numpy()
                    return preds
        else:
            def predict_fn(x):
                return self.model.predict(x)
        
        return predict_fn
    
    def _detect_mode(self, input_data: np.ndarray) -> str:
        """Detect the data mode from input shape."""
        if len(input_data.shape) == 1:
            return 'tabular'
        elif len(input_data.shape) == 2:
            return 'tabular'
        elif len(input_data.shape) == 3 or len(input_data.shape) == 4:
            return 'image'
        else:
            return 'tabular'
    
    def explain(
        self,
        input_data: np.ndarray,
        num_features: int = 10,
        num_samples: int = 5000,
        top_labels: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate LIME explanation for input data.
        
        Args:
            input_data: Input sample to explain
            num_features: Number of features to include in explanation
            num_samples: Number of samples to generate for local model
            top_labels: Number of top labels to explain (for classification)
            
        Returns:
            Dictionary containing LIME explanations
        """
        try:
            import lime
            import lime.lime_image
            import lime.lime_tabular
        except ImportError:
            raise ImportError(
                "LIME is not installed. Install it with: pip install lime"
            )
        
        if self.mode == 'auto':
            self.mode = self._detect_mode(input_data)
        
        predict_fn = self._create_predict_function()
        
        if self.mode == 'tabular':
            if self.explainer is None:
                training_data = self._generate_training_data(input_data)
                self.explainer = lime.lime_tabular.LimeTabularExplainer(
                    training_data,
                    feature_names=self.feature_names,
                    class_names=self.class_names,
                    mode='classification' if self.task_type == 'classification' else 'regression'
                )
            
            if len(input_data.shape) == 1:
                sample = input_data
            else:
                sample = input_data[0]
            
            explanation = self.explainer.explain_instance(
                sample,
                predict_fn,
                num_features=num_features,
                num_samples=num_samples,
                top_labels=top_labels
            )
            
            results = {
                'explanation': explanation,
                'feature_weights': dict(explanation.as_list()),
                'local_pred': explanation.local_pred,
                'score': explanation.score
            }
        
        elif self.mode == 'image':
            from lime import lime_image
            
            if self.explainer is None:
                self.explainer = lime_image.LimeImageExplainer()
            
            if len(input_data.shape) == 3:
                sample = input_data
            else:
                sample = input_data[0]
            
            explanation = self.explainer.explain_instance(
                sample,
                predict_fn,
                top_labels=top_labels or 5,
                hide_color=0,
                num_samples=num_samples
            )
            
            results = {
                'explanation': explanation,
                'image': sample,
                'segments': explanation.segments
            }
        
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
        
        logger.info(f"Generated LIME explanation for {self.mode} data")
        
        return results
    
    def _generate_training_data(
        self,
        input_data: np.ndarray,
        num_samples: int = 1000
    ) -> np.ndarray:
        """Generate synthetic training data for tabular explainer."""
        if len(input_data.shape) == 1:
            input_data = input_data.reshape(1, -1)
        
        mean = np.mean(input_data, axis=0)
        std = np.std(input_data, axis=0)
        std = np.where(std == 0, 1, std)
        
        training_data = np.random.normal(mean, std, size=(num_samples, input_data.shape[1]))
        
        return training_data
    
    def visualize_explanation(
        self,
        explanation: Any,
        output_path: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Visualize LIME explanation.
        
        Args:
            explanation: LIME explanation object
            output_path: Path to save visualization
            **kwargs: Additional visualization parameters
            
        Returns:
            Figure object
        """
        try:
            import matplotlib.pyplot as plt
            
            if self.mode == 'tabular':
                fig = explanation.as_pyplot_figure(**kwargs)
            elif self.mode == 'image':
                from skimage.segmentation import mark_boundaries
                
                temp, mask = explanation.get_image_and_mask(
                    explanation.top_labels[0],
                    positive_only=kwargs.get('positive_only', True),
                    num_features=kwargs.get('num_features', 5),
                    hide_rest=kwargs.get('hide_rest', False)
                )
                
                fig, ax = plt.subplots()
                ax.imshow(mark_boundaries(temp, mask))
                ax.axis('off')
            else:
                fig = None
            
            if output_path and fig:
                fig.savefig(output_path, bbox_inches='tight', dpi=150)
                logger.info(f"Saved LIME visualization to {output_path}")
            
            return fig
            
        except ImportError:
            logger.warning("matplotlib or skimage not available for visualization")
            return None
