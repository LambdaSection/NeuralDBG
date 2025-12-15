"""
SHAP (SHapley Additive exPlanations) integration for model interpretability.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from neural.exceptions import DependencyError

logger = logging.getLogger(__name__)


# Lazy load SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False


class SHAPExplainer:
    """
    SHAP-based explainer for neural network models.
    
    SHAP values represent the contribution of each feature to the prediction,
    based on game theory concepts.
    """
    
    def __init__(
        self,
        model: Any,
        backend: str = 'tensorflow',
        task_type: str = 'classification',
        explainer_type: str = 'auto'
    ):
        """
        Initialize SHAP explainer.
        
        Args:
            model: The model to explain
            backend: ML framework ('tensorflow', 'pytorch', 'onnx')
            task_type: Type of task ('classification', 'regression')
            explainer_type: SHAP explainer type ('auto', 'deep', 'gradient', 'kernel')
        """
        self.model = model
        self.backend = backend.lower()
        self.task_type = task_type
        self.explainer_type = explainer_type
        self.explainer = None
        
        logger.info(f"Initialized SHAP explainer for {backend} model")
    
    def _create_explainer(self, background_data: Optional[np.ndarray] = None):
        """Create the appropriate SHAP explainer."""
        if not SHAP_AVAILABLE:
            raise DependencyError(
                dependency="shap",
                feature="SHAP explainer",
                install_hint="pip install shap"
            )
        
        if self.explainer_type == 'auto':
            if self.backend in ['tensorflow', 'pytorch']:
                self.explainer_type = 'deep'
            else:
                self.explainer_type = 'kernel'
        
        if self.explainer_type == 'deep':
            if self.backend == 'tensorflow':
                self.explainer = shap.DeepExplainer(self.model, background_data)
            elif self.backend == 'pytorch':
                self.explainer = shap.DeepExplainer(self.model, background_data)
            else:
                raise ValueError(f"DeepExplainer not supported for backend: {self.backend}")
        
        elif self.explainer_type == 'gradient':
            if self.backend == 'tensorflow':
                self.explainer = shap.GradientExplainer(self.model, background_data)
            elif self.backend == 'pytorch':
                self.explainer = shap.GradientExplainer(self.model, background_data)
            else:
                raise ValueError(f"GradientExplainer not supported for backend: {self.backend}")
        
        elif self.explainer_type == 'kernel':
            predict_fn = self._create_predict_function()
            self.explainer = shap.KernelExplainer(predict_fn, background_data)
        
        else:
            raise ValueError(f"Unknown explainer type: {self.explainer_type}")
        
        logger.info(f"Created SHAP {self.explainer_type} explainer")
    
    def _create_predict_function(self):
        """Create a prediction function for the model."""
        if self.backend == 'tensorflow':
            def predict_fn(x):
                return self.model(x).numpy()
        elif self.backend == 'pytorch':
            import torch
            def predict_fn(x):
                if not isinstance(x, torch.Tensor):
                    x = torch.FloatTensor(x)
                with torch.no_grad():
                    return self.model(x).numpy()
        else:
            def predict_fn(x):
                return self.model.predict(x)
        
        return predict_fn
    
    def explain(
        self,
        input_data: np.ndarray,
        background_data: Optional[np.ndarray] = None,
        num_background_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanations for input data.
        
        Args:
            input_data: Input sample(s) to explain
            background_data: Background dataset for explainer
            num_background_samples: Number of background samples to use
            
        Returns:
            Dictionary containing SHAP values and visualizations
        """
        if background_data is None:
            background_data = self._generate_background_data(input_data, num_background_samples)
        
        if self.explainer is None:
            self._create_explainer(background_data)
        
        try:
            import shap
            shap_values = self.explainer.shap_values(input_data)
            
            results = {
                'shap_values': shap_values,
                'input_data': input_data,
                'background_data': background_data,
                'expected_value': getattr(self.explainer, 'expected_value', None)
            }
            
            logger.info(f"Generated SHAP explanations for {len(input_data) if len(input_data.shape) > 1 else 1} samples")
            
            return results
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanations: {str(e)}")
            raise
    
    def explain_dataset(
        self,
        dataset: np.ndarray,
        background_data: Optional[np.ndarray] = None,
        num_background_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Generate global SHAP explanations for a dataset.
        
        Args:
            dataset: Dataset to explain
            background_data: Background dataset
            num_background_samples: Number of background samples
            
        Returns:
            Dictionary containing aggregated SHAP values and summary statistics
        """
        shap_results = self.explain(dataset, background_data, num_background_samples)
        
        shap_values = shap_results['shap_values']
        
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)
        
        results = {
            'mean_abs_shap': np.mean(np.abs(shap_values), axis=0),
            'mean_shap': np.mean(shap_values, axis=0),
            'std_shap': np.std(shap_values, axis=0),
            'shap_values': shap_values,
            'expected_value': shap_results['expected_value']
        }
        
        return results
    
    def _generate_background_data(
        self,
        input_data: np.ndarray,
        num_samples: int
    ) -> np.ndarray:
        """Generate synthetic background data."""
        if len(input_data.shape) == 1:
            input_data = input_data.reshape(1, -1)
        
        mean = np.mean(input_data, axis=0)
        std = np.std(input_data, axis=0)
        
        background = np.random.normal(mean, std, size=(num_samples,) + input_data.shape[1:])
        
        return background
    
    def plot_summary(
        self,
        shap_values: np.ndarray,
        features: np.ndarray,
        feature_names: Optional[List[str]] = None,
        plot_type: str = 'bar',
        max_display: int = 20,
        output_path: Optional[str] = None
    ) -> Any:
        """
        Create SHAP summary plot.
        
        Args:
            shap_values: SHAP values to plot
            features: Feature values
            feature_names: Names of features
            plot_type: Type of plot ('bar', 'dot', 'violin')
            max_display: Maximum number of features to display
            output_path: Path to save plot
            
        Returns:
            Figure object
        """
        try:
            import shap
            import matplotlib.pyplot as plt
            
            if plot_type == 'bar':
                shap.summary_plot(
                    shap_values,
                    features,
                    feature_names=feature_names,
                    plot_type='bar',
                    max_display=max_display,
                    show=False
                )
            else:
                shap.summary_plot(
                    shap_values,
                    features,
                    feature_names=feature_names,
                    plot_type=plot_type,
                    max_display=max_display,
                    show=False
                )
            
            if output_path:
                plt.savefig(output_path, bbox_inches='tight', dpi=150)
                logger.info(f"Saved SHAP summary plot to {output_path}")
            
            return plt.gcf()
            
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return None
