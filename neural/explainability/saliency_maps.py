"""
Gradient-based saliency map generation for neural networks.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np


logger = logging.getLogger(__name__)


class SaliencyMapGenerator:
    """
    Generate gradient-based saliency maps to understand which input features
    are most important for predictions.
    
    Supports multiple saliency methods:
    - Vanilla Gradients
    - Integrated Gradients
    - Grad-CAM (for CNNs)
    - Guided Backpropagation
    - SmoothGrad
    """
    
    def __init__(
        self,
        model: Any,
        backend: str = 'tensorflow'
    ):
        """
        Initialize saliency map generator.
        
        Args:
            model: The model to generate saliency maps for
            backend: ML framework ('tensorflow', 'pytorch')
        """
        self.model = model
        self.backend = backend.lower()
        
        logger.info(f"Initialized SaliencyMapGenerator for {backend} model")
    
    def generate(
        self,
        input_data: np.ndarray,
        target_class: Optional[int] = None,
        method: str = 'vanilla',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate saliency map for input data.
        
        Args:
            input_data: Input sample
            target_class: Target class for gradient computation
            method: Saliency method ('vanilla', 'integrated', 'gradcam', 'smoothgrad')
            **kwargs: Additional method-specific parameters
            
        Returns:
            Dictionary containing saliency maps and visualizations
        """
        if method == 'vanilla':
            saliency = self._vanilla_gradients(input_data, target_class)
        elif method == 'integrated':
            saliency = self._integrated_gradients(input_data, target_class, **kwargs)
        elif method == 'gradcam':
            saliency = self._gradcam(input_data, target_class, **kwargs)
        elif method == 'smoothgrad':
            saliency = self._smoothgrad(input_data, target_class, **kwargs)
        else:
            raise ValueError(f"Unknown saliency method: {method}")
        
        results = {
            'saliency_map': saliency,
            'input_data': input_data,
            'method': method,
            'target_class': target_class
        }
        
        logger.info(f"Generated {method} saliency map")
        
        return results
    
    def _vanilla_gradients(
        self,
        input_data: np.ndarray,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """Compute vanilla gradients."""
        if self.backend == 'tensorflow':
            return self._tf_vanilla_gradients(input_data, target_class)
        elif self.backend == 'pytorch':
            return self._torch_vanilla_gradients(input_data, target_class)
        else:
            raise ValueError(f"Backend {self.backend} not supported")
    
    def _tf_vanilla_gradients(
        self,
        input_data: np.ndarray,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """Compute vanilla gradients using TensorFlow."""
        import tensorflow as tf
        
        if len(input_data.shape) == 3:
            input_data = np.expand_dims(input_data, axis=0)
        
        input_tensor = tf.Variable(input_data, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            predictions = self.model(input_tensor)
            
            if target_class is None:
                target_class = tf.argmax(predictions[0])
            
            target_score = predictions[0, target_class]
        
        gradients = tape.gradient(target_score, input_tensor)
        
        saliency = tf.abs(gradients).numpy()
        
        return saliency[0]
    
    def _torch_vanilla_gradients(
        self,
        input_data: np.ndarray,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """Compute vanilla gradients using PyTorch."""
        import torch
        
        if len(input_data.shape) == 3:
            input_data = np.expand_dims(input_data, axis=0)
        
        input_tensor = torch.FloatTensor(input_data)
        input_tensor.requires_grad = True
        
        predictions = self.model(input_tensor)
        
        if target_class is None:
            target_class = predictions[0].argmax().item()
        
        self.model.zero_grad()
        
        target_score = predictions[0, target_class]
        target_score.backward()
        
        gradients = input_tensor.grad
        
        saliency = torch.abs(gradients).detach().cpu().numpy()
        
        return saliency[0]
    
    def _integrated_gradients(
        self,
        input_data: np.ndarray,
        target_class: Optional[int] = None,
        baseline: Optional[np.ndarray] = None,
        num_steps: int = 50
    ) -> np.ndarray:
        """
        Compute integrated gradients.
        
        Args:
            input_data: Input sample
            target_class: Target class
            baseline: Baseline for integration (defaults to zeros)
            num_steps: Number of integration steps
            
        Returns:
            Integrated gradients
        """
        if baseline is None:
            baseline = np.zeros_like(input_data)
        
        if len(input_data.shape) == 3:
            input_data = np.expand_dims(input_data, axis=0)
            baseline = np.expand_dims(baseline, axis=0)
        
        alphas = np.linspace(0, 1, num_steps)
        
        integrated_grads = np.zeros_like(input_data)
        
        for alpha in alphas:
            interpolated = baseline + alpha * (input_data - baseline)
            
            grads = self._vanilla_gradients(interpolated[0], target_class)
            
            if len(grads.shape) == 3:
                grads = np.expand_dims(grads, axis=0)
            
            integrated_grads += grads
        
        integrated_grads = integrated_grads * (input_data - baseline) / num_steps
        
        return integrated_grads[0]
    
    def _gradcam(
        self,
        input_data: np.ndarray,
        target_class: Optional[int] = None,
        layer_name: Optional[str] = None
    ) -> np.ndarray:
        """
        Compute Grad-CAM visualization.
        
        Args:
            input_data: Input sample
            target_class: Target class
            layer_name: Name of convolutional layer for Grad-CAM
            
        Returns:
            Grad-CAM heatmap
        """
        if self.backend == 'tensorflow':
            return self._tf_gradcam(input_data, target_class, layer_name)
        elif self.backend == 'pytorch':
            return self._torch_gradcam(input_data, target_class, layer_name)
        else:
            raise ValueError(f"Backend {self.backend} not supported")
    
    def _tf_gradcam(
        self,
        input_data: np.ndarray,
        target_class: Optional[int] = None,
        layer_name: Optional[str] = None
    ) -> np.ndarray:
        """Compute Grad-CAM using TensorFlow."""
        import tensorflow as tf
        
        if len(input_data.shape) == 3:
            input_data = np.expand_dims(input_data, axis=0)
        
        if layer_name is None:
            for layer in reversed(self.model.layers):
                if 'conv' in layer.name.lower():
                    layer_name = layer.name
                    break
        
        grad_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=[self.model.get_layer(layer_name).output, self.model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(input_data)
            
            if target_class is None:
                target_class = tf.argmax(predictions[0])
            
            target_score = predictions[0, target_class]
        
        grads = tape.gradient(target_score, conv_outputs)
        
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        
        heatmap = tf.nn.relu(heatmap)
        heatmap = heatmap / tf.reduce_max(heatmap)
        
        return heatmap.numpy()
    
    def _torch_gradcam(
        self,
        input_data: np.ndarray,
        target_class: Optional[int] = None,
        layer_name: Optional[str] = None
    ) -> np.ndarray:
        """Compute Grad-CAM using PyTorch."""
        import torch
        
        if len(input_data.shape) == 3:
            input_data = np.expand_dims(input_data, axis=0)
        
        input_tensor = torch.FloatTensor(input_data)
        input_tensor.requires_grad = True
        
        activations = []
        gradients = []
        
        def forward_hook(module, input, output):
            activations.append(output)
        
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])
        
        if layer_name is None:
            for name, module in self.model.named_modules():
                if 'conv' in name.lower():
                    target_layer = module
        else:
            target_layer = dict(self.model.named_modules())[layer_name]
        
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)
        
        predictions = self.model(input_tensor)
        
        if target_class is None:
            target_class = predictions[0].argmax().item()
        
        self.model.zero_grad()
        target_score = predictions[0, target_class]
        target_score.backward()
        
        forward_handle.remove()
        backward_handle.remove()
        
        activation = activations[0]
        gradient = gradients[0]
        
        pooled_grads = torch.mean(gradient, dim=(0, 2, 3))
        
        for i in range(activation.shape[1]):
            activation[0, i] *= pooled_grads[i]
        
        heatmap = torch.mean(activation, dim=1).squeeze()
        heatmap = torch.relu(heatmap)
        heatmap = heatmap / torch.max(heatmap)
        
        return heatmap.detach().cpu().numpy()
    
    def _smoothgrad(
        self,
        input_data: np.ndarray,
        target_class: Optional[int] = None,
        num_samples: int = 50,
        noise_level: float = 0.15
    ) -> np.ndarray:
        """
        Compute SmoothGrad by averaging gradients over noisy samples.
        
        Args:
            input_data: Input sample
            target_class: Target class
            num_samples: Number of noisy samples
            noise_level: Standard deviation of noise
            
        Returns:
            Smoothed gradients
        """
        smoothed_grads = np.zeros_like(input_data)
        
        stdev = noise_level * (np.max(input_data) - np.min(input_data))
        
        for _ in range(num_samples):
            noise = np.random.normal(0, stdev, input_data.shape)
            noisy_input = input_data + noise
            
            grads = self._vanilla_gradients(noisy_input, target_class)
            
            smoothed_grads += grads
        
        smoothed_grads = smoothed_grads / num_samples
        
        return smoothed_grads
    
    def visualize(
        self,
        saliency_map: np.ndarray,
        input_data: np.ndarray,
        overlay: bool = True,
        colormap: str = 'jet',
        output_path: Optional[str] = None
    ) -> Any:
        """
        Visualize saliency map.
        
        Args:
            saliency_map: Generated saliency map
            input_data: Original input data
            overlay: Whether to overlay saliency on input
            colormap: Matplotlib colormap to use
            output_path: Path to save visualization
            
        Returns:
            Figure object
        """
        try:
            from matplotlib import cm
            import matplotlib.pyplot as plt
            
            saliency_normalized = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
            
            if len(saliency_map.shape) == 3:
                saliency_normalized = np.mean(saliency_normalized, axis=-1)
            
            fig, axes = plt.subplots(1, 3 if overlay else 2, figsize=(15 if overlay else 10, 5))
            
            if len(input_data.shape) == 3 and input_data.shape[-1] in [1, 3]:
                axes[0].imshow(input_data.squeeze(), cmap='gray' if input_data.shape[-1] == 1 else None)
            else:
                axes[0].imshow(input_data, cmap='gray')
            axes[0].set_title('Input')
            axes[0].axis('off')
            
            im = axes[1].imshow(saliency_normalized, cmap=colormap)
            axes[1].set_title('Saliency Map')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1])
            
            if overlay:
                if len(input_data.shape) == 3 and input_data.shape[-1] in [1, 3]:
                    axes[2].imshow(input_data.squeeze(), cmap='gray' if input_data.shape[-1] == 1 else None, alpha=0.6)
                else:
                    axes[2].imshow(input_data, cmap='gray', alpha=0.6)
                axes[2].imshow(saliency_normalized, cmap=colormap, alpha=0.4)
                axes[2].set_title('Overlay')
                axes[2].axis('off')
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, bbox_inches='tight', dpi=150)
                logger.info(f"Saved saliency visualization to {output_path}")
            
            return fig
            
        except ImportError:
            logger.warning("matplotlib not available for visualization")
            return None
