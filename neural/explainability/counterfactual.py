"""
Counterfactual explanation generation for neural networks.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class CounterfactualGenerator:
    """
    Generate counterfactual explanations.
    
    Counterfactuals answer "what if" questions by finding minimal changes
    to the input that would change the model's prediction.
    """
    
    def __init__(
        self,
        model: Any,
        backend: str = 'tensorflow',
        task_type: str = 'classification'
    ):
        """
        Initialize counterfactual generator.
        
        Args:
            model: The model to generate counterfactuals for
            backend: ML framework ('tensorflow', 'pytorch', 'onnx')
            task_type: Type of task ('classification', 'regression')
        """
        self.model = model
        self.backend = backend.lower()
        self.task_type = task_type
        
        logger.info(f"Initialized CounterfactualGenerator for {backend} model")
    
    def generate(
        self,
        input_data: np.ndarray,
        target_class: Optional[int] = None,
        num_samples: int = 5,
        method: str = 'gradient',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate counterfactual explanations.
        
        Args:
            input_data: Input sample to generate counterfactuals for
            target_class: Target class (if None, uses opposite of current prediction)
            num_samples: Number of counterfactual samples to generate
            method: Generation method ('gradient', 'genetic', 'random')
            **kwargs: Additional method-specific parameters
            
        Returns:
            Dictionary containing counterfactual samples and explanations
        """
        if method == 'gradient':
            counterfactuals = self._gradient_based_counterfactuals(
                input_data, target_class, num_samples, **kwargs
            )
        elif method == 'genetic':
            counterfactuals = self._genetic_counterfactuals(
                input_data, target_class, num_samples, **kwargs
            )
        elif method == 'random':
            counterfactuals = self._random_search_counterfactuals(
                input_data, target_class, num_samples, **kwargs
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        results = {
            'counterfactuals': counterfactuals,
            'original_input': input_data,
            'target_class': target_class,
            'method': method,
            'distances': self._compute_distances(input_data, counterfactuals)
        }
        
        logger.info(f"Generated {len(counterfactuals)} counterfactuals using {method}")
        
        return results
    
    def _get_prediction(self, input_data: np.ndarray) -> np.ndarray:
        """Get model prediction."""
        if self.backend == 'tensorflow':
            return self.model(input_data).numpy()
        elif self.backend == 'pytorch':
            import torch
            if not isinstance(input_data, torch.Tensor):
                input_data = torch.FloatTensor(input_data)
            with torch.no_grad():
                return self.model(input_data).numpy()
        else:
            return self.model.predict(input_data)
    
    def _gradient_based_counterfactuals(
        self,
        input_data: np.ndarray,
        target_class: Optional[int] = None,
        num_samples: int = 5,
        learning_rate: float = 0.1,
        max_iterations: int = 1000,
        distance_weight: float = 0.1
    ) -> List[np.ndarray]:
        """
        Generate counterfactuals using gradient-based optimization.
        
        Args:
            input_data: Input sample
            target_class: Target class
            num_samples: Number of counterfactuals to generate
            learning_rate: Optimization learning rate
            max_iterations: Maximum optimization iterations
            distance_weight: Weight for distance penalty
            
        Returns:
            List of counterfactual samples
        """
        if len(input_data.shape) == 1:
            input_data = input_data.reshape(1, -1)
        elif len(input_data.shape) == 3:
            input_data = np.expand_dims(input_data, axis=0)
        
        current_pred = self._get_prediction(input_data)
        current_class = np.argmax(current_pred[0])
        
        if target_class is None:
            num_classes = current_pred.shape[-1]
            target_class = (current_class + 1) % num_classes
        
        counterfactuals = []
        
        if self.backend == 'tensorflow':
            counterfactuals = self._tf_gradient_counterfactuals(
                input_data, target_class, num_samples, learning_rate, max_iterations, distance_weight
            )
        elif self.backend == 'pytorch':
            counterfactuals = self._torch_gradient_counterfactuals(
                input_data, target_class, num_samples, learning_rate, max_iterations, distance_weight
            )
        
        return counterfactuals
    
    def _tf_gradient_counterfactuals(
        self,
        input_data: np.ndarray,
        target_class: int,
        num_samples: int,
        learning_rate: float,
        max_iterations: int,
        distance_weight: float
    ) -> List[np.ndarray]:
        """Generate counterfactuals using TensorFlow."""
        import tensorflow as tf
        
        counterfactuals = []
        
        for _ in range(num_samples):
            noise = np.random.normal(0, 0.01, input_data.shape)
            perturbed = tf.Variable(input_data + noise, dtype=tf.float32)
            
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            
            for iteration in range(max_iterations):
                with tf.GradientTape() as tape:
                    predictions = self.model(perturbed)
                    
                    target_loss = -predictions[0, target_class]
                    
                    distance_loss = tf.reduce_mean(tf.square(perturbed - input_data))
                    
                    total_loss = target_loss + distance_weight * distance_loss
                
                gradients = tape.gradient(total_loss, perturbed)
                optimizer.apply_gradients([(gradients, perturbed)])
                
                if iteration % 100 == 0:
                    pred_class = tf.argmax(predictions[0]).numpy()
                    if pred_class == target_class:
                        break
            
            counterfactuals.append(perturbed.numpy()[0])
        
        return counterfactuals
    
    def _torch_gradient_counterfactuals(
        self,
        input_data: np.ndarray,
        target_class: int,
        num_samples: int,
        learning_rate: float,
        max_iterations: int,
        distance_weight: float
    ) -> List[np.ndarray]:
        """Generate counterfactuals using PyTorch."""
        import torch
        import torch.optim as optim
        
        counterfactuals = []
        
        for _ in range(num_samples):
            noise = np.random.normal(0, 0.01, input_data.shape)
            perturbed = torch.FloatTensor(input_data + noise)
            perturbed.requires_grad = True
            
            optimizer = optim.Adam([perturbed], lr=learning_rate)
            
            for iteration in range(max_iterations):
                optimizer.zero_grad()
                
                predictions = self.model(perturbed)
                
                target_loss = -predictions[0, target_class]
                
                distance_loss = torch.mean((perturbed - torch.FloatTensor(input_data)) ** 2)
                
                total_loss = target_loss + distance_weight * distance_loss
                
                total_loss.backward()
                optimizer.step()
                
                if iteration % 100 == 0:
                    pred_class = predictions[0].argmax().item()
                    if pred_class == target_class:
                        break
            
            counterfactuals.append(perturbed.detach().cpu().numpy()[0])
        
        return counterfactuals
    
    def _genetic_counterfactuals(
        self,
        input_data: np.ndarray,
        target_class: Optional[int] = None,
        num_samples: int = 5,
        population_size: int = 50,
        num_generations: int = 100,
        mutation_rate: float = 0.1
    ) -> List[np.ndarray]:
        """
        Generate counterfactuals using genetic algorithm.
        
        Args:
            input_data: Input sample
            target_class: Target class
            num_samples: Number of counterfactuals to generate
            population_size: Size of genetic algorithm population
            num_generations: Number of generations to evolve
            mutation_rate: Probability of mutation
            
        Returns:
            List of counterfactual samples
        """
        if len(input_data.shape) == 1:
            input_data = input_data.reshape(1, -1)
        
        current_pred = self._get_prediction(input_data)
        current_class = np.argmax(current_pred[0])
        
        if target_class is None:
            num_classes = current_pred.shape[-1]
            target_class = (current_class + 1) % num_classes
        
        population = [input_data[0] + np.random.normal(0, 0.1, input_data[0].shape) for _ in range(population_size)]
        
        for generation in range(num_generations):
            fitness_scores = []
            for individual in population:
                pred = self._get_prediction(individual.reshape(1, -1))
                target_score = pred[0, target_class]
                distance = np.mean((individual - input_data[0]) ** 2)
                fitness = target_score - 0.1 * distance
                fitness_scores.append(fitness)
            
            sorted_indices = np.argsort(fitness_scores)[::-1]
            population = [population[i] for i in sorted_indices[:population_size // 2]]
            
            while len(population) < population_size:
                parent1 = population[np.random.randint(len(population))]
                parent2 = population[np.random.randint(len(population))]
                
                crossover_point = np.random.randint(1, len(parent1.flatten()))
                child = parent1.copy()
                child.flat[crossover_point:] = parent2.flat[crossover_point:]
                
                if np.random.random() < mutation_rate:
                    mutation = np.random.normal(0, 0.01, child.shape)
                    child += mutation
                
                population.append(child)
        
        fitness_scores = []
        for individual in population:
            pred = self._get_prediction(individual.reshape(1, -1))
            target_score = pred[0, target_class]
            distance = np.mean((individual - input_data[0]) ** 2)
            fitness = target_score - 0.1 * distance
            fitness_scores.append(fitness)
        
        sorted_indices = np.argsort(fitness_scores)[::-1]
        counterfactuals = [population[i] for i in sorted_indices[:num_samples]]
        
        return counterfactuals
    
    def _random_search_counterfactuals(
        self,
        input_data: np.ndarray,
        target_class: Optional[int] = None,
        num_samples: int = 5,
        num_trials: int = 1000,
        noise_scale: float = 0.1
    ) -> List[np.ndarray]:
        """
        Generate counterfactuals using random search.
        
        Args:
            input_data: Input sample
            target_class: Target class
            num_samples: Number of counterfactuals to generate
            num_trials: Number of random trials
            noise_scale: Scale of random noise
            
        Returns:
            List of counterfactual samples
        """
        if len(input_data.shape) == 1:
            input_data = input_data.reshape(1, -1)
        
        current_pred = self._get_prediction(input_data)
        current_class = np.argmax(current_pred[0])
        
        if target_class is None:
            num_classes = current_pred.shape[-1]
            target_class = (current_class + 1) % num_classes
        
        candidates = []
        
        for _ in range(num_trials):
            noise = np.random.normal(0, noise_scale, input_data[0].shape)
            candidate = input_data[0] + noise
            
            pred = self._get_prediction(candidate.reshape(1, -1))
            pred_class = np.argmax(pred[0])
            
            if pred_class == target_class:
                distance = np.linalg.norm(candidate - input_data[0])
                candidates.append((candidate, distance))
        
        if not candidates:
            logger.warning("No valid counterfactuals found in random search")
            return [input_data[0] + np.random.normal(0, noise_scale, input_data[0].shape) for _ in range(num_samples)]
        
        candidates.sort(key=lambda x: x[1])
        counterfactuals = [c[0] for c in candidates[:num_samples]]
        
        return counterfactuals
    
    def _compute_distances(
        self,
        original: np.ndarray,
        counterfactuals: List[np.ndarray]
    ) -> List[float]:
        """Compute distances between original and counterfactuals."""
        distances = []
        for cf in counterfactuals:
            distance = np.linalg.norm(cf - original)
            distances.append(float(distance))
        return distances
    
    def visualize_counterfactuals(
        self,
        original: np.ndarray,
        counterfactuals: List[np.ndarray],
        output_path: Optional[str] = None
    ) -> Any:
        """
        Visualize counterfactual explanations.
        
        Args:
            original: Original input
            counterfactuals: Generated counterfactuals
            output_path: Path to save visualization
            
        Returns:
            Figure object
        """
        try:
            import matplotlib.pyplot as plt
            
            num_cf = len(counterfactuals)
            cols = min(3, num_cf + 1)
            rows = (num_cf + 1 + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
            if rows == 1 and cols == 1:
                axes = np.array([[axes]])
            elif rows == 1 or cols == 1:
                axes = axes.reshape(rows, cols)
            axes = axes.flatten()
            
            if len(original.shape) == 3:
                axes[0].imshow(original, cmap='gray' if original.shape[-1] == 1 else None)
            elif len(original.shape) == 2:
                axes[0].imshow(original, cmap='gray')
            else:
                axes[0].bar(range(len(original)), original)
            axes[0].set_title('Original')
            axes[0].axis('off' if len(original.shape) >= 2 else 'on')
            
            for idx, cf in enumerate(counterfactuals):
                if len(cf.shape) == 3:
                    axes[idx + 1].imshow(cf, cmap='gray' if cf.shape[-1] == 1 else None)
                elif len(cf.shape) == 2:
                    axes[idx + 1].imshow(cf, cmap='gray')
                else:
                    axes[idx + 1].bar(range(len(cf)), cf)
                
                distance = np.linalg.norm(cf - original)
                axes[idx + 1].set_title(f'CF {idx + 1} (d={distance:.3f})')
                axes[idx + 1].axis('off' if len(cf.shape) >= 2 else 'on')
            
            for idx in range(num_cf + 1, len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, bbox_inches='tight', dpi=150)
                logger.info(f"Saved counterfactual visualization to {output_path}")
            
            return fig
            
        except ImportError:
            logger.warning("matplotlib not available for visualization")
            return None
