# Neural Explainability Module

Comprehensive model interpretability and explainability tools for the Neural DSL framework.

## Features

- **SHAP Integration**: SHapley Additive exPlanations for feature attribution
- **LIME Integration**: Local Interpretable Model-agnostic Explanations
- **Attention Visualization**: Visualize attention weights in transformer models
- **Saliency Maps**: Gradient-based visualization methods (Vanilla, Integrated Gradients, Grad-CAM, SmoothGrad)
- **Feature Importance**: Rank features using multiple methods
- **Counterfactual Explanations**: Generate "what-if" scenarios
- **Model Cards**: Automated model documentation generation

## Installation

The explainability module requires optional dependencies:

```bash
pip install shap lime scikit-image
```

For full functionality:

```bash
pip install "neural-dsl[full]"
```

## CLI Usage

### Basic Usage

```bash
# Explain a model using SHAP
neural explain model.h5 --method shap --data input_data.npy

# Generate all explanations
neural explain model.h5 --method all --output explanations/

# Generate model card
neural explain model.h5 --generate-model-card
```

### Available Methods

- `shap`: SHAP value explanations
- `lime`: LIME explanations
- `saliency`: Gradient-based saliency maps
- `attention`: Attention weight visualization
- `feature_importance`: Feature importance ranking
- `counterfactual`: Counterfactual explanations
- `all`: Generate all explanations

### Options

- `--method, -m`: Explanation method (default: shap)
- `--backend, -b`: Model backend (tensorflow/pytorch)
- `--data, -d`: Input data file (.npy format)
- `--output, -o`: Output directory (default: explanations)
- `--num-samples`: Number of samples to explain (default: 10)
- `--generate-model-card`: Generate model card documentation

## Programmatic Usage

### SHAP Explanations

```python
from neural.explainability import ModelExplainer
import numpy as np

# Load your model
model = ...  # TensorFlow or PyTorch model

# Create explainer
explainer = ModelExplainer(
    model=model,
    backend='tensorflow',
    task_type='classification'
)

# Explain a prediction
input_sample = np.random.randn(1, 28, 28, 1)
explanation = explainer.explain_prediction(input_sample, method='shap')

# Access SHAP values
shap_values = explanation['shap_values']
```

### LIME Explanations

```python
# Generate LIME explanation
lime_explanation = explainer.explain_prediction(
    input_sample,
    method='lime',
    num_features=10
)

# Get feature weights
feature_weights = lime_explanation['feature_weights']
```

### Saliency Maps

```python
# Generate saliency map
saliency_result = explainer.explain_prediction(
    input_sample,
    method='saliency'
)

# Visualize
from neural.explainability import SaliencyMapGenerator
saliency_gen = SaliencyMapGenerator(model, 'tensorflow')
saliency_gen.visualize(
    saliency_result['saliency_map'],
    input_sample,
    output_path='saliency.png'
)
```

### Attention Visualization

```python
# Visualize attention (for transformer models)
attention_result = explainer.visualize_attention(
    input_sample,
    layer_name='attention_layer',
    tokens=['token1', 'token2', 'token3']
)

# Access attention weights
attention_weights = attention_result['attention_weights']
```

### Feature Importance

```python
# Rank features
importance_result = explainer.rank_features(
    dataset,
    labels=labels,
    method='permutation'
)

# Get rankings
top_features = importance_result['rankings'][:10]
```

### Counterfactual Explanations

```python
# Generate counterfactuals
cf_result = explainer.generate_counterfactuals(
    input_sample,
    target_class=5,
    num_samples=5,
    method='gradient'
)

# Access counterfactuals
counterfactuals = cf_result['counterfactuals']
distances = cf_result['distances']
```

### Model Card Generation

```python
from neural.explainability import ModelCardGenerator

# Create model info
model_info = {
    'model_name': 'My CNN Model',
    'version': '1.0.0',
    'framework': 'TensorFlow',
    'model_details': {
        'developed_by': 'My Team',
        'description': 'Image classification model'
    },
    'intended_use': {
        'primary_uses': ['Image classification'],
        'out_of_scope': ['Medical diagnosis']
    },
    'metrics': {
        'model_performance': {
            'accuracy': 0.95,
            'f1_score': 0.94
        }
    }
}

# Generate model card
generator = ModelCardGenerator()
generator.generate(model_info, 'model_card.md')
```

## Advanced Usage

### Custom SHAP Explainer

```python
from neural.explainability import SHAPExplainer

explainer = SHAPExplainer(
    model=model,
    backend='tensorflow',
    explainer_type='deep'  # or 'gradient', 'kernel'
)

# Generate explanations
result = explainer.explain(
    input_data,
    background_data=background_samples,
    num_background_samples=100
)

# Plot summary
explainer.plot_summary(
    result['shap_values'],
    input_data,
    feature_names=['feature1', 'feature2', ...],
    output_path='shap_summary.png'
)
```

### Multiple Saliency Methods

```python
from neural.explainability import SaliencyMapGenerator

saliency_gen = SaliencyMapGenerator(model, 'tensorflow')

# Vanilla gradients
vanilla = saliency_gen.generate(input_sample, method='vanilla')

# Integrated gradients
integrated = saliency_gen.generate(input_sample, method='integrated', num_steps=50)

# Grad-CAM (for CNNs)
gradcam = saliency_gen.generate(input_sample, method='gradcam')

# SmoothGrad
smoothgrad = saliency_gen.generate(input_sample, method='smoothgrad', num_samples=50)
```

### Global Explanations

```python
# Explain entire dataset
global_explanations = explainer.explain_dataset(
    dataset,
    labels=labels,
    num_samples=100,
    methods=['shap', 'feature_importance']
)

# Access results
shap_summary = global_explanations['shap']
feature_importance = global_explanations['feature_importance']
```

## Supported Backends

- TensorFlow/Keras
- PyTorch
- ONNX (limited support)

## Method Descriptions

### SHAP (SHapley Additive exPlanations)
Based on game theory, SHAP values represent the contribution of each feature to the prediction. Provides both local and global explanations.

### LIME (Local Interpretable Model-agnostic Explanations)
Explains individual predictions by learning an interpretable model locally around the prediction.

### Saliency Maps
- **Vanilla Gradients**: Basic gradient-based attribution
- **Integrated Gradients**: Accumulated gradients along the path from baseline
- **Grad-CAM**: Class Activation Mapping for CNNs
- **SmoothGrad**: Averaged gradients over noisy samples

### Attention Visualization
Extracts and visualizes attention weights from transformer models, showing which parts of the input the model focuses on.

### Feature Importance
Ranks features by importance using:
- Permutation importance
- Gradient-based importance
- Integrated gradient importance
- SHAP-based importance

### Counterfactual Explanations
Generates minimal changes to input that would change the model's prediction, answering "what if" questions.

## Tips and Best Practices

1. **Start with SHAP or LIME**: These provide comprehensive, interpretable explanations
2. **Use Saliency Maps for Images**: Visual explanations are intuitive for image data
3. **Combine Multiple Methods**: Different methods provide complementary insights
4. **Generate Model Cards**: Document your models for transparency and reproducibility
5. **Consider Computational Cost**: SHAP and LIME can be slow on large datasets
6. **Use Background Data**: Provide representative background data for better SHAP explanations

## Examples

See the `examples/explainability/` directory for complete working examples.

## References

- SHAP: Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions"
- LIME: Ribeiro et al. (2016). "Why Should I Trust You?"
- Grad-CAM: Selvaraju et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks"
- Integrated Gradients: Sundararajan et al. (2017). "Axiomatic Attribution for Deep Networks"

## License

This module is part of the Neural DSL project and follows the same MIT license.
