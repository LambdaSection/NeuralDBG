"""
Model card generation for model documentation and transparency.
"""

from datetime import datetime
import logging
from typing import Any, Dict


logger = logging.getLogger(__name__)


class ModelCardGenerator:
    """
    Generate comprehensive model cards for documentation.
    
    Model cards provide standardized documentation for machine learning models,
    including model details, intended use, performance metrics, limitations,
    and ethical considerations.
    """
    
    def __init__(self):
        """Initialize model card generator."""
        logger.info("Initialized ModelCardGenerator")
    
    def generate(
        self,
        model_info: Dict[str, Any],
        output_path: str = 'model_card.md'
    ) -> str:
        """
        Generate a model card.
        
        Args:
            model_info: Dictionary containing model information
            output_path: Path to save the model card
            
        Returns:
            Path to generated model card
        """
        card_content = self._create_card_content(model_info)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(card_content)
        
        logger.info(f"Generated model card at {output_path}")
        
        return output_path
    
    def _create_card_content(self, model_info: Dict[str, Any]) -> str:
        """Create the model card content."""
        sections = []
        
        sections.append(self._create_header(model_info))
        sections.append(self._create_model_details(model_info))
        sections.append(self._create_intended_use(model_info))
        sections.append(self._create_factors(model_info))
        sections.append(self._create_metrics(model_info))
        sections.append(self._create_training_data(model_info))
        sections.append(self._create_evaluation_data(model_info))
        sections.append(self._create_ethical_considerations(model_info))
        sections.append(self._create_caveats_and_recommendations(model_info))
        
        return '\n\n'.join(sections)
    
    def _create_header(self, model_info: Dict[str, Any]) -> str:
        """Create the header section."""
        model_name = model_info.get('model_name', 'Unnamed Model')
        version = model_info.get('version', '1.0.0')
        date = model_info.get('date', datetime.now().strftime('%Y-%m-%d'))
        
        return f"""# Model Card: {model_name}

**Version:** {version}  
**Date:** {date}  
**Model Type:** {model_info.get('model_type', 'Neural Network')}  
**Framework:** {model_info.get('framework', 'Unknown')}
"""
    
    def _create_model_details(self, model_info: Dict[str, Any]) -> str:
        """Create the model details section."""
        details = model_info.get('model_details', {})
        
        content = "## Model Details\n\n"
        content += f"**Developed by:** {details.get('developed_by', 'Not specified')}\n\n"
        content += f"**Model date:** {details.get('model_date', 'Not specified')}\n\n"
        content += f"**Model version:** {details.get('model_version', '1.0.0')}\n\n"
        content += f"**Model type:** {details.get('model_type', 'Not specified')}\n\n"
        
        if 'description' in details:
            content += f"\n**Description:**\n\n{details['description']}\n\n"
        
        if 'architecture' in details:
            content += "**Architecture:**\n\n"
            arch = details['architecture']
            if isinstance(arch, dict):
                content += f"- **Input shape:** {arch.get('input_shape', 'Not specified')}\n"
                content += f"- **Output shape:** {arch.get('output_shape', 'Not specified')}\n"
                content += f"- **Number of layers:** {arch.get('num_layers', 'Not specified')}\n"
                content += f"- **Total parameters:** {arch.get('total_params', 'Not specified')}\n"
                content += f"- **Trainable parameters:** {arch.get('trainable_params', 'Not specified')}\n"
            else:
                content += str(arch) + "\n"
            content += "\n"
        
        if 'paper' in details:
            content += f"**Paper or other resource for more information:** {details['paper']}\n\n"
        
        if 'citation' in details:
            content += f"**Citation details:** {details['citation']}\n\n"
        
        if 'license' in details:
            content += f"**License:** {details['license']}\n\n"
        
        if 'contact' in details:
            content += f"**Contact:** {details['contact']}\n\n"
        
        return content
    
    def _create_intended_use(self, model_info: Dict[str, Any]) -> str:
        """Create the intended use section."""
        intended_use = model_info.get('intended_use', {})
        
        content = "## Intended Use\n\n"
        
        if 'primary_uses' in intended_use:
            content += "**Primary intended uses:**\n\n"
            for use in intended_use['primary_uses']:
                content += f"- {use}\n"
            content += "\n"
        
        if 'primary_users' in intended_use:
            content += "**Primary intended users:**\n\n"
            for user in intended_use['primary_users']:
                content += f"- {user}\n"
            content += "\n"
        
        if 'out_of_scope' in intended_use:
            content += "**Out-of-scope use cases:**\n\n"
            for case in intended_use['out_of_scope']:
                content += f"- {case}\n"
            content += "\n"
        
        return content
    
    def _create_factors(self, model_info: Dict[str, Any]) -> str:
        """Create the factors section."""
        factors = model_info.get('factors', {})
        
        if not factors:
            return ""
        
        content = "## Factors\n\n"
        
        if 'relevant_factors' in factors:
            content += "**Relevant factors:**\n\n"
            for factor in factors['relevant_factors']:
                content += f"- {factor}\n"
            content += "\n"
        
        if 'evaluation_factors' in factors:
            content += "**Evaluation factors:**\n\n"
            for factor in factors['evaluation_factors']:
                content += f"- {factor}\n"
            content += "\n"
        
        return content
    
    def _create_metrics(self, model_info: Dict[str, Any]) -> str:
        """Create the metrics section."""
        metrics = model_info.get('metrics', {})
        
        content = "## Metrics\n\n"
        
        if 'model_performance' in metrics:
            content += "**Model performance measures:**\n\n"
            for metric_name, metric_value in metrics['model_performance'].items():
                if isinstance(metric_value, float):
                    content += f"- **{metric_name}:** {metric_value:.4f}\n"
                else:
                    content += f"- **{metric_name}:** {metric_value}\n"
            content += "\n"
        
        if 'decision_thresholds' in metrics:
            content += "**Decision thresholds:**\n\n"
            content += f"{metrics['decision_thresholds']}\n\n"
        
        if 'variation_approaches' in metrics:
            content += "**Variation approaches:**\n\n"
            content += f"{metrics['variation_approaches']}\n\n"
        
        return content
    
    def _create_training_data(self, model_info: Dict[str, Any]) -> str:
        """Create the training data section."""
        training_data = model_info.get('training_data', {})
        
        content = "## Training Data\n\n"
        
        if 'dataset' in training_data:
            content += f"**Dataset:** {training_data['dataset']}\n\n"
        
        if 'motivation' in training_data:
            content += f"**Motivation:** {training_data['motivation']}\n\n"
        
        if 'preprocessing' in training_data:
            content += "**Preprocessing:**\n\n"
            for step in training_data['preprocessing']:
                content += f"- {step}\n"
            content += "\n"
        
        if 'size' in training_data:
            content += f"**Size:** {training_data['size']}\n\n"
        
        if 'splits' in training_data:
            content += "**Data splits:**\n\n"
            for split, size in training_data['splits'].items():
                content += f"- **{split}:** {size}\n"
            content += "\n"
        
        return content
    
    def _create_evaluation_data(self, model_info: Dict[str, Any]) -> str:
        """Create the evaluation data section."""
        evaluation_data = model_info.get('evaluation_data', {})
        
        if not evaluation_data:
            return ""
        
        content = "## Evaluation Data\n\n"
        
        if 'dataset' in evaluation_data:
            content += f"**Dataset:** {evaluation_data['dataset']}\n\n"
        
        if 'motivation' in evaluation_data:
            content += f"**Motivation:** {evaluation_data['motivation']}\n\n"
        
        if 'preprocessing' in evaluation_data:
            content += "**Preprocessing:**\n\n"
            for step in evaluation_data['preprocessing']:
                content += f"- {step}\n"
            content += "\n"
        
        return content
    
    def _create_ethical_considerations(self, model_info: Dict[str, Any]) -> str:
        """Create the ethical considerations section."""
        ethical = model_info.get('ethical_considerations', {})
        
        if not ethical:
            return ""
        
        content = "## Ethical Considerations\n\n"
        
        if 'sensitive_data' in ethical:
            content += "**Sensitive data:**\n\n"
            content += f"{ethical['sensitive_data']}\n\n"
        
        if 'human_life' in ethical:
            content += "**Human life:**\n\n"
            content += f"{ethical['human_life']}\n\n"
        
        if 'mitigations' in ethical:
            content += "**Mitigations:**\n\n"
            for mitigation in ethical['mitigations']:
                content += f"- {mitigation}\n"
            content += "\n"
        
        if 'risks_and_harms' in ethical:
            content += "**Risks and harms:**\n\n"
            for risk in ethical['risks_and_harms']:
                content += f"- {risk}\n"
            content += "\n"
        
        if 'use_cases' in ethical:
            content += "**Use cases:**\n\n"
            for case in ethical['use_cases']:
                content += f"- {case}\n"
            content += "\n"
        
        return content
    
    def _create_caveats_and_recommendations(self, model_info: Dict[str, Any]) -> str:
        """Create the caveats and recommendations section."""
        caveats = model_info.get('caveats_and_recommendations', {})
        
        if not caveats:
            return ""
        
        content = "## Caveats and Recommendations\n\n"
        
        if 'limitations' in caveats:
            content += "**Known limitations:**\n\n"
            for limitation in caveats['limitations']:
                content += f"- {limitation}\n"
            content += "\n"
        
        if 'recommendations' in caveats:
            content += "**Recommendations:**\n\n"
            for recommendation in caveats['recommendations']:
                content += f"- {recommendation}\n"
            content += "\n"
        
        if 'additional_information' in caveats:
            content += "**Additional information:**\n\n"
            content += f"{caveats['additional_information']}\n\n"
        
        return content
    
    def create_example_model_info(self) -> Dict[str, Any]:
        """Create an example model_info dictionary."""
        return {
            'model_name': 'Example Neural Network',
            'version': '1.0.0',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'model_type': 'Convolutional Neural Network',
            'framework': 'TensorFlow',
            'model_details': {
                'developed_by': 'Neural DSL Team',
                'model_date': datetime.now().strftime('%Y-%m-%d'),
                'model_version': '1.0.0',
                'model_type': 'Image Classification CNN',
                'description': 'A convolutional neural network for image classification tasks.',
                'architecture': {
                    'input_shape': '(224, 224, 3)',
                    'output_shape': '(10,)',
                    'num_layers': 15,
                    'total_params': '25,000,000',
                    'trainable_params': '24,500,000'
                },
                'license': 'MIT',
                'contact': 'contact@example.com'
            },
            'intended_use': {
                'primary_uses': [
                    'Image classification for research purposes',
                    'Transfer learning base model'
                ],
                'primary_users': [
                    'Machine learning researchers',
                    'Data scientists'
                ],
                'out_of_scope': [
                    'Medical diagnosis without expert validation',
                    'Autonomous vehicle decision-making'
                ]
            },
            'metrics': {
                'model_performance': {
                    'accuracy': 0.92,
                    'precision': 0.91,
                    'recall': 0.90,
                    'f1_score': 0.905
                }
            },
            'training_data': {
                'dataset': 'Custom Image Dataset',
                'motivation': 'Diverse image dataset for general classification',
                'preprocessing': [
                    'Resized to 224x224 pixels',
                    'Normalized to [0, 1] range',
                    'Data augmentation applied'
                ],
                'size': '100,000 images',
                'splits': {
                    'train': '80,000',
                    'validation': '10,000',
                    'test': '10,000'
                }
            },
            'ethical_considerations': {
                'mitigations': [
                    'Balanced dataset across classes',
                    'Regular bias audits performed'
                ],
                'risks_and_harms': [
                    'Potential misclassification in edge cases',
                    'May perform poorly on out-of-distribution data'
                ]
            },
            'caveats_and_recommendations': {
                'limitations': [
                    'Performance may degrade on low-resolution images',
                    'Not tested on all possible image domains'
                ],
                'recommendations': [
                    'Validate on domain-specific data before deployment',
                    'Monitor performance regularly in production'
                ]
            }
        }
