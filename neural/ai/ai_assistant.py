"""
AI Assistant for Neural DSL

Main interface for AI-powered features including natural language to DSL conversion.
"""

from typing import Optional, Dict, List
from .natural_language_processor import NaturalLanguageProcessor, IntentType, DSLGenerator
from .llm_integration import LLMIntegration
from .multi_language import MultiLanguageSupport


class NeuralAIAssistant:
    """
    AI Assistant for Neural DSL.
    
    Provides natural language to DSL conversion with multi-language support.
    """
    
    def __init__(self, use_llm: bool = True, llm_provider: Optional[str] = None):
        """
        Initialize AI Assistant.
        
        Args:
            use_llm: Whether to use LLM for advanced processing (default: True)
            llm_provider: Specific LLM provider to use ('openai', 'anthropic', 'ollama', or None for auto)
        """
        self.nlp = NaturalLanguageProcessor()
        self.dsl_generator = DSLGenerator()
        self.multi_lang = MultiLanguageSupport()
        
        self.use_llm = use_llm
        self.llm = None
        if use_llm:
            try:
                self.llm = LLMIntegration(provider=llm_provider)
            except Exception as e:
                print(f"Warning: LLM not available: {e}. Using rule-based processing.")
                self.use_llm = False
    
    def chat(self, user_input: str, context: Optional[Dict] = None) -> Dict[str, any]:
        """
        Process user input and generate response.
        
        Args:
            user_input: User's natural language input
            context: Optional context (current model state, conversation history, etc.)
            
        Returns:
            Dictionary with:
            - response: Text response to user
            - dsl_code: Generated DSL code (if any)
            - intent: Detected intent
            - success: Whether operation was successful
        """
        # Detect and translate language if needed
        lang_result = self.multi_lang.process(user_input, target_lang='en')
        processed_text = lang_result['final']
        
        # Try LLM-based generation if available
        if self.use_llm and self.llm and self.llm.is_available():
            try:
                dsl_code = self.llm.generate_dsl(processed_text, context)
                return {
                    'response': f"I've generated a Neural DSL model for you:\n\n{dsl_code}",
                    'dsl_code': dsl_code,
                    'intent': 'generate_model',
                    'success': True,
                    'language': lang_result['detected_lang']
                }
            except Exception as e:
                # Fallback to rule-based if LLM fails
                print(f"LLM generation failed: {e}. Falling back to rule-based processing.")
        
        # Rule-based processing
        intent_type, params = self.nlp.extract_intent(processed_text)
        
        if intent_type == IntentType.UNKNOWN:
            return {
                'response': "I didn't understand that. Try: 'Create a CNN for image classification' or 'Add a dense layer with 128 units'",
                'dsl_code': None,
                'intent': 'unknown',
                'success': False,
                'language': lang_result['detected_lang']
            }
        
        # Generate DSL based on intent
        dsl_snippet = self.dsl_generator.generate_from_intent(intent_type, params)
        
        # Update context
        if intent_type == IntentType.CREATE_MODEL:
            self.dsl_generator.current_model.update(params)
            full_dsl = self.dsl_generator.get_full_model()
            return {
                'response': f"Created model '{params.get('name', 'MyModel')}'. Here's the DSL:\n\n{full_dsl}",
                'dsl_code': full_dsl,
                'intent': 'create_model',
                'success': True,
                'language': lang_result['detected_lang']
            }
        
        elif intent_type == IntentType.ADD_LAYER:
            # Add layer to current model
            layer_dsl = dsl_snippet
            self.dsl_generator.current_model['layers'].append(layer_dsl.strip())
            return {
                'response': f"Added layer: {layer_dsl.strip()}",
                'dsl_code': layer_dsl,
                'intent': 'add_layer',
                'success': True,
                'language': lang_result['detected_lang']
            }
        
        else:
            return {
                'response': f"Processed: {intent_type.value}",
                'dsl_code': dsl_snippet,
                'intent': intent_type.value,
                'success': True,
                'language': lang_result['detected_lang']
            }
    
    def get_current_model(self) -> str:
        """Get the current model as DSL."""
        return self.dsl_generator.get_full_model()
    
    def reset(self):
        """Reset the assistant state."""
        self.dsl_generator = DSLGenerator()

