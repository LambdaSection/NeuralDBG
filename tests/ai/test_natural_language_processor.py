"""
Tests for Natural Language Processor
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from neural.ai.natural_language_processor import NaturalLanguageProcessor, IntentType, DSLGenerator


def test_create_model_intent():
    """Test creating a model from natural language."""
    nlp = NaturalLanguageProcessor()
    
    # Test basic model creation
    intent, params = nlp.extract_intent("Create a CNN for image classification")
    assert intent == IntentType.CREATE_MODEL
    assert 'name' in params
    assert 'input_shape' in params
    print("✓ Create model intent extraction works")


def test_add_layer_intent():
    """Test adding layers from natural language."""
    nlp = NaturalLanguageProcessor()
    
    # Test adding Conv2D
    intent, params = nlp.extract_intent("Add a convolutional layer with 32 filters")
    assert intent == IntentType.ADD_LAYER
    assert params['layer_type'] == 'conv2d'
    assert params['filters'] == 32
    print("✓ Add layer intent extraction works")


def test_dsl_generation():
    """Test DSL generation from intents."""
    generator = DSLGenerator()
    
    # Test model generation
    params = {
        'name': 'TestModel',
        'input_shape': (28, 28, 1),
        'num_classes': 10
    }
    dsl = generator._generate_model(params)
    assert 'network TestModel' in dsl
    assert 'input: (28, 28, 1)' in dsl
    assert 'Conv2D' in dsl
    print("✓ DSL generation works")


def test_layer_generation():
    """Test layer DSL generation."""
    generator = DSLGenerator()
    
    # Test Conv2D layer
    params = {
        'layer_type': 'conv2d',
        'filters': 32,
        'kernel_size': 3,
        'activation': 'relu'
    }
    dsl = generator._generate_layer(params)
    assert 'Conv2D(32' in dsl
    assert 'relu' in dsl
    print("✓ Layer generation works")


if __name__ == "__main__":
    print("Testing Natural Language Processor...")
    print("=" * 50)
    
    try:
        test_create_model_intent()
        test_add_layer_intent()
        test_dsl_generation()
        test_layer_generation()
        
        print("=" * 50)
        print("✓ All tests passed!")
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

