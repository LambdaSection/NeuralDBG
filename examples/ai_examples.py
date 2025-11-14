"""
Examples of using Neural AI Assistant

This script demonstrates how to use the AI-powered features of Neural DSL.
"""

from neural.ai.ai_assistant import NeuralAIAssistant


def example_basic_usage():
    """Example 1: Basic rule-based natural language to DSL."""
    print("=" * 70)
    print("Example 1: Basic Natural Language to DSL (Rule-Based)")
    print("=" * 70)
    
    assistant = NeuralAIAssistant(use_llm=False)
    
    # Create a model
    result = assistant.chat("Create a CNN for MNIST classification")
    print("\nUser: Create a CNN for MNIST classification")
    print(f"\nResponse:\n{result['response']}")
    print(f"\nGenerated DSL:\n{result['dsl_code']}")
    print(f"Success: {result['success']}")
    print(f"Detected Language: {result.get('language', 'en')}")


def example_add_layers():
    """Example 2: Incrementally building a model."""
    print("\n" + "=" * 70)
    print("Example 2: Incrementally Building a Model")
    print("=" * 70)
    
    assistant = NeuralAIAssistant(use_llm=False)
    
    # Start with a model
    result1 = assistant.chat("Create a model named MyCNN")
    print("\nUser: Create a model named MyCNN")
    print(f"Response: {result1['response']}")
    
    # Add layers
    result2 = assistant.chat("Add a convolutional layer with 32 filters")
    print("\nUser: Add a convolutional layer with 32 filters")
    print(f"Response: {result2['response']}")
    
    result3 = assistant.chat("Add max pooling")
    print("\nUser: Add max pooling")
    print(f"Response: {result3['response']}")
    
    result4 = assistant.chat("Add a dense layer with 128 units")
    print("\nUser: Add a dense layer with 128 units")
    print(f"Response: {result4['response']}")
    
    # Get final model
    final_model = assistant.get_current_model()
    print("\n" + "=" * 70)
    print("Final Model:")
    print("=" * 70)
    print(final_model)


def example_different_commands():
    """Example 3: Various natural language commands."""
    print("\n" + "=" * 70)
    print("Example 3: Various Natural Language Commands")
    print("=" * 70)
    
    assistant = NeuralAIAssistant(use_llm=False)
    
    commands = [
        "Create a CNN for image classification",
        "Add dropout with rate 0.5",
        "Add a dense layer with 64 units and relu activation",
        "Change optimizer to Adam with learning rate 0.001"
    ]
    
    for cmd in commands:
        result = assistant.chat(cmd)
        print(f"\nUser: {cmd}")
        print(f"Response: {result['response']}")
        if result['dsl_code']:
            print(f"Generated: {result['dsl_code'].strip()}")


def example_with_llm():
    """Example 4: Using LLM for advanced generation (if available)."""
    print("\n" + "=" * 70)
    print("Example 4: Using LLM for Advanced Generation")
    print("=" * 70)
    
    try:
        assistant = NeuralAIAssistant(use_llm=True)
        
        if assistant.llm and assistant.llm.is_available():
            print("✓ LLM available!")
            result = assistant.chat("Create a ResNet50 model for ImageNet classification")
            print(f"\nUser: Create a ResNet50 model for ImageNet classification")
            print(f"\nResponse:\n{result['response']}")
        else:
            print("⚠ LLM not available. Install and configure a provider:")
            print("  - OpenAI: pip install openai && set OPENAI_API_KEY")
            print("  - Anthropic: pip install anthropic && set ANTHROPIC_API_KEY")
            print("  - Ollama: Install from https://ollama.ai")
    except Exception as e:
        print(f"⚠ LLM not available: {e}")
        print("Falling back to rule-based processing...")


def example_multi_language():
    """Example 5: Multi-language support (if available)."""
    print("\n" + "=" * 70)
    print("Example 5: Multi-Language Support")
    print("=" * 70)
    
    assistant = NeuralAIAssistant(use_llm=False)
    
    # Test with different languages (will be translated to English)
    languages = {
        'English': "Create a CNN for image classification",
        'French': "Créer un CNN pour la classification d'images",
        'Spanish': "Crear una CNN para clasificación de imágenes"
    }
    
    for lang_name, text in languages.items():
        result = assistant.chat(text)
        print(f"\n{lang_name}: {text}")
        print(f"Detected Language: {result.get('language', 'unknown')}")
        print(f"Response: {result['response'][:100]}...")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Neural AI Assistant Examples")
    print("=" * 70)
    
    # Run examples
    example_basic_usage()
    example_add_layers()
    example_different_commands()
    
    # Optional examples (require additional setup)
    try:
        example_with_llm()
    except Exception as e:
        print(f"\n⚠ LLM example skipped: {e}")
    
    try:
        example_multi_language()
    except Exception as e:
        print(f"\n⚠ Multi-language example skipped: {e}")
    
    print("\n" + "=" * 70)
    print("Examples Complete!")
    print("=" * 70)
    print("\nTo use the AI assistant in your code:")
    print("  from neural.ai.ai_assistant import NeuralAIAssistant")
    print("  assistant = NeuralAIAssistant()")
    print("  result = assistant.chat('your natural language command')")
    print("\nTo use in chat interface:")
    print("  from neural.neural_chat.neural_chat import NeuralChat")
    print("  chat = NeuralChat(use_ai=True)")
    print("  response = chat.process_command('your command')")

