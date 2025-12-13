#!/usr/bin/env python3
"""
Quick Prototyping Script
Launch interactive model builder for rapid experimentation.
"""

import click

from neural.templates import get_template, list_templates


def print_banner():
    print("""
╔═══════════════════════════════════════════════════════╗
║   Neural DSL - Quick Prototype Generator             ║
║   Build models in minutes, not hours                  ║
╚═══════════════════════════════════════════════════════╝
    """)


@click.command()
@click.option('--interactive', '-i', is_flag=True, help='Interactive mode')
@click.option('--template', '-t', help='Template name')
@click.option('--output', '-o', default='prototype.neural', help='Output file')
def main(interactive, template, output):
    """Quick prototyping tool for Neural DSL."""
    print_banner()
    
    if not interactive and not template:
        print("💡 Tip: Use --interactive for guided model creation")
        print("   Or: --template <name> for quick template generation\n")
        print("Available templates:")
        templates = list_templates()
        for t in templates:
            print(f"  • {t['name']:<20} - {t['description']}")
        print("\nExample: python scripts/quick_prototype.py -t mnist_cnn -o my_model.neural")
        return
    
    if interactive:
        run_interactive_mode(output)
    elif template:
        generate_from_template(template, output)


def run_interactive_mode(output_file):
    """Interactive model builder."""
    print("\n🚀 Interactive Model Builder\n")
    
    # Step 1: Choose domain
    print("1. What domain are you working in?")
    domains = {
        '1': ('Computer Vision', 'image'),
        '2': ('Natural Language Processing', 'text'),
        '3': ('Time Series', 'sequence'),
        '4': ('Other', 'general'),
    }
    
    for key, (name, _) in domains.items():
        print(f"   {key}) {name}")
    
    domain_choice = input("\nChoice (1-4): ").strip()
    domain_name, domain_type = domains.get(domain_choice, domains['4'])
    
    print(f"\n✓ Selected: {domain_name}\n")
    
    # Step 2: Domain-specific questions
    model_config = {}
    
    if domain_type == 'image':
        model_config = configure_image_model()
    elif domain_type == 'text':
        model_config = configure_text_model()
    elif domain_type == 'sequence':
        model_config = configure_sequence_model()
    else:
        model_config = configure_general_model()
    
    # Step 3: Generate model
    print("\n📝 Generating model...")
    
    # Select appropriate template
    template_name = model_config.get('template', 'image_classifier')
    try:
        model_code = get_template(template_name, **model_config)
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write(model_code)
        
        print(f"✅ Model generated: {output_file}\n")
        
        # Next steps
        print("📋 Next steps:")
        print(f"   1. Review: cat {output_file}")
        print(f"   2. Visualize: neural visualize {output_file} --format png")
        print(f"   3. Compile: neural compile {output_file} --backend tensorflow")
        print(f"   4. Experiment: Edit {output_file} to customize\n")
        
        print("💡 Tips:")
        print("   • Start simple, add complexity gradually")
        print("   • Use --educational flag for learning: neural compile --educational")
        print("   • Try HPO for automatic tuning: neural compile --hpo")
        
    except Exception as e:
        print(f"❌ Error generating model: {e}")


def configure_image_model():
    """Configure image classification model."""
    print("2. Image dimensions?")
    print("   Common: 28 (MNIST), 32 (CIFAR), 224 (ImageNet)")
    
    size_input = input("   Size (default: 224): ").strip()
    size = int(size_input) if size_input else 224
    
    print("\n3. Color or grayscale?")
    color = input("   (color/grayscale, default: color): ").strip().lower()
    channels = 1 if color == 'grayscale' else 3
    
    print("\n4. Number of classes to predict?")
    num_classes_input = input("   (default: 1000): ").strip()
    num_classes = int(num_classes_input) if num_classes_input else 1000
    
    print("\n5. Model complexity?")
    print("   1) Simple (fast, good for small datasets)")
    print("   2) Medium (balanced)")
    print("   3) Complex (slow, best accuracy)")
    
    complexity = input("   Choice (default: 2): ").strip() or '2'
    
    template_map = {
        '1': 'mnist_cnn',
        '2': 'image_classifier',
        '3': 'vgg_style'
    }
    
    return {
        'template': template_map.get(complexity, 'image_classifier'),
        'input_shape': (size, size, channels),
        'num_classes': num_classes,
    }


def configure_text_model():
    """Configure text classification model."""
    print("2. Vocabulary size?")
    vocab_input = input("   (default: 10000): ").strip()
    vocab_size = int(vocab_input) if vocab_input else 10000
    
    print("\n3. Maximum text length (in tokens)?")
    length_input = input("   (default: 500): ").strip()
    max_length = int(length_input) if length_input else 500
    
    print("\n4. Number of classes?")
    classes_input = input("   (default: 3): ").strip()
    num_classes = int(classes_input) if classes_input else 3
    
    print("\n5. Model architecture?")
    print("   1) LSTM (good for sequences, slower)")
    print("   2) CNN (faster, good for shorter texts)")
    print("   3) Transformer (state-of-the-art, slowest)")
    
    arch = input("   Choice (default: 1): ").strip() or '1'
    
    template_map = {
        '1': 'text_lstm',
        '2': 'text_cnn',
        '3': 'simple_transformer'
    }
    
    return {
        'template': template_map.get(arch, 'text_lstm'),
        'vocab_size': vocab_size,
        'max_length': max_length,
        'num_classes': num_classes,
    }


def configure_sequence_model():
    """Configure time series model."""
    print("2. Sequence length (number of time steps)?")
    length_input = input("   (default: 100): ").strip()
    sequence_length = int(length_input) if length_input else 100
    
    print("\n3. Number of features per time step?")
    features_input = input("   (default: 1): ").strip()
    num_features = int(features_input) if features_input else 1
    
    return {
        'template': 'time_series',
        'sequence_length': sequence_length,
        'num_features': num_features,
    }


def configure_general_model():
    """Configure general model."""
    print("For custom architectures, starting with image_classifier template")
    print("You can edit it to match your needs.")
    
    return {
        'template': 'image_classifier',
        'input_shape': (224, 224, 3),
        'num_classes': 10,
    }


def generate_from_template(template_name, output_file):
    """Generate model from template directly."""
    print(f"\n📝 Generating from template: {template_name}")
    
    try:
        model_code = get_template(template_name)
        
        with open(output_file, 'w') as f:
            f.write(model_code)
        
        print(f"✅ Model generated: {output_file}\n")
        print("Next steps:")
        print(f"   1. Customize: edit {output_file}")
        print(f"   2. Visualize: neural visualize {output_file}")
        print(f"   3. Compile: neural compile {output_file} --backend tensorflow")
        
    except ValueError as e:
        print(f"❌ {e}")
        print("\nAvailable templates:")
        templates = list_templates()
        for t in templates:
            print(f"  • {t['name']}")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == '__main__':
    main()
