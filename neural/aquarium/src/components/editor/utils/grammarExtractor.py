"""
Grammar Extractor Utility

This script extracts token definitions from the Neural DSL Lark grammar
and generates TypeScript/JavaScript constants for use in Monaco editor.

Usage:
    python grammarExtractor.py
"""

import re
import json
from pathlib import Path


def extract_grammar_info():
    """Extract information from the Neural DSL grammar."""
    grammar_file = Path(__file__).parent.parent.parent.parent.parent / "parser" / "grammar.py"
    
    if not grammar_file.exists():
        print(f"Grammar file not found at {grammar_file}")
        return None
    
    with open(grammar_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    grammar_match = re.search(r'NEURAL_DSL_GRAMMAR = r"""(.+?)"""', content, re.DOTALL)
    if not grammar_match:
        print("Could not extract grammar from file")
        return None
    
    grammar = grammar_match.group(1)
    
    layer_types = []
    keywords = []
    
    for line in grammar.split('\n'):
        line = line.strip()
        
        if line.startswith('//') or not line:
            continue
        
        if re.match(r'[A-Z_]+:\s*"[a-z]+"i', line):
            match = re.match(r'([A-Z_]+):\s*"([a-z]+)"i', line)
            if match:
                token_name = match.group(1)
                token_value = match.group(2)
                
                if token_name in ['DENSE', 'CONV2D', 'CONV1D', 'CONV3D', 'LSTM', 'GRU', 
                                  'DROPOUT', 'FLATTEN', 'OUTPUT', 'MAXPOOLING1D', 'MAXPOOLING2D',
                                  'MAXPOOLING3D', 'BATCHNORMALIZATION', 'TRANSFORMER']:
                    layer_types.append(token_value.capitalize())
                elif token_name not in ['TRUE', 'FALSE', 'NONE']:
                    keywords.append(token_value)
    
    return {
        'layer_types': sorted(set(layer_types)),
        'keywords': sorted(set(keywords)),
    }


def generate_typescript_constants(grammar_info):
    """Generate TypeScript constants from grammar info."""
    output = []
    
    output.append("// Auto-generated from Neural DSL grammar")
    output.append("// DO NOT EDIT MANUALLY\n")
    
    output.append("export const LAYER_TYPES = [")
    for layer in grammar_info['layer_types']:
        output.append(f"  '{layer}',")
    output.append("];\n")
    
    output.append("export const KEYWORDS = [")
    for keyword in grammar_info['keywords']:
        output.append(f"  '{keyword}',")
    output.append("];\n")
    
    return '\n'.join(output)


def main():
    """Main function."""
    print("Extracting grammar information...")
    grammar_info = extract_grammar_info()
    
    if not grammar_info:
        print("Failed to extract grammar information")
        return
    
    print(f"Found {len(grammar_info['layer_types'])} layer types")
    print(f"Found {len(grammar_info['keywords'])} keywords")
    
    ts_constants = generate_typescript_constants(grammar_info)
    
    output_file = Path(__file__).parent / "grammarConstants.ts"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(ts_constants)
    
    print(f"Generated TypeScript constants at {output_file}")
    
    print("\nLayer types:")
    for layer in grammar_info['layer_types']:
        print(f"  - {layer}")
    
    print("\nKeywords:")
    for keyword in grammar_info['keywords']:
        print(f"  - {keyword}")


if __name__ == '__main__':
    main()
