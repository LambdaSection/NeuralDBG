#!/usr/bin/env python3
"""
Visualization utility for NeuralDBG.

Exports the causal chain of semantic events as a Mermaid diagram.
"""

import sys
from neuraldbg import NeuralDbg

def main():
    """Main visualization entry point."""
    print("🎨 NeuralDBG Causal Chain Visualizer")
    print("=" * 50)
    
    # In a real scenario, this would load a pickle or JSON export
    print("Note: This is a standalone utility that expects a NeuralDbg instance.")
    print("For a live demonstration, run: python3 demo_vanishing_gradients.py")
    print()
    
    if len(sys.argv) < 2:
        print("Usage: python3 visualize_causality.py <output_file.md>")
        return

    print(f"Generating example Mermaid diagram to {sys.argv[1]}...")
    
    example_mermaid = """## Causal Chain Visualization

```mermaid
graph TD
    E0["activation_regime_shift in layer1 (Step 10)"]
    E1["gradient_health_transition in layer1 (Step 11)"]
    E2["gradient_health_transition in layer2 (Step 12)"]
    
    E0 -->|causes| E1
    E1 -->|propagates| E2
    E0 <-->|coupled| E2
```
"""
    
    with open(sys.argv[1], "w") as f:
        f.write(example_mermaid)
        
    print(f"✅ Visualization saved to {sys.argv[1]}")

if __name__ == "__main__":
    main()
