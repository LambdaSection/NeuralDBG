#!/usr/bin/env python
"""
Transformer Tutorial: Interactive Guide to Attention Mechanisms
================================================================

This tutorial demonstrates the Transformer architecture using Neural DSL,
including the recently added layers: PositionalEncoding, MultiHeadAttention,
TransformerEncoder, and TransformerDecoder.

Based on "Attention Is All You Need" (Vaswani et al., 2017)

Topics covered:
1. Scaled Dot-Product Attention
2. Multi-Head Attention
3. Positional Encoding
4. Encoder-Decoder Architecture
5. Complete Transformer Examples
"""

import math
from typing import Tuple

import numpy as np


def print_section(title: str, char: str = "=") -> None:
    """Print a formatted section header."""
    print(f"\n{char * 70}")
    print(f"{title.center(70)}")
    print(f"{char * 70}\n")


def print_subsection(title: str) -> None:
    """Print a formatted subsection header."""
    print(f"\n{'-' * 60}")
    print(f"{title}")
    print(f"{'-' * 60}\n")


class AttentionDemo:
    """Interactive demonstration of attention mechanisms."""
    
    @staticmethod
    def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, 
                                     mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute scaled dot-product attention.
        
        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
        
        Args:
            Q: Query matrix (seq_len_q, d_k)
            K: Key matrix (seq_len_k, d_k)
            V: Value matrix (seq_len_v, d_v)
            mask: Optional mask (seq_len_q, seq_len_k)
            
        Returns:
            output: Attention output (seq_len_q, d_v)
            attention_weights: Attention weights (seq_len_q, seq_len_k)
        """
        d_k = Q.shape[-1]
        
        # Step 1: Compute attention scores
        scores = np.matmul(Q, K.T) / math.sqrt(d_k)
        
        # Step 2: Apply mask (if provided)
        if mask is not None:
            scores = scores + (mask * -1e9)
        
        # Step 3: Apply softmax
        attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention_weights = attention_weights / np.sum(attention_weights, axis=-1, keepdims=True)
        
        # Step 4: Apply attention to values
        output = np.matmul(attention_weights, V)
        
        return output, attention_weights
    
    @staticmethod
    def demo_attention() -> None:
        """Demonstrate attention mechanism with example."""
        print_subsection("Scaled Dot-Product Attention Demo")
        
        # Create simple example: 3 tokens, embedding dim = 4
        seq_len, d_k = 3, 4
        
        # Initialize Q, K, V
        np.random.seed(42)
        Q = np.random.randn(seq_len, d_k)
        K = np.random.randn(seq_len, d_k)
        V = np.random.randn(seq_len, d_k)
        
        print("Input shapes:")
        print(f"  Query (Q):  {Q.shape} - Where to look")
        print(f"  Key (K):    {K.shape} - What to look for")
        print(f"  Value (V):  {V.shape} - What to retrieve")
        
        # Compute attention
        output, weights = AttentionDemo.scaled_dot_product_attention(Q, K, V)
        
        print("\nAttention Computation:")
        print(f"  1. Scores = Q @ K^T / sqrt({d_k}) = Q @ K^T / {math.sqrt(d_k):.2f}")
        print("  2. Attention Weights = softmax(Scores)")
        print("  3. Output = Attention Weights @ V")
        
        print("\nAttention weights (how much each token attends to others):")
        for i in range(seq_len):
            print(f"  Token {i}: {weights[i]}")
        
        print(f"\nOutput shape: {output.shape}")
        print("Each output token is a weighted combination of all value vectors")


class PositionalEncodingDemo:
    """Demonstration of positional encoding."""
    
    @staticmethod
    def sinusoidal_encoding(position: int, d_model: int) -> np.ndarray:
        """
        Generate sinusoidal positional encoding for a position.
        
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        """
        pe = np.zeros(d_model)
        for i in range(0, d_model, 2):
            div_term = math.exp(i * -(math.log(10000.0) / d_model))
            pe[i] = math.sin(position * div_term)
            if i + 1 < d_model:
                pe[i + 1] = math.cos(position * div_term)
        return pe
    
    @staticmethod
    def demo_positional_encoding() -> None:
        """Demonstrate positional encoding."""
        print_subsection("Positional Encoding Demo")
        
        print("Why Positional Encoding?")
        print("  Transformers have NO inherent notion of sequence order")
        print("  Positional encoding adds position information to embeddings")
        print("  Uses sine/cosine functions of different frequencies\n")
        
        seq_len, d_model = 10, 8
        
        print(f"Generating positional encodings for {seq_len} positions, {d_model} dimensions:")
        
        # Generate encodings
        encodings = np.array([
            PositionalEncodingDemo.sinusoidal_encoding(pos, d_model)
            for pos in range(seq_len)
        ])
        
        print(f"Shape: {encodings.shape}\n")
        
        print("First 3 positions (showing pattern):")
        for pos in range(3):
            print(f"  Position {pos}: {encodings[pos][:4]}... (first 4 dims)")
        
        print("\nKey Properties:")
        print("  ✓ Unique encoding for each position")
        print("  ✓ Relative position information encoded in dot products")
        print("  ✓ Can extrapolate to longer sequences than seen in training")


class TransformerLayerExamples:
    """Examples of using Transformer layers in Neural DSL."""
    
    @staticmethod
    def example_multihead_attention() -> str:
        """Example: Standalone MultiHeadAttention layer."""
        return """
# Example 1: Self-Attention with MultiHeadAttention
# ==================================================

network SelfAttentionClassifier {
  input: (128, 512)  # (sequence_length, embedding_dim)
  
  layers:
    # Multi-head self-attention
    # - 8 heads split the 512-dim space into 8 subspaces of 64-dim each
    # - Each head learns different attention patterns
    # - Dropout prevents overfitting on attention weights
    MultiHeadAttention(num_heads:8, key_dim:64, dropout:0.1)
    
    # Normalize for stable training
    LayerNormalization()
    
    # Feed-forward transformation
    Dense(units:256, activation:"relu")
    Dropout(rate:0.1)
    
    # Output layer
    Output(units:10, activation:"softmax")
  
  loss: "categorical_crossentropy"
  optimizer: Adam(learning_rate:0.001)
  
  train {
    epochs: 20
    batch_size: 32
  }
}
"""
    
    @staticmethod
    def example_positional_encoding() -> str:
        """Example: PositionalEncoding layer."""
        return """
# Example 2: Positional Encoding
# ===============================

network PositionalEncodingExample {
  input: (100, 512)
  
  layers:
    # Add positional information to embeddings
    # - max_len: Maximum sequence length (default: 5000)
    # - encoding_type: 'sinusoidal' or 'learnable'
    PositionalEncoding(max_len:1000, encoding_type:"sinusoidal")
    
    # Dropout on embeddings + position
    Dropout(rate:0.1)
    
    # Process with attention
    MultiHeadAttention(num_heads:8, key_dim:64, dropout:0.1)
    LayerNormalization()
    
    # Output
    Dense(units:256, activation:"relu")
    Output(units:10, activation:"softmax")
  
  optimizer: Adam(learning_rate:0.0001)
  loss: "categorical_crossentropy"
}
"""
    
    @staticmethod
    def example_transformer_encoder() -> str:
        """Example: TransformerEncoder layer."""
        return """
# Example 3: Complete Transformer Encoder Block
# ==============================================

network EncoderModel {
  input: (512,)  # Token IDs
  
  layers:
    # Embed tokens to dense vectors
    Embedding(input_dim:30000, output_dim:512, mask_zero:True)
    
    # Add positional information
    PositionalEncoding(max_len:512)
    Dropout(rate:0.1)
    
    # TransformerEncoder includes:
    # - Multi-head self-attention
    # - Add & Norm (residual + layer norm)
    # - Feed-forward network
    # - Add & Norm (residual + layer norm)
    TransformerEncoder(num_heads:8, ff_dim:2048, dropout:0.1)
    TransformerEncoder(num_heads:8, ff_dim:2048, dropout:0.1)
    TransformerEncoder(num_heads:8, ff_dim:2048, dropout:0.1)
    
    # Global pooling for classification
    GlobalAveragePooling1D()
    
    # Classification head
    Dense(units:128, activation:"relu")
    Dropout(rate:0.1)
    Output(units:20, activation:"softmax")
  
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate:0.0001, beta_1:0.9, beta_2:0.98)
  
  train {
    epochs: 50
    batch_size: 64
  }
}
"""
    
    @staticmethod
    def example_encoder_decoder() -> str:
        """Example: Encoder-Decoder Transformer."""
        return """
# Example 4: Encoder-Decoder Transformer (Seq2Seq)
# =================================================

network Seq2SeqTransformer {
  input: (100, 512)
  
  layers:
    # ENCODER STACK
    # Token embedding + positional encoding
    PositionalEncoding(max_len:512)
    Dropout(rate:0.1)
    
    # Stack of encoder layers
    # Each layer has self-attention + feed-forward
    TransformerEncoder(num_heads:8, ff_dim:2048, dropout:0.1)
    TransformerEncoder(num_heads:8, ff_dim:2048, dropout:0.1)
    TransformerEncoder(num_heads:8, ff_dim:2048, dropout:0.1)
    TransformerEncoder(num_heads:8, ff_dim:2048, dropout:0.1)
    
    # DECODER STACK
    # TransformerDecoder includes:
    # - Masked self-attention (prevents looking ahead)
    # - Cross-attention to encoder outputs
    # - Feed-forward network
    # - Residual connections & layer normalization
    TransformerDecoder(num_heads:8, ff_dim:2048, dropout:0.1)
    TransformerDecoder(num_heads:8, ff_dim:2048, dropout:0.1)
    TransformerDecoder(num_heads:8, ff_dim:2048, dropout:0.1)
    TransformerDecoder(num_heads:8, ff_dim:2048, dropout:0.1)
    
    # Output projection to vocabulary
    Dense(units:32000, activation:"linear")
    Output(units:32000, activation:"softmax")
  
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate:0.0001, beta_1:0.9, beta_2:0.98, epsilon:1e-9)
  
  train {
    epochs: 100
    batch_size: 32
    validation_split: 0.1
  }
}
"""
    
    @staticmethod
    def example_bert_style() -> str:
        """Example: BERT-style encoder."""
        return """
# Example 5: BERT-Style Encoder (Bidirectional)
# ==============================================

network BertStyleEncoder {
  input: (512,)
  
  layers:
    # Embedding layer
    Embedding(input_dim:30522, output_dim:768, mask_zero:True)
    LayerNormalization(epsilon:1e-12)
    Dropout(rate:0.1)
    
    # Stack of 12 transformer encoder layers (BERT-Base)
    # Each layer has:
    # - Multi-head attention (12 heads)
    # - Feed-forward network (3072 = 4 * 768)
    # - Layer normalization and residual connections
    TransformerEncoder(num_heads:12, ff_dim:3072, dropout:0.1)
    TransformerEncoder(num_heads:12, ff_dim:3072, dropout:0.1)
    TransformerEncoder(num_heads:12, ff_dim:3072, dropout:0.1)
    TransformerEncoder(num_heads:12, ff_dim:3072, dropout:0.1)
    TransformerEncoder(num_heads:12, ff_dim:3072, dropout:0.1)
    TransformerEncoder(num_heads:12, ff_dim:3072, dropout:0.1)
    TransformerEncoder(num_heads:12, ff_dim:3072, dropout:0.1)
    TransformerEncoder(num_heads:12, ff_dim:3072, dropout:0.1)
    TransformerEncoder(num_heads:12, ff_dim:3072, dropout:0.1)
    TransformerEncoder(num_heads:12, ff_dim:3072, dropout:0.1)
    TransformerEncoder(num_heads:12, ff_dim:3072, dropout:0.1)
    TransformerEncoder(num_heads:12, ff_dim:3072, dropout:0.1)
    
    # Classification head (for masked language modeling)
    Dense(units:768, activation:"gelu")
    LayerNormalization(epsilon:1e-12)
    Dense(units:30522, activation:"linear")
    Activation("softmax")
  
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate:0.0001, beta_1:0.9, beta_2:0.999, epsilon:1e-6)
  
  train {
    epochs: 40
    batch_size: 32
    validation_split: 0.05
  }
}
"""


def explain_transformer_architecture() -> None:
    """Explain the overall Transformer architecture."""
    print_section("Transformer Architecture Overview")
    
    print("""
The Transformer architecture consists of:

1. INPUT EMBEDDING + POSITIONAL ENCODING
   ├─ Token Embedding: Maps words/subwords to dense vectors
   └─ Positional Encoding: Adds position information

2. ENCODER (Left side)
   ├─ Multi-Head Self-Attention: Captures relationships between input tokens
   ├─ Add & Norm: Residual connection + Layer Normalization
   ├─ Feed-Forward Network: Position-wise transformation
   └─ Add & Norm: Residual connection + Layer Normalization
   (Repeat N times, typically N=6 or N=12)

3. DECODER (Right side)
   ├─ Masked Multi-Head Self-Attention: Prevents looking at future tokens
   ├─ Add & Norm
   ├─ Cross-Attention: Attends to encoder outputs
   ├─ Add & Norm
   ├─ Feed-Forward Network
   └─ Add & Norm
   (Repeat N times)

4. OUTPUT LAYER
   └─ Linear + Softmax: Predicts next token probabilities
""")


def explain_attention_mechanism() -> None:
    """Explain how attention works."""
    print_section("Understanding Attention Mechanism", "-")
    
    print("""
ATTENTION: A mechanism to focus on relevant parts of input

Intuition:
  When translating "The cat sat on the mat" to French,
  translating "sat" should attend more to "cat" than "mat"

Scaled Dot-Product Attention:
  1. Query (Q): What I'm looking for
  2. Key (K):   What each position offers
  3. Value (V): Actual content to retrieve
  
  Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

Steps:
  1. Compute similarity scores: Q @ K^T
  2. Scale by 1/sqrt(d_k) to prevent vanishing gradients
  3. Apply softmax to get attention weights (probabilities)
  4. Use weights to take weighted sum of values

Multi-Head Attention:
  Instead of single attention, use H parallel attention heads
  Each head can focus on different aspects (syntax, semantics, etc.)
  Concat all heads and project back to model dimension
""")


def usage_guide() -> None:
    """Provide usage instructions."""
    print_section("Usage Guide")
    
    print("""
Compiling Neural DSL files:
  
  TensorFlow backend:
    $ neural compile my_transformer.neural --backend tensorflow
  
  PyTorch backend:
    $ neural compile my_transformer.neural --backend pytorch
  
  ONNX export:
    $ neural compile my_transformer.neural --backend onnx

Visualizing architectures:
    $ neural visualize my_transformer.neural

Debugging with NeuralDbg:
    $ neural debug my_transformer.neural

Layer Multiplication (for stacking):
    TransformerEncoder(num_heads:8, ff_dim:2048) * 6
    # Creates 6 identical encoder layers
""")


def practical_tips() -> None:
    """Provide practical implementation tips."""
    print_section("Practical Implementation Tips")
    
    print("""
1. Hyperparameter Selection:
   - d_model: 256, 512, 768, 1024 (model dimension)
   - num_heads: 4, 8, 12, 16 (more heads = more patterns)
   - ff_dim: 4 * d_model (feed-forward hidden size)
   - dropout: 0.1 - 0.2 (prevents overfitting)
   - num_layers: 3-12 (deeper = more capacity, slower training)

2. Learning Rate:
   - Use learning rate warmup (first 4000-10000 steps)
   - Peak LR: 1e-4 to 1e-3
   - Use Adam with beta_1=0.9, beta_2=0.98, epsilon=1e-9

3. Training Tips:
   - Start with smaller models for faster iteration
   - Use gradient clipping (max_norm=1.0)
   - Apply label smoothing (0.1) for better generalization
   - Use mixed precision training for speed

4. Sequence Length:
   - Attention is O(n²) in sequence length
   - Use max_len appropriate for your task
   - Consider sparse attention for very long sequences

5. Memory Optimization:
   - Reduce batch_size if OOM
   - Use gradient accumulation for effective larger batches
   - Enable gradient checkpointing for deep models

6. Common Pitfalls:
   - Forgetting positional encoding → model can't use order
   - Wrong mask in decoder → model sees future tokens
   - Learning rate too high → unstable training
   - No warmup → poor convergence
""")


def layer_comparison() -> None:
    """Compare different Transformer-related layers."""
    print_section("Layer Comparison")
    
    print("""
┌─────────────────────────────────────────────────────────────────┐
│ Layer                  │ Components                              │
├─────────────────────────────────────────────────────────────────┤
│ MultiHeadAttention     │ • Multi-head attention only             │
│                        │ • Manual control over architecture      │
│                        │ • Use for custom designs                │
├─────────────────────────────────────────────────────────────────┤
│ TransformerEncoder     │ • Multi-head self-attention             │
│                        │ • Feed-forward network                  │
│                        │ • 2x Layer normalization                │
│                        │ • 2x Residual connections               │
│                        │ • Complete encoder block                │
├─────────────────────────────────────────────────────────────────┤
│ TransformerDecoder     │ • Masked self-attention                 │
│                        │ • Cross-attention (to encoder)          │
│                        │ • Feed-forward network                  │
│                        │ • 3x Layer normalization                │
│                        │ • 3x Residual connections               │
│                        │ • Complete decoder block                │
├─────────────────────────────────────────────────────────────────┤
│ PositionalEncoding     │ • Adds position information             │
│                        │ • Sinusoidal or learnable               │
│                        │ • Required for Transformers             │
└─────────────────────────────────────────────────────────────────┘

When to use each:
  - MultiHeadAttention: Building custom architectures
  - TransformerEncoder: Standard transformer encoder (BERT, etc.)
  - TransformerDecoder: Seq2seq models (translation, generation)
  - PositionalEncoding: Always with Transformers (before encoder/decoder)
""")


def main():
    """Main tutorial execution."""
    print_section("TRANSFORMER TUTORIAL: ATTENTION IS ALL YOU NEED")
    print("Interactive guide to Transformer architecture in Neural DSL\n")
    
    # Section 1: Core Concepts
    explain_attention_mechanism()
    
    # Section 2: Interactive Demos
    attention_demo = AttentionDemo()
    attention_demo.demo_attention()
    
    positional_demo = PositionalEncodingDemo()
    positional_demo.demo_positional_encoding()
    
    # Section 3: Architecture Overview
    explain_transformer_architecture()
    
    # Section 4: Layer Comparison
    layer_comparison()
    
    # Section 5: Neural DSL Examples
    print_section("Neural DSL Examples")
    
    examples = TransformerLayerExamples()
    
    print_subsection("Example 1: MultiHeadAttention Layer")
    print(examples.example_multihead_attention())
    
    print_subsection("Example 2: PositionalEncoding Layer")
    print(examples.example_positional_encoding())
    
    print_subsection("Example 3: TransformerEncoder Layer")
    print(examples.example_transformer_encoder())
    
    print_subsection("Example 4: Encoder-Decoder Architecture")
    print(examples.example_encoder_decoder())
    
    print_subsection("Example 5: BERT-Style Encoder")
    print(examples.example_bert_style())
    
    # Section 6: Practical Tips
    practical_tips()
    
    # Section 7: Usage Guide
    usage_guide()
    
    # Final message
    print_section("Next Steps")
    print("""
1. Explore the example files:
   - examples/multihead_attention.neural
   - examples/positional_encoding_example.ndsl
   - examples/bert_encoder.neural
   - examples/seq2seq_transformer.neural

2. Try modifying hyperparameters:
   - Experiment with different num_heads
   - Adjust ff_dim and dropout rates
   - Stack different numbers of layers

3. Compile and run:
   - Start with small models
   - Monitor training with NeuralDbg
   - Use visualization tools

4. Read the paper:
   "Attention Is All You Need" (Vaswani et al., 2017)
   https://arxiv.org/abs/1706.03762

Happy building! 🚀
""")


if __name__ == "__main__":
    main()
