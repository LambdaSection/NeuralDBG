# NeuralDbg Logic Graph

This document provides a visual representation of the NeuralDbg debugging workflow and the Event class logic.

## Overall System Logic Flow

```mermaid
graph TD
    A[Start Neural Network Training] --> B[Wrap Model with NeuralDbg]
    B --> C[Forward Pass]
    C --> D[Create Event for each layer]
    D --> E[Store input/output tensors]
    E --> F[Backward Pass]
    F --> G[Capture gradients for each Event]
    G --> H{Gradient Analysis}
    H --> I[Detect Vanishing Gradients?]
    I --> J[Set Breakpoint]
    J --> K[Query State for Debugging]
    K --> L[End Training/Debugging]

    D --> M[Event Object]
    M --> N[step, layer_name]
    M --> O[input tensor (detached clone)]
    M --> P[output tensor (detached clone)]
    M --> Q[gradient = None initially]
    Q --> R[After backward: capture_gradient() called]
    R --> S[gradient = tensor.grad (detached clone)]
```

## Event Class Logic Flow

```mermaid
graph TD
    A[New Event Created] --> B{__init__ called}
    B --> C[Store step number]
    B --> D[Store layer_name]
    B --> E[Detach and clone input_tensor]
    B --> F[Detach and clone output_tensor]
    B --> G[Set gradient = None]

    H[During Backward Pass] --> I[capture_gradient(tensor) called]
    I --> J{tensor.grad exists?}
    J -->|Yes| K[Detach and clone gradient]
    J -->|No| L[Keep gradient = None]

    C --> M[Event ready for inspection]
    D --> M
    E --> M
    F --> M
    K --> M
```

## Key Concepts Illustrated

1. **Event Creation**: Happens during forward pass, captures layer state
2. **Tensor Handling**: All tensors are detached and cloned to prevent interference
3. **Gradient Capture**: Occurs during backward pass, enables gradient analysis
4. **Debugging Integration**: Events enable breakpoint setting and state querying

## Tensor Lifecycle in Event

```mermaid
graph LR
    A[Original Input Tensor] --> B[detach()]
    B --> C[clone()]
    C --> D[Event.input]

    E[Original Output Tensor] --> F[detach()]
    F --> G[clone()]
    G --> H[Event.output]

    I[Gradient Computed] --> J[detach()]
    J --> K[clone()]
    K --> L[Event.gradient]
```

This logic graph shows how NeuralDbg captures and preserves neural network state for debugging purposes, with the Event class serving as the core data structure for storing computation snapshots.
