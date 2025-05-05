---
title: "Neural DSL v0.2.9: Early Preview of Aquarium IDE for Visual Neural Network Design"
published: true
description: "Neural DSL v0.2.9 introduces an early preview of Aquarium IDE, a new development environment for visual neural network design with basic shape propagation capabilities."
tags: machinelearning, python, neuralnetworks, ide, visualization
cover_image: https://github.com/user-attachments/assets/f92005cc-7b1c-4020-aec6-0e6922c36b1b
---

# Neural DSL v0.2.9: Early Preview of Aquarium IDE for Visual Neural Network Design

![Neural DSL Logo](https://github.com/user-attachments/assets/f92005cc-7b1c-4020-aec6-0e6922c36b1b)

We're pleased to announce the release of Neural DSL v0.2.9, which includes an early preview of Aquarium IDE, a new development environment for neural network design. This initial release provides basic visual tools for network design and integrates with Neural's shape propagation system.

> "Aquarium IDE is our first step toward making neural network development more visual and accessible. While still in early development, we believe this approach will help both beginners and experienced developers better understand their network architectures." — Neural DSL Team

## 🚀 Spotlight Feature: Aquarium IDE (Early Preview)

Aquarium IDE is a new development environment for neural network design that we're releasing as an early preview. In this initial version, it provides a basic visual interface for designing simple neural networks and viewing tensor shapes.

### Current Features

- **Basic Visual Designer**: Simple interface for adding and configuring common layer types
- **Shape Calculation**: View tensor dimensions for each layer in your network
- **Neural DSL Code Generation**: Generate basic Neural DSL code from your visual design
- **Parameter Estimation**: Basic calculation of parameter counts for each layer

### Technology Stack

Aquarium IDE is built with:

- **Frontend**: Tauri with JavaScript/HTML/CSS for cross-platform compatibility
- **Backend**: Rust components for shape calculation
- **Neural Integration**: Integration with Neural's shape propagator for tensor dimension calculations

## 🔍 How Aquarium IDE Works (Current Implementation)

### 1. Basic Network Design

In this early preview, Aquarium IDE provides a simple interface where you can add layers to your network. The current version supports a limited set of common layer types (Input, Conv2D, MaxPooling2D, Flatten, Dense, and Output). Each layer can be configured through a basic properties panel.

```
+----------------+     +----------------+     +----------------+
|    Input       |     |    Conv2D      |     |  MaxPooling2D  |
| (28, 28, 1)    | --> | filters=32     | --> | pool_size=(2,2)|
|                |     | kernel=(3,3)   |     |                |
+----------------+     +----------------+     +----------------+
        |
        v
+----------------+     +----------------+     +----------------+
|    Flatten     |     |     Dense      |     |    Output      |
|                | --> | units=128      | --> | units=10       |
|                |     | activation=relu|     | activation=soft|
+----------------+     +----------------+     +----------------+
```

### 2. Shape Calculation

The current version calculates basic tensor dimensions for each layer in your network. This is a simplified implementation that works for common layer types and configurations but may not handle all edge cases or complex architectures.

```
Layer         | Input Shape      | Output Shape     | Parameters
--------------|------------------|------------------|------------
Input Layer   | -                | [null,28,28,1]   | 0
Conv2D        | [null,28,28,1]   | [null,28,28,32]  | 320
MaxPooling2D  | [null,28,28,32]  | [null,14,14,32]  | 0
Flatten       | [null,14,14,32]  | [null,6272]      | 0
Dense         | [null,6272]      | [null,128]       | 802,944
Output        | [null,128]       | [null,10]        | 1,290
```

### 3. Basic Code Generation

The current version generates simple Neural DSL code from your visual design. The code generation is limited to the supported layer types and basic configurations.

```yaml
# Neural DSL Model

Input(shape=[28, 28, 1])
Conv2D(filters=32, kernel_size=[3, 3], padding="same", activation="relu")
MaxPooling2D(pool_size=[2, 2])
Flatten()
Dense(units=128, activation="relu")
Output(units=10, activation="softmax")
```

### Current Limitations

It's important to note that this early preview has several limitations:

- Only supports a small set of layer types
- Limited parameter configuration options
- Basic shape calculation that may not handle all edge cases
- Simple code generation without advanced features
- No support for complex network architectures (e.g., multi-input/output, skip connections)
- Limited error checking and validation

## 🛠️ Getting Started with Aquarium IDE

### Installation

Aquarium IDE is included as a submodule in the Neural repository. To try this early preview:

```bash
# Clone the Neural repository
git clone https://github.com/Lemniscate-world/Neural.git
cd Neural

# Update submodules to get Aquarium
git submodule update --init --recursive

# Install Rust if you don't have it already
# https://www.rust-lang.org/tools/install

# Install Tauri CLI
cargo install tauri-cli

# Navigate to the Aquarium directory
cd Aquarium

# Install Node.js dependencies
npm install

# Run the development server (this may take a few minutes the first time)
cargo tauri dev
```

Note: As this is an early preview, you may encounter some issues during installation or runtime. Please report any problems on our GitHub issues page.

### Trying the Basic Features

1. **Add Layers**: Use the buttons in the left panel to add some basic layers
2. **Configure Parameters**: Try adjusting some simple parameters like units or filters
3. **View Shapes**: Switch to the shape tab to see basic tensor dimensions
4. **See Generated Code**: Check the code tab to view the generated Neural DSL code
5. **Experiment**: This is an early preview, so feel free to experiment and provide feedback

## 🔧 Code Quality Improvements

In addition to the Aquarium IDE preview, Neural v0.2.9 includes some code quality improvements:

- Fixed trailing whitespace and missing newlines at end of files across the codebase
- Improved code consistency and adherence to style guidelines
- Enhanced readability and maintainability of the codebase

These changes, while not user-facing, help maintain a healthy codebase for future development.

## 📦 Installation

To try Neural DSL v0.2.9 with the Aquarium IDE preview:

```bash
# Install the core Neural DSL package
pip install neural-dsl==0.2.9

# To try Aquarium IDE, follow the installation instructions above
# as it requires additional dependencies (Rust, Node.js, etc.)
```

Or upgrade from a previous version:

```bash
pip install --upgrade neural-dsl
```

## 🔍 Roadmap for Aquarium IDE

Aquarium IDE is in very early development, and we have a long roadmap ahead. Some of the features we're planning to work on:

- **Support for More Layer Types**: Add support for additional layer types beyond the basic ones
- **Improved Shape Propagation**: More accurate and detailed shape calculations
- **Better Error Handling**: Provide more helpful error messages and validation
- **Visual Connections**: Allow creating connections between layers visually
- **Save/Load Functionality**: Save and load network designs
- **Export to Multiple Formats**: Export to different backends and formats

We welcome feedback and contributions to help shape the future of Aquarium IDE.

## 🔗 Resources

- [GitHub Repository](https://github.com/Lemniscate-world/Neural)
- [Documentation](https://github.com/Lemniscate-world/Neural/blob/main/docs/dsl.md)
- [Discord Community](https://discord.gg/KFku4KvS)
- [Example Notebooks](https://github.com/Lemniscate-world/Neural/tree/main/examples)
- [Blog Archive](https://github.com/Lemniscate-world/Neural/tree/main/docs/blog)

## 🙏 Feedback and Contributions

As Aquarium IDE is in early development, we're especially interested in:

- **Bug Reports**: If you encounter issues, please report them on GitHub
- **Feature Requests**: Let us know what features would be most useful to you
- **Usability Feedback**: Tell us about your experience using the early preview
- **Contributions**: If you're interested in contributing to the development, check out our [Contributing Guidelines](https://github.com/Lemniscate-world/Neural/blob/main/CONTRIBUTING.md)

## 🏁 Conclusion

Neural DSL v0.2.9 introduces an early preview of Aquarium IDE, our first step toward making neural network development more visual and accessible. While this is just the beginning and the current implementation has limitations, we believe this approach has potential to help both beginners and experienced developers better understand their network architectures.

We're looking forward to your feedback as we continue to develop Aquarium IDE. Please share your thoughts, suggestions, and questions with us on [Discord](https://discord.gg/KFku4KvS) or [GitHub](https://github.com/Lemniscate-world/Neural/discussions).
