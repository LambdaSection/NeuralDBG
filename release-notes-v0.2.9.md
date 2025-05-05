# Neural v0.2.9 Release Notes

## Overview

Neural v0.2.9 introduces the Aquarium IDE, a specialized development environment for neural network design and debugging. This release focuses on enhancing the developer experience with visual tools for network design and real-time shape propagation.

## Key Features

### Aquarium IDE

- **Visual Network Designer**: Drag-and-drop interface for creating neural network architectures
- **Real-time Shape Propagation**: Visualize tensor dimensions as they flow through your network
- **Integrated Debugging**: Catch dimensional errors before training begins
- **Neural DSL Integration**: Automatically generate Neural DSL code from visual designs
- **Performance Analysis**: Estimate computational requirements and memory usage

### Technology Stack

- **Frontend**: Tauri with JavaScript/HTML/CSS
- **Backend**: Rust for performance-critical components
- **Neural Integration**: Direct integration with Neural's shape propagator

## Code Quality Improvements

- Fixed trailing whitespace and missing newlines at end of files across the codebase
- Improved code consistency and adherence to style guidelines
- Enhanced readability and maintainability

## Repository Organization

- Added Aquarium IDE as a Git submodule for better separation of concerns
- Enhanced project structure with clear separation between Neural core and IDE
- Updated documentation to reflect new repository organization

## Installation

```bash
# Install the latest version
pip install neural-dsl==0.2.9
```

## Future Improvements

- **Enhanced Shape Propagation**: More detailed visualization of tensor dimensions
- **Additional Layer Support**: Support for more layer types and configurations
- **Interactive Canvas**: More interactive design experience with drag-and-drop connections
- **Integration Testing**: Comprehensive tests for Aquarium IDE integration with Neural

## Documentation

For more information, see:
- [Aquarium README](Aquarium/README.md)
- [Updated Repository Structure](REPOSITORY_STRUCTURE.md)
- [Changelog](CHANGELOG.md)
