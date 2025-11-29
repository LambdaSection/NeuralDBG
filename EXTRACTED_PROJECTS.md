# Extracted Projects - Neural Ecosystem

This document tracks projects that were extracted from the main Neural repository into separate repositories as part of the repository cleanup on November 28, 2025.

## Extracted Projects

### 1. Neural-Aquarium
- **Original Location**: `Aquarium/` directory
- **Description**: Tauri-based desktop GUI application for Neural DSL
- **Technology**: Rust + Tauri + Frontend (React/Vue)
- **New Repository**: TBD (to be created)
- **Status**: Pending extraction

### 2. NeuralPaper
- **Original Location**: `neuralpaper/` directory
- **Description**: Full-stack web application for Neural research and papers
- **Technology**: Python backend + JavaScript frontend
- **Components**: Backend API, Frontend UI, deployment config (render.yaml)
- **New Repository**: TBD (to be created)
- **Status**: Pending extraction

### 3. Neural-Research (Paper Annotation)
- **Original Location**: `paper_annotation/` directory
- **Description**: Historical neural network paper analysis and annotations
- **Content**: McCulloch-Pitts 1943 paper PDF, Jupyter notebooks, Medium export scripts
- **New Repository**: TBD (to be created)
- **Status**: Pending extraction

### 4. Lambda-sec Models
- **Original Location**: `Lambda-sec Models/` directory
- **Description**: Production neural network models for λ-S startup
- **Content**: MathLang models (Newton differential equations), Transformer architectures, training data
- **New Repository**: TBD (to be created)
- **Part of**: λ-S startup ecosystem (which includes Neural)
- **Status**: Pending extraction

### 5. NeuralDbg (Part of Core Neural)
- **Location**: `neural/dashboard/` directory
- **Description**: Built-in debugger and visualizer for Neural DSL
- **Features**: Real-time execution tracing, gradient analysis, anomaly detection
- **Technology**: Python (Dash, Flask, Plotly)
- **Status**: Integrated into main Neural repository
- **Note**: NeuralDbg is the debugging component for the AI IDE - remains part of core Neural, not extracted

## Integration Notes

While these projects are now in separate repositories, they remain part of the Neural ecosystem:

- **Aquarium** provides a desktop GUI for Neural DSL
- **NeuralPaper** hosts web-based research and documentation
- **Neural-Research** contains historical research and paper annotations
- **Lambda-sec Models** provides production models for the λ-S startup

## Links

Once extracted, links to the new repositories will be added here:
- [ ] Neural-Aquarium: [GitHub URL]
- [ ] NeuralPaper: [GitHub URL]
- [ ] Neural-Research: [GitHub URL]
- [ ] Lambda-sec Models: [GitHub URL]

## Extraction Checklist

- [ ] Create new GitHub repositories
- [ ] Move code to new repos with full git history
- [ ] Update README.md in main Neural repo with links
- [ ] Add deprecation notices in old directories
- [ ] Update DISTRIBUTION_JOURNAL.md
- [ ] Remove old directories from main repo
