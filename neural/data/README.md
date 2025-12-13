# Neural Data Versioning & Lineage Tracking

Comprehensive data versioning and lineage tracking system for Neural DSL with DVC integration, dataset versioning with checksums, preprocessing pipeline tracking, feature store integration, data quality validation rules, and lineage visualization.

## Features

### 1. Dataset Versioning
- **Checksum-based tracking**: SHA256 checksums for data integrity
- **Version management**: Automatic or manual version naming
- **Metadata & tags**: Rich metadata and tagging system
- **Version comparison**: Compare datasets across versions
- **History tracking**: Full audit trail of dataset changes

### 2. DVC Integration
- **Seamless DVC workflow**: Initialize, add, pull, push datasets
- **Remote storage**: Configure and manage DVC remotes
- **Status tracking**: Monitor DVC-tracked files
- **Diff support**: Compare dataset versions in git-style
- **Automatic fallback**: Works without DVC installed

### 3. Preprocessing Pipeline Tracking
- **Pipeline definition**: Define multi-step preprocessing workflows
- **Step tracking**: Record each preprocessing operation
- **Parameter logging**: Track hyperparameters for each step
- **Pipeline comparison**: Compare different preprocessing approaches
- **Import/Export**: Share pipelines across projects

### 4. Feature Store
- **Feature groups**: Organize features into logical groups
- **Type safety**: Track data types for each feature
- **Metadata**: Rich feature descriptions and metadata
- **Search**: Find features by name or description
- **Statistics**: Feature usage and distribution analytics

### 5. Data Quality Validation
- **Built-in rules**: No missing values, no duplicates, valid shape
- **Custom rules**: Define domain-specific validation rules
- **Validation history**: Track quality metrics over time
- **Quality reports**: Generate comprehensive quality reports
- **Automatic validation**: Integrate into versioning workflow

### 6. Lineage Tracking & Visualization
- **Graph-based lineage**: Track data→preprocessing→model→predictions flow
- **Node types**: Data, preprocessing, model, prediction nodes
- **Edge metadata**: Track relationships between nodes
- **Bidirectional tracing**: Upstream and downstream lineage
- **Visualization**: Generate graphviz diagrams of lineage

## Installation

```bash
# Core functionality (no DVC)
pip install -e .

# With DVC support
pip install -e . dvc

# With visualization support
pip install -e ".[visualization]"

# Full installation
pip install -e ".[full]"
```

## Quick Start

### Dataset Versioning

```python
from neural.data import DatasetVersionManager

# Initialize manager
manager = DatasetVersionManager(base_dir=".neural_data")

# Create a version
version = manager.create_version(
    dataset_path="data/train.npy",
    metadata={"samples": 10000, "split": "train"},
    tags=["training", "v1"]
)

print(f"Version: {version.version}")
print(f"Checksum: {version.checksum}")

# List versions
versions = manager.list_versions(tags=["training"])

# Compare versions
comparison = manager.compare_versions("v1", "v2")
```

### DVC Integration

```python
from neural.data import DVCIntegration

# Initialize DVC
dvc = DVCIntegration()
dvc.init()

# Add dataset to DVC
dvc.add("data/large_dataset.npy")

# Configure remote
dvc.add_remote("storage", "s3://mybucket/dvc-storage", default=True)

# Push to remote
dvc.pull()
```

### Preprocessing Pipeline

```python
from neural.data import PreprocessingTracker

# Create tracker
tracker = PreprocessingTracker(base_dir=".neural_data")

# Define pipeline
pipeline = tracker.create_pipeline(
    name="image_preprocessing",
    description="Standard image preprocessing"
)

# Add steps
pipeline.add_step(
    name="normalize",
    params={"min": 0, "max": 255},
    description="Normalize to [0, 1]"
)

pipeline.add_step(
    name="augment",
    params={"rotation": 15, "flip": True},
    description="Data augmentation"
)

# Save pipeline
tracker.update_pipeline(pipeline)
```

### Feature Store

```python
from neural.data import FeatureStore

# Initialize store
store = FeatureStore(base_dir=".neural_data")

# Create feature group
group = store.create_feature_group(
    name="user_features",
    description="User-level features"
)

# Add features
store.add_feature(
    "user_features",
    "age",
    "int32",
    "User age in years"
)

store.add_feature(
    "user_features",
    "activity_score",
    "float32",
    "User activity score",
    metadata={"range": [0, 1]}
)

# Search features
results = store.search_features("age")
```

### Data Quality Validation

```python
from neural.data import DataQualityValidator
import numpy as np

# Initialize validator
validator = DataQualityValidator(base_dir=".neural_data")

# Validate data
data = np.random.rand(100, 10)
results = validator.validate(data)

# Check results
for result in results:
    print(f"{result.rule_name}: {'PASS' if result.passed else 'FAIL'}")

# Create custom rule
validator.create_custom_rule(
    name="min_samples",
    rule_type="size",
    condition="min_rows",
    threshold=1000,
    description="Dataset must have at least 1000 samples"
)
```

### Lineage Tracking

```python
from neural.data import LineageTracker

# Initialize tracker
tracker = LineageTracker(base_dir=".neural_data")

# Create lineage graph
graph = tracker.create_graph("experiment_1")

# Add nodes
data_node = tracker.add_data_node(
    "experiment_1",
    "data_v1",
    "Training Data v1.0",
    {"samples": 10000}
)

model_node = tracker.add_model_node(
    "experiment_1",
    "model_cnn",
    "CNN Classifier",
    {"architecture": "ResNet50"}
)

pred_node = tracker.add_prediction_node(
    "experiment_1",
    "predictions",
    "Test Predictions",
    {"accuracy": 0.95}
)

# Add edges
tracker.add_edge("experiment_1", data_node.node_id, model_node.node_id, "train")
tracker.add_edge("experiment_1", model_node.node_id, pred_node.node_id, "predict")

# Visualize
tracker.visualize_lineage("experiment_1", output_path="lineage.png")

# Trace lineage
lineage = tracker.get_full_lineage("experiment_1", "data_v1")
print(f"Downstream nodes: {len(lineage['downstream'])}")
```

## CLI Commands

### Version a Dataset

```bash
# Create a version
neural data version data/train.npy --tags training v1 --metadata '{"samples": 10000}'

# List versions
neural data list

# List versions with specific tags
neural data list --tags training
```

### Validate Data Quality

```bash
# Validate a dataset
neural data validate data/train.npy

# Validate with specific rules
neural data validate data/train.npy --rules no_missing_values no_duplicates

# Save validation results
neural data validate data/train.npy --save --name train_data
```

### View Lineage

```bash
# Show lineage graph
neural data lineage experiment_1

# Trace lineage from a node
neural data lineage experiment_1 --trace data_v1

# Visualize lineage
neural data lineage experiment_1 --visualize --output lineage.png
```

## Advanced Usage

### Complete Data→Model→Predictions Workflow

```python
from neural.data.utils import create_data_to_model_lineage

# Create complete lineage
graph_name = create_data_to_model_lineage(
    graph_name="mnist_experiment",
    dataset_version="v1.0",
    preprocessing_pipeline="standard_pipeline",
    model_name="cnn_classifier",
    prediction_name="test_predictions"
)

print(f"Created lineage graph: {graph_name}")
```

### Setup Complete Project

```python
from neural.data.utils import setup_data_versioning_project

# Setup all components
project = setup_data_versioning_project(
    base_dir=".neural_data",
    use_dvc=True
)

# Access components
version_manager = project["version_manager"]
feature_store = project["feature_store"]
quality_validator = project["quality_validator"]
```

### Validate and Version Together

```python
from neural.data.utils import validate_and_version_dataset

# Validate and version in one step
result = validate_and_version_dataset(
    dataset_path="data/train.npy",
    version="v1.0",
    tags=["training", "validated"],
    validate=True
)

if result["success"]:
    print(f"Version created: {result['version']}")
    print(f"Checksum: {result['checksum']}")
else:
    print(f"Validation failed: {result['error']}")
    print(f"Failed rules: {result['failed_rules']}")
```

## Directory Structure

```
.neural_data/
├── versions.json              # Dataset version registry
├── datasets/                  # Versioned datasets
│   ├── v_20240101_120000/
│   └── v_20240101_130000/
├── preprocessing_pipelines/   # Pipeline definitions
│   ├── mnist_pipeline.json
│   └── cifar_pipeline.json
├── features/                  # Feature store
│   ├── image_features.json
│   └── text_features.json
├── quality_rules/            # Validation rules
│   └── custom_rules.json
├── validation_results/       # Validation history
│   └── train_data_20240101_120000.json
└── lineage/                  # Lineage graphs
    └── experiment_1.json
```

## Examples

Run all examples:

```bash
python -m neural.data.examples
```

Or run specific examples:

```python
from neural.data.examples import (
    example_dataset_versioning,
    example_dvc_integration,
    example_preprocessing_pipeline,
    example_feature_store,
    example_quality_validation,
    example_lineage_tracking,
)

example_dataset_versioning()
example_lineage_tracking()
```

## Integration with Neural DSL

### Automatic Lineage Tracking

The data module integrates seamlessly with Neural DSL compilation:

```python
from neural.data import LineageTracker
from neural.parser.parser import create_parser, ModelTransformer

# Parse Neural DSL
parser = create_parser()
tree = parser.parse(dsl_code)
model_data = ModelTransformer().transform(tree)

# Track in lineage
tracker = LineageTracker()
graph = tracker.create_graph("my_experiment")

# Add model node
model_node = tracker.add_model_node(
    "my_experiment",
    "model_1",
    model_data["model"]["name"],
    metadata=model_data
)
```

## Best Practices

1. **Version early and often**: Create versions at key milestones
2. **Use meaningful tags**: Tag versions with purpose (training, validation, test)
3. **Add metadata**: Include sample counts, feature descriptions, etc.
4. **Validate before versioning**: Catch issues early with quality validation
5. **Track preprocessing**: Document all data transformations
6. **Visualize lineage**: Generate diagrams for documentation
7. **Use DVC for large files**: Integrate DVC for datasets > 100MB
8. **Export pipelines**: Share preprocessing pipelines across projects

## API Reference

See individual module docstrings for detailed API documentation:

- `DatasetVersion` and `DatasetVersionManager`: Dataset versioning
- `DVCIntegration`: DVC integration
- `PreprocessingPipeline` and `PreprocessingTracker`: Pipeline tracking
- `Feature`, `FeatureGroup`, and `FeatureStore`: Feature management
- `QualityRule`, `ValidationResult`, and `DataQualityValidator`: Quality validation
- `LineageNode`, `LineageEdge`, `LineageGraph`, and `LineageTracker`: Lineage tracking

## Troubleshooting

### DVC not found

```bash
pip install dvc
# or for S3 support
pip install dvc[s3]
```

### Graphviz visualization fails

```bash
# Ubuntu/Debian
sudo apt-get install graphviz

# macOS
brew install graphviz

# Windows
# Download from https://graphviz.org/download/

# Python package
pip install graphviz
```

### Large dataset handling

For datasets > 1GB, use DVC integration or disable copying:

```python
version = manager.create_version(
    dataset_path="large_data.npy",
    copy_data=False  # Only track, don't copy
)
```

## Contributing

Contributions are welcome! Please ensure:

- Type hints for all functions
- Docstrings with parameter descriptions
- Unit tests for new features
- Updated examples

## License

MIT License - see LICENSE file for details
