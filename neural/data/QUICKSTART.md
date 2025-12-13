# Quick Start Guide: Data Versioning & Lineage Tracking

Get started with Neural's data versioning and lineage tracking in 5 minutes!

## Installation

```bash
# Basic installation
pip install -e .

# With data versioning support
pip install -e ".[data]"

# With visualization
pip install -e ".[data,visualization]"
```

## 1. Version Your First Dataset (30 seconds)

```bash
# Create a test dataset
python -c "import numpy as np; np.save('my_data.npy', np.random.rand(1000, 10))"

# Version it
neural data version my_data.npy --tags training v1 --metadata '{"samples": 1000, "features": 10}'

# List versions
neural data list
```

**Output:**
```
✓ Dataset version created!

Version Information:
  Version:    v_20240101_120000
  Checksum:   a1b2c3d4e5f6...
  Created:    2024-01-01T12:00:00
  Tags:       training, v1
```

## 2. Validate Data Quality (30 seconds)

```bash
# Validate the dataset
neural data validate my_data.npy --save --name training_data

# Check validation results
neural data validate my_data.npy
```

**Output:**
```
✓ Validation complete!

Validation Results:
  Total Rules: 3
  Passed:      3
  Failed:      0

Details:
  ✓ [PASS] no_missing_values: Validation passed
  ✓ [PASS] no_duplicates: Validation passed
  ✓ [PASS] valid_shape: Validation passed
```

## 3. Track Preprocessing Pipeline (1 minute)

```python
from neural.data import PreprocessingTracker

# Create tracker
tracker = PreprocessingTracker()

# Define pipeline
pipeline = tracker.create_pipeline(
    name="standard_preprocessing",
    description="Normalize and augment"
)

# Add steps
pipeline.add_step("normalize", params={"min": 0, "max": 255})
pipeline.add_step("augment", params={"rotation": 15, "flip": True})

# Save pipeline
tracker.update_pipeline(pipeline)

print(f"Pipeline created with {len(pipeline.steps)} steps")
```

## 4. Create Data Lineage (1 minute)

```python
from neural.data import LineageTracker

# Initialize tracker
tracker = LineageTracker()

# Create lineage graph
graph = tracker.create_graph("my_experiment")

# Add nodes for data → model → predictions flow
data_node = tracker.add_data_node(
    "my_experiment", "data_v1", "Training Data v1.0"
)

model_node = tracker.add_model_node(
    "my_experiment", "model_cnn", "CNN Classifier"
)

pred_node = tracker.add_prediction_node(
    "my_experiment", "predictions", "Test Predictions"
)

# Connect them
tracker.add_edge("my_experiment", "data_v1", "model_cnn", "train")
tracker.add_edge("my_experiment", "model_cnn", "predictions", "predict")

print("Lineage graph created!")
```

## 5. Visualize Lineage (30 seconds)

```bash
# View lineage statistics
neural data lineage my_experiment

# Visualize as diagram
neural data lineage my_experiment --visualize --output my_lineage.png
```

**Output:**
```
✓ Lineage graph: my_experiment

Graph Statistics:
  Total Nodes:  3
  Total Edges:  2

Node Types:
  data: 1
  model: 1
  prediction: 1

✓ Lineage visualization saved: my_lineage.png
```

## 6. Complete Workflow Example (2 minutes)

Putting it all together:

```python
from neural.data import (
    DatasetVersionManager,
    PreprocessingTracker,
    DataQualityValidator,
    LineageTracker,
)
import numpy as np

# 1. Create and version dataset
data = np.random.rand(10000, 784)
np.save("mnist_train.npy", data)

version_mgr = DatasetVersionManager()
dataset_v1 = version_mgr.create_version(
    "mnist_train.npy",
    metadata={"samples": 10000, "features": 784},
    tags=["mnist", "training", "v1"]
)
print(f"✓ Dataset versioned: {dataset_v1.version}")

# 2. Validate data quality
validator = DataQualityValidator()
results = validator.validate(data)
passed = sum(1 for r in results if r.passed)
print(f"✓ Validation: {passed}/{len(results)} checks passed")

# 3. Create preprocessing pipeline
prep_tracker = PreprocessingTracker()
pipeline = prep_tracker.create_pipeline("mnist_pipeline")
pipeline.add_step("normalize", params={"min": 0, "max": 255})
pipeline.add_step("flatten", params={"shape": [-1, 784]})
prep_tracker.update_pipeline(pipeline)
print(f"✓ Pipeline created: {len(pipeline.steps)} steps")

# 4. Track complete lineage
lineage = LineageTracker()
graph = lineage.create_graph("mnist_experiment")

# Add all nodes
data_node = lineage.add_data_node(
    "mnist_experiment", 
    f"data_{dataset_v1.version}",
    "MNIST Training Data"
)
prep_node = lineage.add_preprocessing_node(
    "mnist_experiment",
    "preprocess_mnist",
    "MNIST Pipeline"
)
model_node = lineage.add_model_node(
    "mnist_experiment",
    "model_cnn",
    "CNN Classifier"
)
pred_node = lineage.add_prediction_node(
    "mnist_experiment",
    "predictions",
    "Test Predictions"
)

# Connect pipeline
lineage.add_edge("mnist_experiment", data_node.node_id, prep_node.node_id, "input")
lineage.add_edge("mnist_experiment", prep_node.node_id, model_node.node_id, "train")
lineage.add_edge("mnist_experiment", model_node.node_id, pred_node.node_id, "predict")

print(f"✓ Lineage tracked: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

# 5. Generate visualization
viz_path = lineage.visualize_lineage("mnist_experiment")
print(f"✓ Visualization saved: {viz_path}")
```

## Common Use Cases

### Use Case 1: Track ML Experiment

```python
from neural.data.utils import create_data_to_model_lineage

# One-line lineage creation
create_data_to_model_lineage(
    graph_name="experiment_001",
    dataset_version="v1.0",
    preprocessing_pipeline="standard",
    model_name="resnet50",
    prediction_name="val_predictions"
)
```

### Use Case 2: Validate Before Versioning

```python
from neural.data.utils import validate_and_version_dataset

result = validate_and_version_dataset(
    dataset_path="data.npy",
    tags=["production"],
    validate=True
)

if result["success"]:
    print(f"✓ Version: {result['version']}")
else:
    print(f"✗ Failed: {result['error']}")
    print(f"  Failed rules: {result['failed_rules']}")
```

### Use Case 3: Compare Dataset Versions

```python
from neural.data import DatasetVersionManager

mgr = DatasetVersionManager()

# Create two versions
v1 = mgr.create_version("data_v1.npy", version="v1.0")
v2 = mgr.create_version("data_v2.npy", version="v2.0")

# Compare
diff = mgr.compare_versions("v1.0", "v2.0")
print(f"Checksums match: {diff['checksum_match']}")
print(f"Metadata changes: {len(diff['metadata_diff'])}")
```

### Use Case 4: Feature Store Management

```python
from neural.data import FeatureStore

store = FeatureStore()

# Create feature group
group = store.create_feature_group("user_features")

# Add features
store.add_feature("user_features", "age", "int32", "User age")
store.add_feature("user_features", "score", "float32", "Activity score")

# Search features
results = store.search_features("age")
print(f"Found {len(results)} features matching 'age'")
```

### Use Case 5: DVC Integration

```python
from neural.data import DVCIntegration

dvc = DVCIntegration()

# Initialize DVC
if dvc.is_dvc_available():
    dvc.init()
    
    # Add large dataset to DVC
    dvc.add("large_dataset.npy")
    
    # Configure remote storage
    dvc.add_remote("s3storage", "s3://mybucket/dvc", default=True)
    
    print("✓ DVC setup complete")
else:
    print("DVC not installed: pip install dvc")
```

## Next Steps

1. **Read the full documentation**: `neural/data/README.md`
2. **Run examples**: `python -m neural.data.examples`
3. **Check CLI help**: `neural data --help`
4. **Explore tests**: `tests/test_data_versioning.py`

## Tips & Best Practices

1. **Always validate before versioning** - Catch data quality issues early
2. **Use meaningful tags** - Makes filtering and discovery easier
3. **Track preprocessing steps** - Essential for reproducibility
4. **Visualize lineage regularly** - Helps understand data flow
5. **Use DVC for large files** - Keep git repo lightweight
6. **Export pipelines** - Share preprocessing across projects
7. **Set up remotes early** - Enable collaboration from the start

## Troubleshooting

**Problem**: "DVC not found"
```bash
pip install dvc
# or with S3 support
pip install dvc[s3]
```

**Problem**: "Graphviz visualization failed"
```bash
# Ubuntu/Debian
sudo apt-get install graphviz

# macOS
brew install graphviz

# Then install Python package
pip install graphviz
```

**Problem**: "Large datasets slow"
```python
# Use no-copy mode for large datasets
mgr.create_version(
    "large_data.npy",
    copy_data=False  # Only track metadata
)
```

## Get Help

- Documentation: `neural/data/README.md`
- Examples: `neural/data/examples.py`
- Tests: `tests/test_data_versioning.py`
- CLI Help: `neural data --help`

## Summary

You've learned how to:
- ✅ Version datasets with checksums
- ✅ Validate data quality
- ✅ Track preprocessing pipelines
- ✅ Create and visualize lineage graphs
- ✅ Integrate with DVC
- ✅ Use CLI commands

Now you're ready to build reproducible ML workflows with full data lineage tracking!
