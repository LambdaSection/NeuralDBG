# Neural DSL Integration Guide

How to integrate data versioning and lineage tracking with Neural DSL workflows.

## Overview

The data versioning system integrates seamlessly with Neural DSL, enabling:
- Automatic lineage tracking during model compilation
- Dataset versioning before training
- Quality validation in the training pipeline
- Complete data→model→predictions tracking

## Integration Patterns

### Pattern 1: Version Data Before Compilation

```python
from neural.data import DatasetVersionManager
from neural.parser.parser import create_parser, ModelTransformer
from neural.code_generation import generate_code

# 1. Version your training data
version_mgr = DatasetVersionManager()
dataset_version = version_mgr.create_version(
    "data/mnist_train.npy",
    tags=["mnist", "training"],
    metadata={"samples": 60000}
)

# 2. Compile Neural DSL model
parser = create_parser()
with open("mnist_model.neural", "r") as f:
    dsl_code = f.read()

tree = parser.parse(dsl_code)
model_data = ModelTransformer().transform(tree)

# 3. Generate code with version metadata
model_data["dataset_version"] = dataset_version.version
code = generate_code(model_data, "tensorflow")

# 4. Track in lineage
from neural.data import LineageTracker

tracker = LineageTracker()
graph = tracker.create_graph("mnist_experiment")

data_node = tracker.add_data_node(
    "mnist_experiment",
    f"data_{dataset_version.version}",
    "MNIST Training Data",
    metadata={"version": dataset_version.version}
)

model_node = tracker.add_model_node(
    "mnist_experiment",
    "model_1",
    model_data["model"]["name"],
    metadata={"dsl_file": "mnist_model.neural"}
)

tracker.add_edge("mnist_experiment", data_node.node_id, model_node.node_id, "train")
```

### Pattern 2: Validate Data in Training Loop

```python
from neural.data import DataQualityValidator
import numpy as np

# Create validator
validator = DataQualityValidator()

# Add custom rules for your domain
validator.create_custom_rule(
    name="mnist_shape",
    rule_type="consistency",
    condition="valid_shape",
    description="MNIST images must be 28x28"
)

def train_with_validation(data_path, model):
    # Load data
    X_train = np.load(data_path)
    
    # Validate before training
    results = validator.validate(X_train)
    
    failed = [r for r in results if not r.passed]
    if failed:
        print(f"❌ Validation failed: {len(failed)} rules")
        for r in failed:
            print(f"  - {r.rule_name}: {r.message}")
        return None
    
    # Proceed with training
    print(f"✅ Data validation passed")
    model.fit(X_train, y_train)
    
    return model
```

### Pattern 3: Full Pipeline with Neural CLI

Create a shell script that orchestrates everything:

```bash
#!/bin/bash
# train_with_lineage.sh

set -e

# 1. Validate data
echo "Validating training data..."
neural data validate data/train.npy --save --name mnist_train

# 2. Version data
echo "Versioning dataset..."
VERSION=$(neural data version data/train.npy \
    --tags training mnist \
    --metadata '{"samples": 60000}' \
    --format json | jq -r '.version')

echo "Dataset version: $VERSION"

# 3. Compile Neural DSL model
echo "Compiling model..."
neural compile mnist_model.neural \
    --backend tensorflow \
    --output mnist_tf.py

# 4. Train model
echo "Training model..."
python mnist_tf.py

# 5. Track lineage
python << EOF
from neural.data import LineageTracker

tracker = LineageTracker()
graph = tracker.create_graph("mnist_exp_$VERSION")

tracker.add_data_node("mnist_exp_$VERSION", "data_$VERSION", "Training Data")
tracker.add_model_node("mnist_exp_$VERSION", "model_cnn", "CNN Classifier")
tracker.add_prediction_node("mnist_exp_$VERSION", "predictions", "Test Results")

tracker.add_edge("mnist_exp_$VERSION", "data_$VERSION", "model_cnn", "train")
tracker.add_edge("mnist_exp_$VERSION", "model_cnn", "predictions", "predict")

viz_path = tracker.visualize_lineage("mnist_exp_$VERSION")
print(f"Lineage saved: {viz_path}")
EOF

echo "✅ Training pipeline complete!"
```

### Pattern 4: Preprocessing Integration

```python
from neural.data import PreprocessingTracker, LineageTracker
import numpy as np

# Define preprocessing pipeline
prep_tracker = PreprocessingTracker()
pipeline = prep_tracker.create_pipeline(
    name="mnist_preprocessing",
    description="Standard MNIST preprocessing"
)

# Add preprocessing steps
def normalize(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)

def flatten(data, shape):
    return data.reshape(shape)

pipeline.add_step(
    "normalize",
    function=normalize,
    params={"min_val": 0, "max_val": 255},
    description="Normalize pixels to [0, 1]"
)

pipeline.add_step(
    "flatten",
    function=flatten,
    params={"shape": (-1, 784)},
    description="Flatten to vector"
)

prep_tracker.update_pipeline(pipeline)

# Apply preprocessing and track in lineage
data = np.load("mnist_raw.npy")
processed_data = pipeline.apply(data)

# Track preprocessing in lineage
tracker = LineageTracker()
graph = tracker.create_graph("mnist_with_prep")

raw_node = tracker.add_data_node(
    "mnist_with_prep", "raw_data", "Raw MNIST Data"
)

prep_node = tracker.add_preprocessing_node(
    "mnist_with_prep",
    "preprocess",
    "MNIST Preprocessing",
    metadata={"pipeline": "mnist_preprocessing"}
)

processed_node = tracker.add_data_node(
    "mnist_with_prep", "processed_data", "Processed MNIST Data"
)

tracker.add_edge("mnist_with_prep", raw_node.node_id, prep_node.node_id, "input")
tracker.add_edge("mnist_with_prep", prep_node.node_id, processed_node.node_id, "output")
```

### Pattern 5: Feature Store Integration

```python
from neural.data import FeatureStore
from neural.parser.parser import create_parser, ModelTransformer

# 1. Define features in feature store
store = FeatureStore()
group = store.create_feature_group(
    "mnist_features",
    description="MNIST image features"
)

store.add_feature("mnist_features", "pixels", "float32", "Normalized pixel values")
store.add_feature("mnist_features", "label", "int32", "Digit label (0-9)")

# 2. Parse Neural DSL
parser = create_parser()
tree = parser.parse(dsl_code)
model_data = ModelTransformer().transform(tree)

# 3. Link model to features
model_data["features"] = {
    "group": "mnist_features",
    "input_features": ["pixels"],
    "output_features": ["label"]
}

# 4. Track in lineage
from neural.data import LineageTracker

tracker = LineageTracker()
graph = tracker.create_graph("mnist_with_features")

feature_node = tracker.add_data_node(
    "mnist_with_features",
    "features",
    "MNIST Features",
    metadata={"feature_group": "mnist_features"}
)

model_node = tracker.add_model_node(
    "mnist_with_features",
    "model",
    model_data["model"]["name"],
    metadata={"features": model_data["features"]}
)

tracker.add_edge("mnist_with_features", feature_node.node_id, model_node.node_id, "train")
```

## Complete Example: End-to-End Pipeline

```python
"""
Complete Neural DSL pipeline with data versioning and lineage tracking
"""

import numpy as np
from pathlib import Path

from neural.data import (
    DatasetVersionManager,
    DataQualityValidator,
    PreprocessingTracker,
    FeatureStore,
    LineageTracker,
)
from neural.parser.parser import create_parser, ModelTransformer
from neural.code_generation import generate_code


def run_complete_pipeline(
    dsl_file: str,
    train_data: str,
    test_data: str,
    experiment_name: str,
):
    """Run complete ML pipeline with full tracking"""
    
    print(f"🚀 Starting experiment: {experiment_name}")
    
    # 1. Validate training data
    print("\n📊 Validating data quality...")
    validator = DataQualityValidator()
    
    train_array = np.load(train_data)
    results = validator.validate_and_save(train_array, "train_data")
    
    failed = [r for r in results if not r.passed]
    if failed:
        raise ValueError(f"Data validation failed: {[r.rule_name for r in failed]}")
    
    print(f"✅ Validation passed: {len(results)} checks")
    
    # 2. Version datasets
    print("\n📦 Versioning datasets...")
    version_mgr = DatasetVersionManager()
    
    train_version = version_mgr.create_version(
        train_data,
        tags=["training", experiment_name],
        metadata={"samples": len(train_array)}
    )
    
    test_version = version_mgr.create_version(
        test_data,
        tags=["testing", experiment_name]
    )
    
    print(f"✅ Train version: {train_version.version}")
    print(f"✅ Test version: {test_version.version}")
    
    # 3. Define preprocessing pipeline
    print("\n⚙️  Setting up preprocessing...")
    prep_tracker = PreprocessingTracker()
    
    pipeline = prep_tracker.create_pipeline(
        f"{experiment_name}_preprocessing",
        "Standard preprocessing pipeline"
    )
    
    pipeline.add_step("normalize", params={"min": 0, "max": 255})
    pipeline.add_step("reshape", params={"shape": (-1, 28, 28, 1)})
    prep_tracker.update_pipeline(pipeline)
    
    print(f"✅ Pipeline: {len(pipeline.steps)} steps")
    
    # 4. Register features
    print("\n🏪 Registering features...")
    feature_store = FeatureStore()
    
    features = feature_store.create_feature_group(
        f"{experiment_name}_features",
        "Features for this experiment"
    )
    
    feature_store.add_feature(
        f"{experiment_name}_features",
        "image_pixels",
        "float32",
        "Normalized image pixels"
    )
    
    print(f"✅ Features registered")
    
    # 5. Compile Neural DSL model
    print("\n🔧 Compiling model...")
    parser = create_parser()
    
    with open(dsl_file, "r") as f:
        dsl_code = f.read()
    
    tree = parser.parse(dsl_code)
    model_data = ModelTransformer().transform(tree)
    
    code = generate_code(model_data, "tensorflow")
    
    output_file = f"{experiment_name}_model.py"
    with open(output_file, "w") as f:
        f.write(code)
    
    print(f"✅ Model compiled: {output_file}")
    
    # 6. Create complete lineage graph
    print("\n🔗 Building lineage graph...")
    tracker = LineageTracker()
    graph = tracker.create_graph(experiment_name)
    
    # Add all nodes
    train_data_node = tracker.add_data_node(
        experiment_name,
        f"train_{train_version.version}",
        "Training Data",
        {"version": train_version.version}
    )
    
    prep_node = tracker.add_preprocessing_node(
        experiment_name,
        "preprocessing",
        "Preprocessing Pipeline",
        {"pipeline": f"{experiment_name}_preprocessing"}
    )
    
    feature_node = tracker.add_data_node(
        experiment_name,
        "features",
        "Extracted Features",
        {"feature_group": f"{experiment_name}_features"}
    )
    
    model_node = tracker.add_model_node(
        experiment_name,
        "model",
        model_data["model"]["name"],
        {"dsl_file": dsl_file, "output": output_file}
    )
    
    test_data_node = tracker.add_data_node(
        experiment_name,
        f"test_{test_version.version}",
        "Test Data",
        {"version": test_version.version}
    )
    
    pred_node = tracker.add_prediction_node(
        experiment_name,
        "predictions",
        "Test Predictions",
        {}
    )
    
    # Add edges
    tracker.add_edge(experiment_name, train_data_node.node_id, prep_node.node_id, "input")
    tracker.add_edge(experiment_name, prep_node.node_id, feature_node.node_id, "extract")
    tracker.add_edge(experiment_name, feature_node.node_id, model_node.node_id, "train")
    tracker.add_edge(experiment_name, test_data_node.node_id, model_node.node_id, "evaluate")
    tracker.add_edge(experiment_name, model_node.node_id, pred_node.node_id, "predict")
    
    print(f"✅ Lineage: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    
    # 7. Visualize lineage
    print("\n📊 Generating visualizations...")
    viz_path = tracker.visualize_lineage(experiment_name)
    print(f"✅ Lineage diagram: {viz_path}")
    
    print(f"\n✨ Experiment {experiment_name} setup complete!")
    print(f"\n📝 Next steps:")
    print(f"  1. Run training: python {output_file}")
    print(f"  2. View lineage: neural data lineage {experiment_name} --visualize")
    print(f"  3. Check versions: neural data list --tags {experiment_name}")
    
    return {
        "train_version": train_version.version,
        "test_version": test_version.version,
        "model_file": output_file,
        "lineage_graph": experiment_name,
    }


# Example usage
if __name__ == "__main__":
    # Create dummy data for demo
    np.save("train_data.npy", np.random.rand(1000, 28, 28))
    np.save("test_data.npy", np.random.rand(200, 28, 28))
    
    # Create dummy Neural DSL file
    with open("demo_model.neural", "w") as f:
        f.write("""
model MNISTClassifier {
    input: Image(28, 28, 1)
    
    Conv2D(32, 3, activation=relu)
    MaxPool2D(2)
    Conv2D(64, 3, activation=relu)
    MaxPool2D(2)
    Flatten()
    Dense(128, activation=relu)
    Dense(10, activation=softmax)
    
    output: Probabilities(10)
}
""")
    
    # Run complete pipeline
    result = run_complete_pipeline(
        dsl_file="demo_model.neural",
        train_data="train_data.npy",
        test_data="test_data.npy",
        experiment_name="mnist_demo_001"
    )
    
    print(f"\n✅ Pipeline complete!")
    print(f"Results: {result}")
```

## CLI Integration Examples

### Example 1: Pre-flight Checks

```bash
#!/bin/bash
# pre_train_checks.sh

echo "Running pre-flight checks..."

# Validate data
neural data validate data/train.npy --rules no_missing_values valid_shape
if [ $? -ne 0 ]; then
    echo "❌ Data validation failed"
    exit 1
fi

# Version data
neural data version data/train.npy --tags production
if [ $? -ne 0 ]; then
    echo "❌ Data versioning failed"
    exit 1
fi

echo "✅ Pre-flight checks passed"
```

### Example 2: Post-training Lineage

```bash
#!/bin/bash
# post_train_lineage.sh

EXPERIMENT_NAME=$1
MODEL_FILE=$2
ACCURACY=$3

python << EOF
from neural.data import LineageTracker

tracker = LineageTracker()

# Update model node with training results
graph = tracker.get_graph("$EXPERIMENT_NAME")
model_node = graph.get_node("model")
model_node.metadata["accuracy"] = $ACCURACY
model_node.metadata["model_file"] = "$MODEL_FILE"

tracker.update_graph(graph)

# Visualize final lineage
viz = tracker.visualize_lineage("$EXPERIMENT_NAME")
print(f"Lineage updated: {viz}")
EOF
```

## Best Practices

1. **Version before training**: Always version datasets before starting training
2. **Validate early**: Run quality checks before any processing
3. **Track preprocessing**: Document every transformation step
4. **Use meaningful IDs**: Node IDs should be descriptive
5. **Add metadata**: Include all relevant information in node metadata
6. **Visualize regularly**: Generate lineage diagrams for documentation
7. **Tag consistently**: Use consistent tagging scheme across experiments

## Troubleshooting

### Issue: Lineage graph becomes too large

**Solution**: Create separate graphs for different stages
```python
# Separate graphs for different phases
tracker.create_graph("data_preparation")
tracker.create_graph("model_training")
tracker.create_graph("model_evaluation")
```

### Issue: Dataset versions consume too much disk space

**Solution**: Use no-copy mode and/or DVC
```python
# Just track metadata, don't copy data
version_mgr.create_version("large_data.npy", copy_data=False)

# Or use DVC
from neural.data import DVCIntegration
dvc = DVCIntegration()
dvc.init()
dvc.add("large_data.npy")
```

### Issue: Need to share lineage across team

**Solution**: Export and import graphs
```python
# Export
tracker.export_graph("my_experiment", "lineage.json")

# Import (on another machine)
tracker.import_graph("lineage.json")
```

## Summary

The data versioning system integrates with Neural DSL to provide:
- ✅ Complete data→model→predictions lineage
- ✅ Automatic validation in training pipelines
- ✅ Feature store integration
- ✅ Preprocessing pipeline tracking
- ✅ CLI-based workflow automation
- ✅ Visual lineage documentation

For more examples and details, see:
- Full documentation: `neural/data/README.md`
- Quick start: `neural/data/QUICKSTART.md`
- Examples: `neural/data/examples.py`
