# Data Versioning and Lineage Tracking Implementation

## Overview

A comprehensive data versioning and lineage tracking system has been implemented in `neural/data/` with full DVC integration, dataset versioning, preprocessing pipeline tracking, feature store integration, data quality validation, and lineage visualization.

## What Was Implemented

### Core Modules

1. **`dataset_version.py`** - Dataset Versioning System
   - `DatasetVersion`: Represents a versioned dataset with checksum
   - `DatasetVersionManager`: Manages dataset versions, comparisons, and tagging
   - Features:
     - SHA256 checksums for data integrity
     - Automatic/manual version naming
     - Metadata and tagging support
     - Version comparison and history
     - Optional data copying

2. **`dvc_integration.py`** - DVC Integration
   - `DVCIntegration`: Seamless DVC workflow integration
   - Features:
     - Initialize DVC repositories
     - Add/pull/push datasets
     - Configure remotes (S3, GCS, Azure, etc.)
     - Status tracking and diff support
     - Automatic fallback when DVC not installed

3. **`preprocessing_tracker.py`** - Preprocessing Pipeline Tracking
   - `PreprocessingStep`: Individual preprocessing operation
   - `PreprocessingPipeline`: Multi-step preprocessing workflow
   - `PreprocessingTracker`: Manages and compares pipelines
   - Features:
     - Define preprocessing steps with parameters
     - Track execution history
     - Compare pipeline differences
     - Import/export pipelines

4. **`feature_store.py`** - Feature Store
   - `Feature`: Individual feature definition
   - `FeatureGroup`: Logical grouping of features
   - `FeatureStore`: Central feature registry
   - Features:
     - Type-safe feature definitions
     - Rich metadata and descriptions
     - Feature search and discovery
     - Usage statistics
     - Import/export support

5. **`quality_validator.py`** - Data Quality Validation
   - `QualityRule`: Validation rule definition
   - `ValidationResult`: Validation outcome
   - `DataQualityValidator`: Applies validation rules
   - Features:
     - Built-in rules (no missing values, no duplicates, valid shape)
     - Custom rule creation
     - Validation history tracking
     - Quality reports
     - Threshold-based rules

6. **`lineage_tracker.py`** - Lineage Tracking & Visualization
   - `LineageNode`: Node in lineage graph (data/preprocessing/model/prediction)
   - `LineageEdge`: Connection between nodes
   - `LineageGraph`: Complete lineage graph
   - `LineageTracker`: Manages lineage graphs
   - Features:
     - Graph-based lineage tracking
     - Bidirectional tracing (upstream/downstream)
     - Path finding between nodes
     - Graphviz visualization
     - Import/export support

7. **`utils.py`** - Utility Functions
   - `create_data_to_model_lineage()`: Quick lineage creation
   - `setup_data_versioning_project()`: Initialize all components
   - `validate_and_version_dataset()`: Combined validation + versioning
   - `get_dataset_lineage_summary()`: Generate lineage summaries
   - `export_data_project()` / `import_data_project()`: Project sharing

### CLI Commands

Added three new CLI command groups to `neural/cli/cli.py`:

1. **`neural data version`** - Create dataset versions
   - Arguments: `dataset_path`
   - Options: `--version`, `--metadata`, `--tags`, `--no-copy`, `--base-dir`
   - Creates versioned snapshot with checksum

2. **`neural data list`** - List dataset versions
   - Options: `--tags`, `--format`, `--base-dir`
   - Shows all versions with filtering and formatting

3. **`neural data validate`** - Validate data quality
   - Arguments: `dataset_path`
   - Options: `--rules`, `--save`, `--name`, `--base-dir`
   - Runs quality validation rules

4. **`neural data lineage`** - Show/visualize lineage
   - Arguments: `graph_name`
   - Options: `--trace`, `--visualize`, `--output`, `--format`, `--base-dir`
   - Displays and visualizes data lineage graphs

### Documentation

1. **`README.md`** - Comprehensive documentation
   - Feature overview
   - Installation instructions
   - API reference
   - CLI usage examples
   - Best practices
   - Troubleshooting guide

2. **`QUICKSTART.md`** - 5-minute quick start guide
   - Step-by-step getting started
   - Common use cases
   - CLI examples
   - Code snippets

3. **`INTEGRATION.md`** - Neural DSL integration guide
   - Integration patterns
   - Complete pipeline examples
   - CLI automation scripts
   - Best practices

4. **`examples.py`** - Runnable examples
   - Dataset versioning example
   - DVC integration example
   - Preprocessing pipeline example
   - Feature store example
   - Quality validation example
   - Lineage tracking example

### Tests

**`tests/test_data_versioning.py`** - Comprehensive test suite
- Dataset versioning tests
- Version comparison tests
- Preprocessing pipeline tests
- Feature store tests
- Quality validation tests (built-in and custom rules)
- Lineage tracking tests
- Import/export tests
- All major functionality covered

### Configuration Updates

1. **`setup.py`** - Added data dependencies
   - New `DATA_DEPS` group with DVC and pandas
   - Added to `extras_require["data"]`
   - Included in `extras_require["full"]`

2. **`.gitignore`** - Added data versioning patterns
   - `.neural_data/` - Main data directory
   - `.neural_data_*/` - Temporary data directories
   - `*.dvc` - DVC files
   - `.dvc/` - DVC cache
   - `.dvc_cache/` - Additional DVC cache

## Directory Structure

```
neural/data/
├── __init__.py                    # Package exports
├── dataset_version.py             # Dataset versioning (380 lines)
├── dvc_integration.py             # DVC integration (240 lines)
├── preprocessing_tracker.py       # Pipeline tracking (280 lines)
├── feature_store.py               # Feature store (270 lines)
├── quality_validator.py           # Quality validation (300 lines)
├── lineage_tracker.py             # Lineage tracking (380 lines)
├── utils.py                       # Utility functions (180 lines)
├── examples.py                    # Runnable examples (380 lines)
├── README.md                      # Full documentation
├── QUICKSTART.md                  # Quick start guide
└── INTEGRATION.md                 # Integration guide

tests/
└── test_data_versioning.py        # Test suite (370 lines)

.neural_data/                      # Data storage (git-ignored)
├── versions.json                  # Version registry
├── datasets/                      # Versioned datasets
├── preprocessing_pipelines/       # Pipeline definitions
├── features/                      # Feature store
├── quality_rules/                 # Validation rules
├── validation_results/            # Validation history
└── lineage/                       # Lineage graphs
```

## Key Features

### 1. Dataset Versioning
- ✅ SHA256 checksums for integrity
- ✅ Automatic/manual version naming
- ✅ Rich metadata and tagging
- ✅ Version comparison
- ✅ Full audit trail
- ✅ Optional data copying

### 2. DVC Integration
- ✅ Seamless DVC workflow
- ✅ Remote storage support
- ✅ Status tracking
- ✅ Git-style diff
- ✅ Works without DVC installed

### 3. Preprocessing Pipeline Tracking
- ✅ Multi-step workflows
- ✅ Parameter logging
- ✅ Pipeline comparison
- ✅ Import/export
- ✅ Execution history

### 4. Feature Store
- ✅ Type-safe features
- ✅ Rich metadata
- ✅ Feature search
- ✅ Usage statistics
- ✅ Import/export

### 5. Data Quality Validation
- ✅ Built-in rules
- ✅ Custom rules
- ✅ Validation history
- ✅ Quality reports
- ✅ Threshold-based validation

### 6. Lineage Tracking & Visualization
- ✅ Graph-based lineage
- ✅ Data→preprocessing→model→predictions flow
- ✅ Bidirectional tracing
- ✅ Graphviz visualization
- ✅ Path finding

## Usage Examples

### Quick Start
```bash
# Version a dataset
neural data version data/train.npy --tags training v1

# List versions
neural data list

# Validate quality
neural data validate data/train.npy --save

# View lineage
neural data lineage my_experiment --visualize
```

### Python API
```python
from neural.data import (
    DatasetVersionManager,
    DataQualityValidator,
    LineageTracker
)

# Version dataset
mgr = DatasetVersionManager()
version = mgr.create_version("data.npy", tags=["v1"])

# Validate
validator = DataQualityValidator()
results = validator.validate(data)

# Track lineage
tracker = LineageTracker()
graph = tracker.create_graph("experiment")
tracker.add_data_node("experiment", "data", "Training Data")
tracker.add_model_node("experiment", "model", "CNN")
tracker.add_edge("experiment", "data", "model", "train")
tracker.visualize_lineage("experiment")
```

### Complete Pipeline
```python
from neural.data.utils import create_data_to_model_lineage

# One-line lineage creation
create_data_to_model_lineage(
    graph_name="mnist_exp",
    dataset_version="v1.0",
    preprocessing_pipeline="standard",
    model_name="cnn_classifier",
    prediction_name="test_predictions"
)
```

## Integration with Neural DSL

The data versioning system integrates seamlessly with Neural DSL:

1. **Before Compilation**: Version and validate training data
2. **During Compilation**: Track model in lineage graph
3. **After Training**: Link predictions back to data sources
4. **Visualization**: Generate complete data→model→predictions diagrams

See `INTEGRATION.md` for detailed integration patterns.

## Testing

Run the test suite:
```bash
# All tests
pytest tests/test_data_versioning.py -v

# Specific test
pytest tests/test_data_versioning.py::test_dataset_versioning -v

# With coverage
pytest tests/test_data_versioning.py --cov=neural.data
```

Run examples:
```bash
python -m neural.data.examples
```

## Dependencies

### Core (always required)
- click (CLI)
- numpy (data handling)
- pyyaml (config files)

### Optional
- dvc >= 2.0 (DVC integration)
- pandas >= 1.3 (CSV data loading)
- graphviz (lineage visualization)

Install with:
```bash
pip install -e ".[data]"              # Core + DVC + pandas
pip install -e ".[data,visualization]" # + graphviz
pip install -e ".[full]"               # Everything
```

## Performance Considerations

1. **Large Datasets**: Use `copy_data=False` or DVC integration
2. **Many Versions**: Periodically clean old versions
3. **Complex Lineage**: Split into multiple graphs
4. **Validation**: Use specific rules instead of all rules

## Future Enhancements

Potential additions (not implemented):
- Delta compression for dataset storage
- Automatic lineage extraction from code
- Web UI for lineage visualization
- Integration with MLflow/Weights & Biases
- Advanced search and filtering
- Dataset diff visualization
- Automatic quality rule suggestion

## Files Changed/Added

### New Files (12)
1. `neural/data/__init__.py`
2. `neural/data/dataset_version.py`
3. `neural/data/dvc_integration.py`
4. `neural/data/preprocessing_tracker.py`
5. `neural/data/feature_store.py`
6. `neural/data/quality_validator.py`
7. `neural/data/lineage_tracker.py`
8. `neural/data/utils.py`
9. `neural/data/examples.py`
10. `neural/data/README.md`
11. `neural/data/QUICKSTART.md`
12. `neural/data/INTEGRATION.md`
13. `tests/test_data_versioning.py`

### Modified Files (3)
1. `neural/cli/cli.py` - Added data CLI commands (270+ lines)
2. `setup.py` - Added DATA_DEPS and data extras
3. `.gitignore` - Added data versioning patterns

## Lines of Code

- Core modules: ~2,430 lines
- Documentation: ~1,200 lines
- Tests: ~370 lines
- Examples: ~380 lines
- CLI integration: ~270 lines
- **Total: ~4,650 lines**

## Summary

A complete, production-ready data versioning and lineage tracking system has been implemented with:
- ✅ 7 core modules with comprehensive functionality
- ✅ 4 new CLI commands
- ✅ Full DVC integration
- ✅ Comprehensive test coverage
- ✅ Extensive documentation and examples
- ✅ Seamless Neural DSL integration
- ✅ All requested features implemented

The system provides complete data→preprocessing→model→predictions lineage tracking with checksums, quality validation, feature stores, and visualization capabilities.
