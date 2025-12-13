# Neural DSL Benchmarking Suite

Comprehensive benchmarking suite for comparing Neural DSL against popular ML frameworks including Keras, PyTorch Lightning, Fast.ai, and Ludwig.

## Overview

This benchmarking suite measures and compares:

- **Development Time**: Time to write and compile models
- **Lines of Code**: Code complexity and verbosity
- **Training Speed**: Time to train models
- **Inference Performance**: Prediction latency
- **Model Accuracy**: Classification/regression performance
- **Code Readability**: Subjective code quality metrics
- **Setup Complexity**: Number of imports, classes, and functions required

## Installation

### Core Requirements

```bash
pip install -e ".[full]"
```

### Optional Framework Dependencies

For complete benchmarking, install competitor frameworks:

```bash
# PyTorch Lightning
pip install pytorch-lightning

# Fast.ai
pip install fastai

# Ludwig
pip install ludwig
```

## Usage

### Quick Start

Run all benchmarks with default settings:

```bash
python neural/benchmarks/run_benchmarks.py
```

### Custom Configuration

```bash
# Benchmark specific frameworks
python neural/benchmarks/run_benchmarks.py --frameworks neural keras pytorch-lightning

# Adjust training parameters
python neural/benchmarks/run_benchmarks.py --epochs 10 --batch-size 64

# Specify output directories
python neural/benchmarks/run_benchmarks.py --output-dir results --report-dir reports

# Skip plot generation for faster execution
python neural/benchmarks/run_benchmarks.py --skip-plots

# Quiet mode
python neural/benchmarks/run_benchmarks.py --quiet
```

### Programmatic Usage

```python
from neural.benchmarks import (
    BenchmarkRunner,
    NeuralDSLImplementation,
    KerasImplementation,
    ReportGenerator,
)

# Initialize frameworks
frameworks = [
    NeuralDSLImplementation(),
    KerasImplementation(),
]

# Configure benchmark tasks
tasks = [{
    "name": "MNIST_Classification",
    "dataset": "mnist",
    "epochs": 5,
    "batch_size": 32,
}]

# Run benchmarks
runner = BenchmarkRunner(output_dir="my_results")
results = runner.run_all_benchmarks(frameworks, tasks)

# Generate report
report_gen = ReportGenerator(output_dir="my_reports")
report_path = report_gen.generate_report([r.to_dict() for r in results])
print(f"Report: {report_path}")
```

## Output Structure

```
benchmark_results/
├── benchmark_results_20240101_120000.json

benchmark_reports/
└── neural_dsl_benchmark_20240101_120000/
    ├── index.html                    # Interactive HTML report
    ├── README.md                     # Markdown summary
    ├── raw_data.json                 # Raw benchmark data
    ├── reproduce.py                  # Reproducibility script
    ├── comparison_overview.png       # Overall comparison chart
    ├── lines_of_code.png            # LOC comparison
    ├── training_time_seconds.png    # Training time comparison
    ├── inference_time_ms.png        # Inference latency comparison
    ├── model_accuracy.png           # Accuracy comparison
    └── ...                          # Additional metric plots
```

## Metrics Explained

### Lines of Code (LOC)
Total non-empty, non-comment lines required to define and compile a model.
**Lower is better** - indicates conciseness and ease of use.

### Development Time
Time from starting to write code to having a compiled model ready for training.
**Lower is better** - indicates faster prototyping.

### Training Speed
Wall-clock time to train the model for a fixed number of epochs.
**Lower is better** - indicates computational efficiency.

### Inference Time
Average prediction latency for a single sample.
**Lower is better** - critical for production deployment.

### Model Accuracy
Classification accuracy on held-out test set.
**Higher is better** - indicates model quality.

### Setup Complexity
Number of imports, classes, and functions required.
**Lower is better** - indicates simplicity.

### Code Readability
Subjective score (0-10) based on code length and structure.
**Higher is better** - indicates maintainability.

## Reproducibility

Each benchmark report includes a `reproduce.py` script that can recreate the exact benchmarks:

```bash
cd benchmark_reports/neural_dsl_benchmark_TIMESTAMP/
python reproduce.py
```

## Customization

### Adding New Frameworks

Create a new implementation by subclassing `FrameworkImplementation`:

```python
from neural.benchmarks.framework_implementations import FrameworkImplementation

class MyFrameworkImplementation(FrameworkImplementation):
    def __init__(self):
        super().__init__("MyFramework")
    
    def setup(self):
        # Initialize framework-specific code
        self.code_content = "..."
    
    def build_model(self):
        # Build the model
        self.model = ...
    
    def train(self, dataset, epochs, batch_size):
        # Train and return metrics
        return {
            "training_time": ...,
            "accuracy": ...,
            ...
        }
    
    def predict_single(self):
        # Make a single prediction
        return self.model.predict(...)
    
    def _save_model(self, path):
        # Save model to disk
        ...
    
    def get_parameter_count(self):
        # Return total parameters
        return ...
```

### Adding New Tasks

Define new benchmark tasks:

```python
tasks = [
    {
        "name": "CIFAR10_Classification",
        "dataset": "cifar10",
        "epochs": 10,
        "batch_size": 128,
    },
    {
        "name": "IMDB_Sentiment",
        "dataset": "imdb",
        "epochs": 5,
        "batch_size": 32,
    }
]
```

## Publishing Results

### To GitHub Pages

```bash
# Generate report
python neural/benchmarks/run_benchmarks.py

# Copy to docs/benchmarks
cp -r benchmark_reports/neural_dsl_benchmark_TIMESTAMP docs/benchmarks/latest

# Commit and push
git add docs/benchmarks/latest
git commit -m "Update benchmark results"
git push
```

### To Website

The generated HTML reports are self-contained and can be hosted on any static web server:

```bash
# Copy report directory to web server
scp -r benchmark_reports/neural_dsl_benchmark_TIMESTAMP user@server:/var/www/html/benchmarks/
```

## Notes

- Benchmarks use a subset of training data for faster execution
- Results may vary based on hardware and system load
- GPU availability is automatically detected and used when possible
- All frameworks are benchmarked on the same hardware for fair comparison

## Contributing

To add new benchmarks or improve existing ones:

1. Fork the repository
2. Add your benchmark implementation
3. Test with `python neural/benchmarks/run_benchmarks.py`
4. Submit a pull request with results

## License

MIT License - See LICENSE file for details
