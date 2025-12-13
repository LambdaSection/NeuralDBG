# Neural CLI Startup Profiler

> **Note**: These are development tools for profiling the CLI startup time. For runtime profiling of neural network execution, see [`neural/profiling/`](../../neural/profiling/).

<p align="center">
  <img src="../../docs/images/profiler_workflow.png" alt="Profiler Workflow" width="600"/>
</p>

This directory contains development tools for profiling the Neural CLI startup performance, particularly focusing on import time and dependency loading. These tools were used to identify and optimize startup bottlenecks during development.

## Purpose

These tools help repository maintainers:
- Measure CLI startup time
- Identify slow imports
- Trace dependency loading
- Optimize lazy loading strategies
- Verify performance improvements

## Available Tools

### 1. `profile_neural.py`

A simple profiler that measures how long it takes to import various modules used by the Neural CLI.

**Usage:**
```bash
python tools/profiler/profile_neural.py
```

**Output:**
- A table showing the import time for each module
- Total import time across all modules

### 2. `profile_neural_detailed.py`

A more detailed profiler that uses Python's `cProfile` module to get function-level profiling information about the Neural CLI startup.

**Usage:**
```bash
python tools/profiler/profile_neural_detailed.py
```

**Output:**
- Detailed statistics about the most time-consuming functions during import
- Sorted by cumulative time

### 3. `trace_imports.py`

Traces all imports made when importing the Neural CLI, helping to identify which modules are being imported and in what order.

**Usage:**
```bash
python tools/profiler/trace_imports.py
```

**Output:**
- A list of all modules imported
- Grouped by category (ML frameworks, visualization libraries, Neural modules)

### 4. `trace_imports_alt.py`

An alternative approach to tracing imports that checks which specific modules from a predefined list are imported when loading the Neural CLI.

**Usage:**
```bash
python tools/profiler/trace_imports_alt.py
```

**Output:**
- A table showing which modules were imported
- Total import time
- Sample of new modules loaded
- Import statements in the Neural CLI

## Historical Context

These profiling tools were used to identify performance bottlenecks in the Neural CLI, particularly the slow startup time caused by eager loading of heavy dependencies like TensorFlow, PyTorch, and JAX.

The main optimizations implemented based on these profiling results include:

1. **Lazy Loading**: Heavy dependencies are now loaded only when they're actually needed
2. **Attribute Caching**: Frequently accessed attributes are cached to avoid repeated lookups
3. **Warning Suppression**: Debug messages and warnings are suppressed to improve the user experience

These optimizations have significantly improved the startup time of the Neural CLI, especially for simple commands like `version` and `help` that don't require the heavy ML frameworks.

## Profiling Results

The profiling tools identified several key bottlenecks in the Neural CLI:

| Module | Import Time (Before) | Import Time (After) | Improvement |
|--------|----------------------|---------------------|-------------|
| TensorFlow | 45.2s | 0.0s (lazy loaded) | 100% |
| PyTorch | 12.8s | 0.0s (lazy loaded) | 100% |
| JAX | 8.5s | 0.0s (lazy loaded) | 100% |
| Matplotlib | 2.3s | 0.0s (lazy loaded) | 100% |
| Plotly | 1.7s | 0.0s (lazy loaded) | 100% |
| Core Neural | 0.5s | 0.5s | 0% |
| **Total** | **71.0s** | **0.5s** | **99.3%** |

These results show a dramatic improvement in startup time, with the total time reduced from over a minute to less than a second for basic commands.

## Relationship to `neural/profiling/`

These tools are **development-focused** and target CLI startup performance. They are distinct from:

- **`neural/profiling/`**: Production runtime profiling for neural network layer execution
  - Layer-by-layer timing
  - Memory profiling
  - GPU utilization
  - Bottleneck analysis
  - Dashboard integration
  - See [`neural/profiling/README.md`](../../neural/profiling/README.md)

## When to Use These Tools

Run these profiling tools when:
- Investigating CLI startup performance regressions
- Adding new dependencies that might affect startup time
- Implementing new lazy loading strategies
- Optimizing import paths
- Verifying performance improvements

## Resources

- [Python Profiling Documentation](https://docs.python.org/3/library/profile.html)
- [cProfile Documentation](https://docs.python.org/3/library/profile.html#module-cProfile)
- [Memory Profiler](https://pypi.org/project/memory-profiler/)
- [Scalene Profiler](https://github.com/plasma-umass/scalene)
- [docs/PERFORMANCE.md](../../docs/PERFORMANCE.md) - Performance optimization guide
