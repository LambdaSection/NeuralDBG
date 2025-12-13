# Neural Development Tools

This directory contains development and maintenance tools for Neural DSL repository maintainers and contributors.

## Overview

These tools are intended for:
- Repository maintenance and debugging
- Development workflow optimization
- Performance analysis during development
- Internal testing and validation

**Note**: These are not user-facing tools. For runtime profiling and debugging, see [`neural/profiling/`](../neural/profiling/) and [`neural/dashboard/`](../neural/dashboard/).

## Available Tools

### profiler/

CLI startup time profiling tools for analyzing and optimizing Neural CLI import performance.

**Purpose**: Measure and optimize CLI startup time by analyzing import behavior and dependency loading.

**Tools:**
- `profile_neural.py` - Simple import time profiler
- `profile_neural_detailed.py` - Detailed cProfile profiling
- `trace_imports.py` - Import tracing tool
- `trace_imports_alt.py` - Alternative import tracing

**Documentation**: See [`profiler/README.md`](profiler/README.md)

**When to use**:
- Investigating CLI startup performance regressions
- Adding new dependencies that might affect startup time
- Implementing lazy loading strategies
- Verifying performance improvements

### cli_invoke.py

CLI testing helper that invokes CLI commands and captures output to a file.

**Purpose**: Programmatic CLI testing without subprocess overhead.

**Usage:**
```bash
python tools/cli_invoke.py output.txt --version
python tools/cli_invoke.py output.txt compile model.neural --backend tensorflow
```

**Features:**
- Captures exit code, stdout, and exceptions
- Uses Click's CliRunner for faster execution
- Useful for automated testing scripts

### import_check.py

Dependency verification tool that checks if required modules can be imported.

**Purpose**: Quick sanity check for dependency installation.

**Usage:**
```bash
python tools/import_check.py
cat import_check.txt
```

**Checks:**
- click, lark, numpy, plotly, torch, tensorflow, onnx
- Reports OK or error details for each module
- Outputs results to `import_check.txt`

## Adding New Tools

When adding new development tools:

1. **Add the tool** to this directory
2. **Document it** in this README
3. **Add usage examples** in the tool's docstring
4. **Consider** if it should be in a subdirectory (like `profiler/`)
5. **Update** `.gitignore` if the tool generates files

## Tool Categories

### Performance Analysis
- `profiler/` - CLI startup profiling

### Testing Utilities
- `cli_invoke.py` - CLI command testing
- `import_check.py` - Dependency verification

### Future Additions

Potential tools that could be added:

- **Code Quality**
  - Dead code detector
  - Import organizer
  - Code complexity analyzer

- **Documentation**
  - Docstring coverage checker
  - Documentation link validator
  - README synchronizer

- **Build & Release**
  - Changelog generator
  - Version bumper
  - Release checklist validator

- **Analysis**
  - Dependency tree analyzer
  - Module size reporter
  - Test coverage visualizer

## Related Documentation

- [docs/PERFORMANCE.md](../docs/PERFORMANCE.md) - Performance optimization guide
- [REPOSITORY_STRUCTURE.md](../REPOSITORY_STRUCTURE.md) - Repository structure overview
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution guidelines
- [PROFILER_CONSOLIDATION.md](../PROFILER_CONSOLIDATION.md) - Profiler directory consolidation notes

## Distinction from Production Code

| Aspect | tools/ | neural/ |
|--------|--------|---------|
| **Audience** | Maintainers, contributors | End users |
| **Purpose** | Development, maintenance | Production functionality |
| **Testing** | Manual, ad-hoc | Automated test suite |
| **Documentation** | Internal notes | User-facing docs |
| **Stability** | Can change freely | API stability required |
| **Dependencies** | Can use dev-only deps | Production deps only |

## Contributing

When contributing to this directory:

- Keep tools simple and focused
- Add clear usage documentation
- Consider whether the tool should be a script or module
- Avoid dependencies outside of dev requirements
- Add examples to help future maintainers
