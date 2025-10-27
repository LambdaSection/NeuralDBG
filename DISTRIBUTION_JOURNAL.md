## Distribution Journal

### Version 0.3.0-dev
**Date:** October 18, 2025

#### Bug Fixes Applied
- **Parser Module**: Fixed Conv2D layer method to include 'sublayers': [] attribute for consistency with other layer types

#### Test Results Summary
- ✅ **CLI Tests**: 11 passed, 9 skipped
- ✅ **Shape Propagation Tests**: 60 passed
- ⚠️ **Visualization Tests**: 16 failed, 23 passed (known issues with missing pygraphviz dependency and dashboard module attributes)

#### Issues Identified
1. **Missing Dependencies**: pygraphviz not installed, causing TensorFlow visualization tests to fail
2. **Dashboard Module**: TRACE_DATA attribute missing in neural.dashboard.dashboard module
3. **Flask Configuration**: SocketIO thread exception in visualization tests

#### Recommendations
- Install pygraphviz for graph visualization features
- Fix TRACE_DATA attribute in dashboard module
- Resolve Flask-SocketIO configuration warning

#### Test Session Status
- Core functionality (CLI, parser, shape propagation) working correctly
- Visualization features require dependency installation and minor fixes


#### New Parser Fixes (today)
- Alias method naming: added `ModelTransformer.max_pooling2d()` delegating to `maxpooling2d()` to correctly handle alias rule `max_pooling2d`.
- Grammar: relaxed `conv2d: CONV2D("(" [param_style1] ")")` so `Conv2D()` parses and validation triggers in the transformer with precise error message.
- Dense behavior: allow `Dense()` with `params=None` (no error); enforce that string units (e.g., `Dense("10")`) raise "Dense units must be a number"; preserve negative-units error message.

#### Targeted Verification
- Manual checks confirm `MaxPooling2D((2, 2))` now yields `{type: MaxPooling2D, params: {pool_size: (2, 2)}}`.
- Next steps: run full parser suite, then shape propagation and codegen, fixing failures sequentially.
