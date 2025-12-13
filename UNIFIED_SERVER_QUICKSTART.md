# Unified Server Quick Start

## What Changed?

Neural DSL v0.3.0 introduces a **unified server** that consolidates multiple dashboards and services into a single interface. Instead of running separate servers on different ports, you now have one entry point with tabbed navigation.

## Before vs After

### Before (Multiple Servers)
```bash
# Terminal 1: Debug dashboard
python neural/dashboard/dashboard.py          # Port 8050

# Terminal 2: No-code builder
python neural/no_code/no_code.py              # Port 8051

# Terminal 3: Production monitoring
python neural/monitoring/dashboard.py         # Port 8052

# Terminal 4: API server
python neural/aquarium/backend/server.py      # Port 8000
```

**Problems**:
- 4 different terminals
- 4 different ports to remember
- Inconsistent UI across dashboards
- High maintenance burden

### After (Unified Server)
```bash
# Single terminal: Everything in one place
neural server start                            # Port 8050
```

**Benefits**:
- ✅ One command to start everything
- ✅ Single port (8050)
- ✅ Unified navigation (tabs)
- ✅ Consistent UI/UX
- ✅ Feature toggles
- ✅ Easier maintenance

## Installation

### Option 1: Full Installation (Recommended)
```bash
pip install -e ".[full]"
```

### Option 2: Dashboard Only
```bash
pip install -e .
pip install dash plotly flask dash-bootstrap-components
```

### Option 3: Specific Features
```bash
# Core + Debug Dashboard
pip install -e . && pip install dash plotly flask

# Core + No-Code Builder
pip install -e . && pip install dash dash-bootstrap-components

# Core + Monitoring
pip install -e . && pip install dash plotly
```

## Usage

### Basic Usage
Start the server with all enabled features:
```bash
neural server start
```

The server will:
1. Start on `http://localhost:8050`
2. Open your default browser automatically
3. Display all enabled features in tabs

### Custom Port
```bash
neural server start --port 9000
```

### Specific Features Only
```bash
# Only debug and monitoring
neural server start --features debug,monitoring

# Only no-code builder
neural server start --features nocode
```

### Without Auto-Browser
```bash
neural server start --no-browser
```

### Custom Host
```bash
neural server start --host 0.0.0.0 --port 8080
```

## Features

### 🐛 Debug Tab
Real-time execution monitoring (formerly NeuralDbg on port 8050)
- Layer-by-layer execution tracking
- Resource monitoring (CPU, memory, GPU)
- Performance profiling
- Gradient flow visualization

### 🏗️ Build Tab
Visual model builder (formerly No-Code Interface on port 8051)
- Drag-and-drop layer palette
- Visual architecture editor
- Real-time shape propagation
- Code generation (TensorFlow, PyTorch, ONNX)

### 📊 Monitor Tab
Production monitoring (formerly Monitoring Dashboard on port 8052)
- Model performance metrics
- Data drift detection
- Quality monitoring
- SLO compliance tracking

### ⚙️ Settings Tab
Configuration and feature management
- Enable/disable features
- Server information
- System status

## Configuration

### Method 1: Environment Variables
```bash
# Enable/disable features
export NEURAL_FEATURE_DEBUG=true
export NEURAL_FEATURE_NOCODE=true
export NEURAL_FEATURE_MONITORING=true

# Server settings
export NEURAL_SERVER_HOST=localhost
export NEURAL_SERVER_PORT=8050

# Start server
neural server start
```

### Method 2: Configuration File
Edit `neural/config/features.yaml`:
```yaml
extended:
  debug_dashboard: true
  hpo: true
  automl: true

experimental:
  nocode_builder: true
  monitoring: true
  collaboration: false

server:
  unified:
    enabled: true
    host: localhost
    port: 8050
```

### Method 3: Command Line
```bash
# Override with CLI flags
neural server start --port 9000 --features debug,nocode
```

## Feature Flags

Control which features are available:

```bash
# All features enabled (default)
neural server start

# Only debug
export NEURAL_FEATURE_NOCODE=false
export NEURAL_FEATURE_MONITORING=false
neural server start

# Only specific features
neural server start --features debug,monitoring
```

## Use Cases

### Development Workflow
```bash
# Start server with debug and build features
neural server start --features debug,nocode

# Open browser to http://localhost:8050
# Switch between Debug and Build tabs as needed
```

### Production Monitoring
```bash
# Start with monitoring only
export NEURAL_FEATURE_MONITORING=true
export NEURAL_FEATURE_DEBUG=false
export NEURAL_FEATURE_NOCODE=false
neural server start --port 8080
```

### Team Demo
```bash
# Start with all features
neural server start

# Show:
# 1. Build tab - create model visually
# 2. Debug tab - monitor execution
# 3. Monitor tab - production metrics
```

## Troubleshooting

### Port Already in Use
```bash
# Error: Address already in use
# Solution: Use different port
neural server start --port 9000
```

### Feature Not Available
```bash
# If tab is grayed out, feature is disabled
# Check configuration:
cat neural/config/features.yaml

# Enable feature:
export NEURAL_FEATURE_DEBUG=true
neural server start
```

### Missing Dependencies
```bash
# Error: ImportError: No module named 'dash'
# Solution: Install dependencies
pip install -e ".[dashboard]"

# Or install specific packages:
pip install dash plotly flask dash-bootstrap-components
```

### Browser Not Opening
```bash
# Use --no-browser and open manually
neural server start --no-browser

# Then open: http://localhost:8050
```

## Migration from Legacy

If you have scripts using the old separate dashboards:

### Old Script
```python
# legacy_start.py
import subprocess

subprocess.Popen(['python', 'neural/dashboard/dashboard.py'])
subprocess.Popen(['python', 'neural/no_code/no_code.py'])
subprocess.Popen(['python', 'neural/monitoring/dashboard.py'])
```

### New Script
```python
# unified_start.py
import subprocess

# Single process
subprocess.Popen(['neural', 'server', 'start'])
```

### Old Bookmarks
Update your bookmarks:
- ~~http://localhost:8050~~ (Debug) → http://localhost:8050 (Unified, Debug tab)
- ~~http://localhost:8051~~ (No-Code) → http://localhost:8050 (Unified, Build tab)
- ~~http://localhost:8052~~ (Monitoring) → http://localhost:8050 (Unified, Monitor tab)

## FAQ

**Q: Can I still use the old separate dashboards?**
A: Yes, but they're deprecated and will be removed in v0.4.0. Migration is recommended.

**Q: What if I only want one feature?**
A: Use `--features` flag: `neural server start --features debug`

**Q: How do I enable/disable features permanently?**
A: Edit `neural/config/features.yaml` or set environment variables in your shell profile.

**Q: Does this affect the CLI commands?**
A: No, `neural compile`, `neural run`, etc. work the same way.

**Q: Can I run multiple instances?**
A: Yes, use different ports: `neural server start --port 9000`

**Q: What about Docker/cloud deployment?**
A: Use: `neural server start --host 0.0.0.0 --port 8050`

**Q: How do I check what features are enabled?**
A: Start the server and check the Settings tab, or run `neural server start --help`

## Next Steps

1. **Try the unified server**: `neural server start`
2. **Explore each tab**: Debug → Build → Monitor → Settings
3. **Customize features**: Edit `neural/config/features.yaml`
4. **Update bookmarks**: Single URL for all features
5. **Update scripts**: Replace separate dashboard launches

## Support

- **Documentation**: [SERVICE_REGISTRY.md](./SERVICE_REGISTRY.md)
- **Configuration**: [neural/config/features.yaml](./neural/config/features.yaml)
- **Issues**: GitHub Issues
- **Questions**: GitHub Discussions

---

**Note**: The unified server is the recommended way to use Neural DSL's web features. Legacy separate dashboards will be removed in v0.4.0.
