# Neural DSL Quick Reference

## Essential Commands

### Starting the Server
```bash
# Unified server (recommended)
neural server start                          # All features, port 8050
neural server start --port 9000              # Custom port
neural server start --features debug,nocode  # Specific features
neural server start --no-browser             # Don't auto-open browser

# Legacy (deprecated)
python neural/dashboard/dashboard.py         # Port 8050 (deprecated)
python neural/no_code/no_code.py            # Port 8051 (deprecated)
python neural/monitoring/dashboard.py       # Port 8052 (deprecated)
```

### Compilation
```bash
neural compile model.neural                  # Compile to all backends
neural compile model.neural --backend tensorflow
neural compile model.neural --backend pytorch
neural compile model.neural --backend onnx
```

### Execution
```bash
neural run model.neural                      # Compile and execute
neural run model.neural --backend pytorch
neural run model.neural --dataset MNIST
```

### Visualization
```bash
neural visualize model.neural                # Generate architecture diagram
neural visualize model.neural --format svg
neural visualize model.neural --format png
```

## Feature Flags

### Environment Variables
```bash
# Enable/disable features
export NEURAL_FEATURE_DEBUG=true
export NEURAL_FEATURE_NOCODE=true
export NEURAL_FEATURE_MONITORING=true
export NEURAL_FEATURE_COLLABORATION=false
export NEURAL_FEATURE_MARKETPLACE=false

# Server configuration
export NEURAL_SERVER_HOST=localhost
export NEURAL_SERVER_PORT=8050
```

### Configuration File
Edit `neural/config/features.yaml`:
```yaml
extended:
  debug_dashboard: true
  hpo: true
  automl: true
  integrations: false
  teams: false

experimental:
  nocode_builder: true
  monitoring: true
  collaboration: false
  marketplace: false
  aquarium: false
  federated: false
```

## Installation Profiles

```bash
# Minimal - Core only
pip install -e .

# Standard - Core + common features
pip install -e ".[full]"

# Development - Everything
pip install -e ".[full]" && pip install -r requirements-dev.txt

# Specific features
pip install -e ".[hpo]"        # HPO only
pip install -e ".[automl]"     # AutoML only
pip install -e ".[dashboard]"  # Dashboard features
```

## Port Reference

| Port | Service              | Status      |
|------|---------------------|-------------|
| 8050 | Unified Interface   | ✅ Active   |
| 8000 | REST API (optional) | ⚠️ Optional |
| 8080 | WebSocket (optional)| ⚠️ Optional |
| 8051 | No-Code (legacy)    | ❌ Deprecated|
| 8052 | Monitoring (legacy) | ❌ Deprecated|

## URL Reference

### Unified Server (Current)
```
http://localhost:8050          - Main interface
http://localhost:8050/#debug   - Debug tab
http://localhost:8050/#build   - Build tab
http://localhost:8050/#monitor - Monitor tab
http://localhost:8050/health   - Health check
```

### Legacy (Deprecated)
```
http://localhost:8050          - Debug dashboard (deprecated)
http://localhost:8051          - No-code builder (deprecated)
http://localhost:8052          - Monitoring (deprecated)
```

## Feature Status

### Core (Always Enabled)
- ✅ Parser
- ✅ Code Generation
- ✅ Shape Propagation
- ✅ CLI

### Extended (Default Enabled)
- ✅ Debug Dashboard
- ✅ HPO
- ✅ AutoML
- ⚠️ Integrations (optional)
- ⚠️ Teams (optional)

### Experimental (Optional)
- ✅ No-Code Builder
- ✅ Monitoring
- ❌ Collaboration (disabled)
- ❌ Marketplace (disabled)
- ⚠️ Aquarium (extracting)
- ❌ Federated (disabled)

## Common Workflows

### Build and Debug
```bash
# Start unified server
neural server start

# Open browser to http://localhost:8050
# 1. Go to Build tab
# 2. Create model visually
# 3. Generate code
# 4. Switch to Debug tab
# 5. Monitor execution
```

### HPO Workflow
```bash
# Compile with HPO
neural compile model.neural --hpo --trials 50

# Or programmatically
python -c "
from neural.hpo import run_hpo
results = run_hpo('model.neural', n_trials=50)
print(results.best_params)
"
```

### Production Monitoring
```bash
# Start with monitoring only
neural server start --features monitoring --port 8080

# Access at http://localhost:8080
```

## Code Examples

### Using Feature Registry
```python
from neural.config import is_feature_enabled

if is_feature_enabled('debug_dashboard'):
    from neural.dashboard import create_dashboard
    dashboard = create_dashboard()

if is_feature_enabled('hpo'):
    from neural.hpo import run_hpo
    results = run_hpo(model_file, n_trials=100)
```

### Creating Unified App
```python
from neural.server import create_unified_app

app = create_unified_app(port=8050)
app.run_server(host='0.0.0.0', port=8050, debug=False)
```

### Configuration Management
```python
from neural.config import get_config

config = get_config()
port = config.get('server.unified.port', 8050)
enabled = config.get_all_enabled_features()
```

## Troubleshooting

### Port in Use
```bash
# Check port usage
lsof -i :8050              # macOS/Linux
netstat -ano | findstr :8050  # Windows

# Use different port
neural server start --port 9000
```

### Feature Not Available
```bash
# Check enabled features
cat neural/config/features.yaml

# Enable feature
export NEURAL_FEATURE_DEBUG=true
neural server start
```

### Missing Dependencies
```bash
# Install all dashboard dependencies
pip install -e ".[dashboard]"

# Or specific packages
pip install dash plotly flask dash-bootstrap-components
```

### Import Errors
```bash
# Ensure neural is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install in development mode
pip install -e .
```

## Migration Checklist

### From Separate Dashboards
- [ ] Stop running separate dashboard scripts
- [ ] Update bookmarks to use port 8050 only
- [ ] Update scripts to use `neural server start`
- [ ] Remove port-specific hardcoding
- [ ] Test unified interface
- [ ] Update documentation

### To Unified Server
- [ ] Install dashboard dependencies: `pip install -e ".[dashboard]"`
- [ ] Configure features in `neural/config/features.yaml`
- [ ] Start unified server: `neural server start`
- [ ] Verify all tabs work (Debug, Build, Monitor)
- [ ] Update team documentation
- [ ] Archive old dashboard scripts

## Documentation Links

- **Quick Start**: [UNIFIED_SERVER_QUICKSTART.md](./UNIFIED_SERVER_QUICKSTART.md)
- **Architecture**: [ARCHITECTURE.md](./ARCHITECTURE.md)
- **Services**: [SERVICE_REGISTRY.md](./SERVICE_REGISTRY.md)
- **Consolidation**: [SCOPE_CONSOLIDATION.md](./SCOPE_CONSOLIDATION.md)
- **Development**: [AGENTS.md](./AGENTS.md)

## Getting Help

```bash
# CLI help
neural --help
neural server --help
neural compile --help

# Check version and dependencies
neural version

# View configuration
cat neural/config/features.yaml

# Check server health
curl http://localhost:8050/health
```

## Version Information

- **Current**: v0.3.0
- **Unified Server**: ✅ Available
- **Legacy Dashboards**: ⚠️ Deprecated, removal in v0.4.0
- **Next Release**: v0.4.0 (feature extraction)

---

**Remember**: Use `neural server start` for all web features. Legacy separate dashboards are deprecated and will be removed in v0.4.0.
