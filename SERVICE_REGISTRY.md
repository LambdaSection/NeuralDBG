# Neural DSL Service Registry

## Overview
This document describes all services in the Neural DSL project, their ports, status, and consolidation plan.

## Unified Architecture (Recommended)

### Primary Service
**Unified Web Interface** - Single entry point for all web features
- **Port**: 8050
- **Command**: `neural server start`
- **Features**:
  - Debug Tab: Real-time execution monitoring (NeuralDbg)
  - Build Tab: No-code visual model builder
  - Monitor Tab: Production monitoring and alerting
  - Settings Tab: Feature toggles and configuration
- **Dependencies**: dash, plotly, flask, dash-bootstrap-components
- **Status**: ✅ Active (v0.3.0+)

### Secondary Services
**REST API** - Backend services and integrations
- **Port**: 8000
- **Command**: `python neural/aquarium/backend/server.py`
- **Purpose**: REST API for Aquarium IDE backend
- **Status**: ⚠️ Will be extracted to `neural-aquarium` package

**WebSocket Server** - Real-time features
- **Port**: 8080
- **Command**: `python neural/collaboration/server.py`
- **Purpose**: Real-time collaborative editing
- **Status**: ⚠️ Optional, disabled by default

## Legacy Architecture (Deprecated)

### Individual Dashboards (To Be Removed in v0.4.0)
These separate servers are deprecated in favor of the unified interface:

1. **NeuralDbg Dashboard** (DEPRECATED)
   - Port: 8050
   - File: `neural/dashboard/dashboard.py`
   - Replacement: Use `neural server start` → Debug tab
   - Migration: Automatic, same port

2. **No-Code Interface** (DEPRECATED)
   - Port: 8051
   - File: `neural/no_code/no_code.py`
   - Replacement: Use `neural server start` → Build tab
   - Migration: Integrated into unified dashboard

3. **Monitoring Dashboard** (DEPRECATED)
   - Port: 8052
   - File: `neural/monitoring/dashboard.py`
   - Replacement: Use `neural server start` → Monitor tab
   - Migration: Integrated into unified dashboard

4. **Aquarium Backend** (TO BE EXTRACTED)
   - Port: 8000
   - File: `neural/aquarium/backend/server.py`
   - Replacement: Separate `neural-aquarium` package
   - Migration: Install separately when needed

5. **Collaboration Server** (OPTIONAL)
   - Port: 8080
   - File: `neural/collaboration/server.py`
   - Status: Disabled by default, enable with feature flag
   - Migration: Enable via `NEURAL_FEATURE_COLLABORATION=true`

## Port Allocation

### Current Allocation
```
8050 - Unified Web Interface (primary)
8000 - REST API (Aquarium backend, to be extracted)
8080 - WebSocket Server (optional, disabled by default)
5001 - Internal WebSocket (dashboard, deprecated)
```

### Future Allocation (v0.4.0+)
```
8050 - Unified Web Interface (all web features)
8000 - Optional REST API (if needed for integrations)
8080 - Optional WebSocket (if real-time features enabled)
```

## Feature Status

### Core Features (Always Active)
- ✅ DSL Parser
- ✅ Code Generation
- ✅ Shape Propagation
- ✅ CLI Interface

### Extended Features (Active by Default)
- ✅ Debug Dashboard (unified interface)
- ✅ HPO (Hyperparameter Optimization)
- ✅ AutoML (Neural Architecture Search)
- ⚠️ Integrations (optional, disabled by default)
- ⚠️ Teams Module (optional, disabled by default)

### Experimental Features (Optional)
- ✅ No-Code Builder (unified interface)
- ✅ Monitoring (unified interface)
- ❌ Collaboration (disabled by default)
- ❌ Marketplace (disabled by default)
- ⚠️ Aquarium IDE (to be extracted)
- ❌ Federated Learning (disabled by default)

## Starting Services

### Unified Server (Recommended)
```bash
# Start with all enabled features
neural server start

# Start with specific features
neural server start --features debug,nocode,monitoring

# Start on custom port
neural server start --port 9000

# Start without opening browser
neural server start --no-browser
```

### Individual Services (Legacy, Deprecated)
```bash
# Debug dashboard (deprecated - use unified server)
python neural/dashboard/dashboard.py

# No-code interface (deprecated - use unified server)
python neural/no_code/no_code.py

# Monitoring (deprecated - use unified server)
python neural/monitoring/dashboard.py

# Aquarium backend (will be extracted)
python neural/aquarium/backend/server.py

# Collaboration server (optional)
python neural/collaboration/server.py
```

## Configuration

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

server:
  unified:
    enabled: true
    host: localhost
    port: 8050
```

## Dependencies by Feature

### Unified Server
```bash
pip install -e ".[dashboard]"
```
Includes: dash, plotly, flask, dash-bootstrap-components

### Debug Dashboard
```bash
pip install dash plotly flask
pip install psutil pysnooper  # optional
```

### No-Code Builder
```bash
pip install dash dash-bootstrap-components plotly
```

### Monitoring
```bash
pip install dash plotly
pip install prometheus-client  # optional
```

### HPO
```bash
pip install optuna scikit-learn
```

### AutoML
```bash
pip install optuna scikit-learn scipy
pip install ray dask  # optional for distributed
```

### Collaboration
```bash
pip install websockets aiohttp
```

## Health Checks

### Check Unified Server
```bash
curl http://localhost:8050/health
```

### Check API Server
```bash
curl http://localhost:8000/health
```

### Check Features
```bash
neural server start --help
```

## Migration Guide

### From Separate Dashboards to Unified Server

**Before (v0.2.x)**:
```bash
# Terminal 1
python neural/dashboard/dashboard.py

# Terminal 2
python neural/no_code/no_code.py

# Terminal 3
python neural/monitoring/dashboard.py
```

**After (v0.3.0+)**:
```bash
# Single terminal
neural server start
```

### Accessing Features

**Before**:
- Debug: http://localhost:8050
- No-Code: http://localhost:8051
- Monitoring: http://localhost:8052

**After**:
- All features: http://localhost:8050
  - Debug tab: Click "Debug" in navigation
  - Build tab: Click "Build" in navigation
  - Monitor tab: Click "Monitor" in navigation

## Troubleshooting

### Port Already in Use
```bash
# Check what's using port 8050
lsof -i :8050  # macOS/Linux
netstat -ano | findstr :8050  # Windows

# Use different port
neural server start --port 9000
```

### Feature Not Available
```bash
# Check enabled features
neural server start --help

# Enable specific feature
export NEURAL_FEATURE_DEBUG=true
neural server start
```

### Missing Dependencies
```bash
# Install all dashboard dependencies
pip install -e ".[dashboard]"

# Or install specific feature dependencies
pip install dash plotly flask dash-bootstrap-components
```

## Roadmap

### v0.3.0 (Current)
- ✅ Unified server created
- ✅ Feature registry implemented
- ✅ Configuration system added
- ✅ CLI command `neural server start`

### v0.4.0 (Planned)
- ⏳ Remove legacy dashboard entry points
- ⏳ Extract Aquarium to separate package
- ⏳ Extract Marketplace to separate package
- ⏳ Consolidate WebSocket services

### v0.5.0 (Future)
- ⏳ Extract Federated Learning to separate package
- ⏳ Plugin system for external features
- ⏳ Service mesh for microservices deployment

## Support

For issues or questions:
- Check feature status: `neural server start --help`
- View configuration: `cat neural/config/features.yaml`
- Enable debug logging: `export NEURAL_LOG_LEVEL=DEBUG`
- Report issues: GitHub Issues

## Related Documentation
- [SCOPE_CONSOLIDATION.md](./SCOPE_CONSOLIDATION.md) - Consolidation strategy
- [AGENTS.md](./AGENTS.md) - Development guide
- [README.md](./README.md) - Project overview
