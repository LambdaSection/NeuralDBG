# Scope Consolidation Plan

## Problem Statement
The Neural DSL project has grown to include numerous services running on different ports:
- **Port 8050**: NeuralDbg Dashboard (neural/dashboard/dashboard.py)
- **Port 8051**: No-Code Interface (neural/no_code/no_code.py)
- **Port 8052**: Monitoring Dashboard (neural/monitoring/dashboard.py)
- **Port 8080**: Collaboration Server (neural/collaboration/server.py)
- **Port 8000**: Aquarium Backend API (neural/aquarium/backend/server.py)
- **Port 5001**: WebSocket Server (dashboard internal)

This creates complexity, maintenance burden, and unclear boundaries between features.

## Core vs. Optional Features

### Core Features (Essential)
These are fundamental to Neural DSL's value proposition:
1. **DSL Parser & Compiler** - The foundation of the project
2. **Code Generation** - TensorFlow, PyTorch, ONNX output
3. **Shape Propagation** - Static analysis and validation
4. **CLI Interface** - Command-line tools for compilation
5. **Basic Visualization** - Static architecture diagrams

### Extended Features (Useful but Optional)
These add value but aren't strictly necessary:
1. **Real-time Debugging** - NeuralDbg dashboard for execution monitoring
2. **HPO** - Hyperparameter optimization with Optuna
3. **AutoML** - Neural architecture search
4. **Integrations** - Cloud platform connectors
5. **Teams Module** - Multi-tenancy and RBAC

### Experimental Features (Consider Removal/Separation)
These significantly increase complexity:
1. **No-Code Interface** - Drag-and-drop model builder
2. **Collaboration Server** - Real-time multi-user editing
3. **Aquarium IDE** - Full IDE experience
4. **Marketplace** - Model sharing and distribution
5. **Federated Learning** - Distributed training infrastructure

## Consolidation Strategy

### Phase 1: Unified Server Architecture
Create a single entry point that orchestrates all services:

**File**: `neural/server.py`
```python
# Unified server that runs all services on configurable ports
# - Consolidates dashboard, monitoring, API into one FastAPI app
# - Makes services pluggable and optional
# - Provides service registry and health checks
```

**Ports**:
- **8050**: Main Web Interface (combines dashboard, no-code, monitoring)
- **8000**: REST API (consolidates all API endpoints)
- **8080**: WebSocket Server (real-time features like collaboration)

### Phase 2: Modular Dashboard
Combine multiple dashboards into a single tabbed interface:

**Structure**:
```
Main Dashboard (Port 8050)
├── Debug Tab (NeuralDbg - real-time execution)
├── Build Tab (No-Code interface)
├── Monitor Tab (Production monitoring)
└── Admin Tab (Teams, quotas, analytics)
```

### Phase 3: Feature Flags
Use configuration to enable/disable optional features:

**File**: `neural/config.yaml` or environment variables
```yaml
features:
  core:
    enabled: true  # Always enabled
  
  extended:
    debug_dashboard: true
    hpo: true
    automl: true
    integrations: false
    teams: false
  
  experimental:
    no_code: false
    collaboration: false
    aquarium: false
    marketplace: false
    federated: false
```

### Phase 4: Extract to Separate Projects
Move experimental features to their own repositories:

1. **neural-aquarium** - Full IDE experience (separate repo)
2. **neural-marketplace** - Model marketplace (separate repo)
3. **neural-federated** - Federated learning toolkit (separate repo)

Keep in `neural-dsl`:
- Core DSL and compilation
- Extended features (HPO, AutoML, integrations)
- Unified dashboard for debugging and monitoring

## Implementation Plan

### Step 1: Create Unified Server (`neural/server.py`)
- Single FastAPI app with all endpoints
- Service registry for pluggable features
- Configuration-based feature activation
- Unified logging and error handling

### Step 2: Consolidate Dashboards (`neural/dashboard/unified_dashboard.py`)
- Merge NeuralDbg, No-Code, and Monitoring into tabs
- Shared navigation and state management
- Consistent theming and UI patterns

### Step 3: Configuration System (`neural/config/`)
- Central configuration management
- Environment variable support
- Feature flags and service settings

### Step 4: Documentation Update
- Clear distinction between core and optional
- Installation profiles: minimal, full, dev
- Service architecture diagram

### Step 5: Deprecation Path
- Mark experimental features as deprecated
- Provide migration guide for extracted features
- Document recommended alternatives

## Benefits

1. **Reduced Complexity**: Single server, unified interface
2. **Easier Maintenance**: Less code duplication, shared components
3. **Better UX**: Consistent interface, no port juggling
4. **Clearer Focus**: Core features front and center
5. **Flexible Deployment**: Enable only what you need
6. **Community Clarity**: Clear project scope

## Migration Guide

### For Users
```bash
# Before (multiple servers)
python neural/dashboard/dashboard.py &          # Port 8050
python neural/no_code/no_code.py &              # Port 8051
python neural/monitoring/dashboard.py &         # Port 8052
python neural/aquarium/backend/server.py &      # Port 8000

# After (unified server)
neural server start                             # Port 8050 (all features)
# OR with specific features
neural server start --features debug,monitor    # Only what you need
```

### For Developers
```python
# Before (separate imports)
from neural.dashboard.dashboard import app as debug_app
from neural.no_code.no_code import app as nocode_app

# After (unified import)
from neural.server import app, register_feature
```

## Metrics for Success

1. **Code Reduction**: Target 30% reduction in dashboard code
2. **Port Consolidation**: From 6 ports to 3 ports
3. **Startup Time**: Single command to start all features
4. **Documentation**: Clear "Getting Started" in under 5 minutes
5. **Dependencies**: Separate core deps from experimental

## Timeline

- **Week 1**: Create unified server architecture
- **Week 2**: Consolidate dashboards into tabs
- **Week 3**: Implement feature flags and configuration
- **Week 4**: Update documentation and examples
- **Week 5**: Deprecate old entry points, migration guide

## Recommendation

**Immediate Action**: 
1. Create `neural/server.py` as the single entry point
2. Consolidate the three dashboards (debug, no-code, monitoring)
3. Document which features are core vs. experimental
4. Add feature flags to enable/disable optional components

**Future Consideration**:
1. Extract Aquarium to `neural-aquarium` repository
2. Extract Marketplace to `neural-marketplace` repository  
3. Extract Federated to `neural-federated` repository
4. Keep core DSL, HPO, AutoML, and unified dashboard in main repo

This approach maintains the valuable features while significantly reducing complexity and improving maintainability.
