# Neural DSL Cleanup - Refocusing Implementation

## Directories to Remove

These directories contain features that dilute focus and should be removed:

### Enterprise/Business Features
```
neural/teams/              # Multi-tenancy, RBAC, quotas, billing
neural/marketplace/        # Model/plugin marketplace
neural/cost/              # Cost optimization tracking
```

### Alternative Tool Features (Use specialized tools instead)
```
neural/mlops/             # Model registry, deployment, A/B testing (use MLflow)
neural/api/               # API server, Celery, Redis (wrap in FastAPI if needed)
neural/cloud/             # Cloud integrations (use boto3, google-cloud, azure)
neural/integrations/      # ML platform connectors (use platform SDKs)
neural/monitoring/        # Prometheus integration (use Prometheus directly)
neural/data/              # Data versioning (use DVC)
neural/collaboration/     # Real-time editing, workspaces (use Git)
neural/tracking/          # Experiment tracking (use MLflow, W&B)
```

### Experimental/Scope Creep
```
neural/no_code/           # No-code GUI (DSL is the interface)
neural/neural_chat/       # Chat/LLM integration
neural/neural_llm/        # LLM features
neural/research_generation/  # Research paper generation
neural/aquarium/          # Unknown feature
neural/ai/                # AI-related experiments
neural/plugins/           # Plugin system
neural/hacky/             # Experimental code
```

### Redundant/Overlapping
```
neural/profiling/         # Profiling (excessive for DSL compiler)
neural/benchmarks/        # Benchmarking suite (not core)
neural/execution_optimization/  # Execution optimization (framework's job)
neural/explainability/    # Model explainability (use SHAP, LIME)
neural/docgen/            # Documentation generation
neural/config/            # Configuration management
neural/pretrained_models/ # Pretrained models (in neural/ folder)
neural/federated/         # Federated learning
```

### Directories to Keep (Core + Essential Optional)

**Core (always keep):**
```
neural/parser/            # DSL parser and AST
neural/code_generation/   # Multi-backend code generation
neural/shape_propagation/ # Shape validation
neural/cli/               # Command-line interface
neural/visualization/     # Network visualization
neural/utils/             # Shared utilities
neural/exceptions.py      # Exception definitions
neural/__init__.py        # Package initialization
neural/__main__.py        # CLI entry point
```

**Optional (support core mission):**
```
neural/hpo/               # Hyperparameter optimization
neural/automl/            # Neural Architecture Search
neural/dashboard/         # Debugging dashboard (simplified)
neural/training/          # Training utilities
neural/metrics/           # Metric computation
```

## Files to Remove/Update

### Root Directory Cleanup
```
# Remove implementation documentation for removed features
TEAMS_IMPLEMENTATION.md
MARKETPLACE_IMPLEMENTATION.md
MARKETPLACE_SUMMARY.md
MLOPS_IMPLEMENTATION.md
INTEGRATIONS_SUMMARY.md
INTEGRATION_IMPLEMENTATION.md
NEURAL_API_IMPLEMENTATION.md
COLLABORATION_IMPLEMENTATION.md (if exists)
FEDERATED_IMPLEMENTATION.md (if exists)
COST_OPTIMIZATION_IMPLEMENTATION.md
DATA_VERSIONING_IMPLEMENTATION.md
AQUARIUM_IMPLEMENTATION_SUMMARY.md
CLOUD_IMPROVEMENTS_SUMMARY.md
WEBSITE_IMPLEMENTATION_SUMMARY.md (if website is for removed features)

# Update/consolidate
DEPENDENCY_GUIDE.md -> Update to reflect focused dependencies
INSTALL.md -> Simplify for focused feature set
```

### Requirements Files
```
requirements-api.txt      # Remove (API feature removed)
requirements-backends.txt # Keep but simplify
requirements-viz.txt      # Keep
requirements-minimal.txt  # Keep and update
requirements-dev.txt      # Keep
requirements.txt          # Update to reflect core only
```

## Implementation Steps

### Step 1: Back up removed code (optional)
```bash
# Create archive of removed features for reference
mkdir ../neural-removed-features
cp -r neural/teams ../neural-removed-features/
cp -r neural/marketplace ../neural-removed-features/
# ... etc for all removed directories
```

### Step 2: Remove directories
```powershell
# Remove enterprise features
Remove-Item -Recurse -Force neural/teams
Remove-Item -Recurse -Force neural/marketplace
Remove-Item -Recurse -Force neural/cost
Remove-Item -Recurse -Force neural/mlops

# Remove alternative tool features
Remove-Item -Recurse -Force neural/api
Remove-Item -Recurse -Force neural/cloud
Remove-Item -Recurse -Force neural/integrations
Remove-Item -Recurse -Force neural/monitoring
Remove-Item -Recurse -Force neural/data
Remove-Item -Recurse -Force neural/collaboration
Remove-Item -Recurse -Force neural/tracking

# Remove experimental/scope creep
Remove-Item -Recurse -Force neural/no_code
Remove-Item -Recurse -Force neural/neural_chat
Remove-Item -Recurse -Force neural/neural_llm
Remove-Item -Recurse -Force neural/research_generation
Remove-Item -Recurse -Force neural/aquarium
Remove-Item -Recurse -Force neural/ai
Remove-Item -Recurse -Force neural/plugins
Remove-Item -Recurse -Force neural/hacky

# Remove redundant/overlapping
Remove-Item -Recurse -Force neural/profiling
Remove-Item -Recurse -Force neural/benchmarks
Remove-Item -Recurse -Force neural/execution_optimization
Remove-Item -Recurse -Force neural/explainability
Remove-Item -Recurse -Force neural/docgen
Remove-Item -Recurse -Force neural/config
Remove-Item -Recurse -Force neural/pretrained_models
Remove-Item -Recurse -Force neural/federated
```

### Step 3: Remove documentation for removed features
```powershell
Remove-Item -Force TEAMS_IMPLEMENTATION.md
Remove-Item -Force MARKETPLACE_IMPLEMENTATION.md
Remove-Item -Force MARKETPLACE_SUMMARY.md
Remove-Item -Force MLOPS_IMPLEMENTATION.md
Remove-Item -Force INTEGRATIONS_SUMMARY.md
Remove-Item -Force INTEGRATION_IMPLEMENTATION.md
Remove-Item -Force NEURAL_API_IMPLEMENTATION.md
Remove-Item -Force COST_OPTIMIZATION_IMPLEMENTATION.md
Remove-Item -Force DATA_VERSIONING_IMPLEMENTATION.md
Remove-Item -Force AQUARIUM_IMPLEMENTATION_SUMMARY.md
Remove-Item -Force CLOUD_IMPROVEMENTS_SUMMARY.md
```

### Step 4: Update test directory
Remove tests for removed features:
```powershell
# Remove test files for removed modules
Remove-Item -Recurse -Force tests/teams
Remove-Item -Recurse -Force tests/marketplace
Remove-Item -Recurse -Force tests/mlops
Remove-Item -Recurse -Force tests/api
Remove-Item -Recurse -Force tests/cloud
Remove-Item -Recurse -Force tests/integrations
Remove-Item -Recurse -Force tests/collaboration
Remove-Item -Recurse -Force tests/federated
# ... etc
```

### Step 5: Verify core functionality
```bash
# Test that core features still work
python -m pytest tests/parser/ -v
python -m pytest tests/code_generation/ -v
python -m pytest tests/shape_propagation/ -v
python -m pytest tests/cli/ -v
```

## Benefits After Cleanup

### Code Metrics (Estimated)
- **Before**: ~50,000+ lines across all features
- **After**: ~15,000 lines (70% reduction)
- **Directories**: 40+ -> 10 core modules
- **Dependencies**: 50+ -> 15 core packages

### Maintenance Improvements
- Faster CI/CD (fewer tests)
- Easier onboarding (clearer scope)
- Better documentation (focused on core)
- Reduced bug surface area
- Faster iteration on core features

### User Experience
- Clearer value proposition
- Simpler installation
- Better performance (less overhead)
- More focused documentation
- Easier to learn and use

## Communication Plan

1. **Release Notes**: Clearly explain refocusing and removed features
2. **Migration Guide**: Provide alternatives for removed features
3. **Blog Post**: Explain philosophy behind refocusing
4. **Discord Announcement**: Answer user questions
5. **GitHub Issues**: Update with "wontfix" label for removed features

## Version Numbering
- v0.4.0: Major version bump to signal breaking changes
- Semantic versioning: Breaking change = major version bump
- Clear changelog documenting all removals

## Rollback Plan
If refocusing proves problematic:
1. Restore from ../neural-removed-features/
2. Revert commits
3. Reconsider which features are truly core

However, commitment to focus is key. Rollback should only be for critical issues, not user requests for removed features.
