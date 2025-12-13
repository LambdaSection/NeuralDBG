# Deprecation Notice

## ⚠️ This Dashboard Entry Point is Deprecated

Starting with Neural DSL v0.3.0, running separate dashboards on different ports is deprecated.

### What's Deprecated

Running these files directly:
```bash
python neural/dashboard/dashboard.py          # Port 8050
python neural/no_code/no_code.py              # Port 8051
python neural/monitoring/dashboard.py         # Port 8052
```

### What to Use Instead

Use the unified server:
```bash
neural server start                            # Port 8050 (all features)
```

The unified server provides all dashboard functionality in a single tabbed interface:
- **Debug Tab**: Real-time execution monitoring (NeuralDbg)
- **Build Tab**: No-code visual model builder
- **Monitor Tab**: Production monitoring
- **Settings Tab**: Configuration and feature toggles

### Why the Change?

**Problems with Separate Dashboards**:
- Multiple terminals required
- Multiple ports to manage (8050, 8051, 8052)
- Inconsistent UI/UX across dashboards
- High maintenance burden
- Confusing for new users

**Benefits of Unified Server**:
- ✅ Single command to start
- ✅ Single port (8050)
- ✅ Consistent navigation and UI
- ✅ Feature toggles for flexibility
- ✅ Easier to maintain and extend

### Migration Guide

**Before (Deprecated)**:
```bash
# Terminal 1
python neural/dashboard/dashboard.py

# Terminal 2
python neural/no_code/no_code.py

# Terminal 3
python neural/monitoring/dashboard.py
```

**After (Recommended)**:
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
  - Click "Debug" tab for NeuralDbg
  - Click "Build" tab for No-Code builder
  - Click "Monitor" tab for production monitoring

### Timeline

- **v0.3.0** (Current): Deprecation notice, unified server available
- **v0.3.x**: Both old and new methods work (migration period)
- **v0.4.0** (Planned): Old entry points removed

### Compatibility

For backward compatibility, the old entry points still work in v0.3.x but will show a deprecation warning. Update your scripts and workflows before v0.4.0.

### Support

For questions or issues:
- Read: [UNIFIED_SERVER_QUICKSTART.md](../../UNIFIED_SERVER_QUICKSTART.md)
- Check: [SERVICE_REGISTRY.md](../../SERVICE_REGISTRY.md)
- Report: GitHub Issues

---

**Action Required**: Update your scripts to use `neural server start` instead of running dashboard files directly.
