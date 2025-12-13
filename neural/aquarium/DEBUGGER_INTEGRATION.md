# Debugger Integration Guide

This guide explains how to integrate the Neural Aquarium debugger into your development workflow.

## Architecture Overview

The debugger consists of three main components:

1. **Frontend (React)**: Visual debugger interface in `neural/aquarium/src/components/debugger/`
2. **Backend (Python)**: Debug control and data management in `neural/dashboard/debugger_backend.py`
3. **Dashboard (Dash)**: Real-time visualization in `neural/dashboard/dashboard.py`

```
┌─────────────────────┐
│   Aquarium IDE      │
│   (React Frontend)  │
│   - Debug Controls  │
│   - Code Editor     │
│   - Variable View   │
└──────────┬──────────┘
           │ WebSocket
           │
┌──────────▼──────────┐
│  Dashboard Backend  │
│  (Flask + SocketIO) │
│  - State Manager    │
│  - Breakpoints      │
│  - Data Router      │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  NeuralDbg Dash     │
│  - Visualization    │
│  - Metrics          │
│  - Architecture     │
└─────────────────────┘
```

## Setup Instructions

### 1. Install Dependencies

**Backend:**
```bash
# Core dependencies
pip install flask flask-socketio

# Full Neural DSL installation
pip install -e ".[full]"
```

**Frontend:**
```bash
cd neural/aquarium
npm install
```

### 2. Start the Dashboard

Option A: Basic dashboard (without debugger):
```bash
python neural/dashboard/dashboard.py
```

Option B: Enhanced dashboard (with debugger backend):
```bash
python neural/dashboard/dashboard_with_debugger.py
```

Option C: Custom integration:
```python
from flask import Flask
from flask_socketio import SocketIO
from neural.dashboard import create_debugger_backend, setup_debugger_routes

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

debugger = create_debugger_backend(socketio)
setup_debugger_routes(app, socketio, debugger)

if __name__ == "__main__":
    socketio.run(app, host="localhost", port=8050)
```

### 3. Start the Aquarium IDE

```bash
cd neural/aquarium
npm start
```

The debugger will be available at `http://localhost:3000`

## Integration with Training Code

### Basic Integration

```python
from neural.dashboard import create_debugger_backend

# Create debugger instance
debugger = create_debugger_backend(socketio)

# Start debugging
debugger.handle_start()

# In your training loop
for layer_idx, layer in enumerate(layers):
    # Check for breakpoints
    if debugger.check_breakpoint(layer_idx + 1):
        debugger.set_current_layer(layer.name)
        debugger.handle_pause()
        debugger.wait_if_paused()  # Blocks until user continues
    
    # Execute layer
    output = layer(input_data)
    
    # Update trace data
    debugger.update_trace_data({
        'layer': layer.name,
        'execution_time': execution_time,
        'mean_activation': float(output.mean()),
        'std_activation': float(output.std()),
        'output_shape': list(output.shape),
        'flops': calculate_flops(layer),
    })
    
    # Update progress
    progress = ((layer_idx + 1) / len(layers)) * 100
    debugger.update_execution_progress(forward_pass=progress, phase='forward')
```

### Advanced Integration with TensorFlow

```python
import tensorflow as tf
from neural.dashboard import create_debugger_backend

class DebugCallback(tf.keras.callbacks.Callback):
    def __init__(self, debugger):
        super().__init__()
        self.debugger = debugger
    
    def on_train_begin(self, logs=None):
        self.debugger.handle_start()
    
    def on_train_end(self, logs=None):
        self.debugger.handle_stop()
    
    def on_batch_begin(self, batch, logs=None):
        # Update progress
        progress = (batch / self.params['steps']) * 100
        self.debugger.update_execution_progress(
            forward_pass=progress,
            phase='forward'
        )
    
    def on_batch_end(self, batch, logs=None):
        # Update variables
        self.debugger.update_variables({
            'loss': float(logs.get('loss', 0)),
            'accuracy': float(logs.get('accuracy', 0)),
        })

# Use the callback
debugger = create_debugger_backend(socketio)
model.fit(
    x_train, y_train,
    callbacks=[DebugCallback(debugger)]
)
```

### Advanced Integration with PyTorch

```python
import torch
from neural.dashboard import create_debugger_backend

def debug_forward_hook(module, input, output, debugger, layer_name):
    """Hook to capture layer activations during forward pass."""
    if isinstance(output, torch.Tensor):
        debugger.update_trace_data({
            'layer': layer_name,
            'mean_activation': float(output.mean().item()),
            'std_activation': float(output.std().item()),
            'max_activation': float(output.max().item()),
            'min_activation': float(output.min().item()),
            'output_shape': list(output.shape),
        })

def debug_backward_hook(module, grad_input, grad_output, debugger, layer_name):
    """Hook to capture gradients during backward pass."""
    if grad_output[0] is not None:
        grad = grad_output[0]
        debugger.update_trace_data({
            'layer': f"{layer_name}_grad",
            'grad_norm': float(torch.norm(grad).item()),
            'grad_mean': float(grad.mean().item()),
            'grad_std': float(grad.std().item()),
        })

# Register hooks
debugger = create_debugger_backend(socketio)
for name, module in model.named_modules():
    module.register_forward_hook(
        lambda m, i, o, n=name: debug_forward_hook(m, i, o, debugger, n)
    )
    module.register_backward_hook(
        lambda m, gi, go, n=name: debug_backward_hook(m, gi, go, debugger, n)
    )

# Training loop
debugger.handle_start()
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Check breakpoint (if integrated with source)
        if debugger.check_breakpoint(batch_idx):
            debugger.handle_pause()
            debugger.wait_if_paused()
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Update variables
        debugger.update_variables({
            'loss': float(loss.item()),
            'epoch': epoch,
            'batch': batch_idx,
        })
```

## WebSocket Communication Protocol

### Client → Server Messages

All messages should be JSON objects sent via WebSocket:

```javascript
// Start debugging
ws.send(JSON.stringify({ command: 'start' }))

// Pause execution
ws.send(JSON.stringify({ command: 'pause' }))

// Stop debugging
ws.send(JSON.stringify({ command: 'stop' }))

// Step through execution
ws.send(JSON.stringify({ command: 'step' }))

// Add breakpoint
ws.send(JSON.stringify({ 
  command: 'add_breakpoint',
  lineNumber: 5
}))

// Remove breakpoint
ws.send(JSON.stringify({ 
  command: 'remove_breakpoint',
  lineNumber: 5
}))

// Clear all breakpoints
ws.send(JSON.stringify({ command: 'clear_all_breakpoints' }))
```

### Server → Client Messages

The server emits events via Socket.IO:

```javascript
// State change notification
{
  type: 'state_change',
  isRunning: true,
  isPaused: false,
  currentLayer: 'Conv2D_1',
  currentStep: 'running'
}

// Trace data update
{
  type: 'trace_update',
  trace: [
    {
      layer: 'Conv2D_1',
      execution_time: 0.002,
      mean_activation: 0.234,
      output_shape: [32, 28, 28, 32],
      flops: 1234567
    }
  ]
}

// Variable data update
{
  type: 'variable_update',
  variables: {
    loss: 0.234,
    accuracy: 0.89,
    learning_rate: 0.001
  }
}

// Execution progress update
{
  type: 'execution_progress',
  progress: {
    forwardPass: 50.0,
    backwardPass: 0.0,
    currentPhase: 'forward'
  }
}
```

## REST API Endpoints

The debugger backend exposes the following REST endpoints:

### GET /api/debugger/status
Get current debugger status.

**Response:**
```json
{
  "isRunning": true,
  "isPaused": false,
  "currentLayer": "Conv2D_1",
  "currentStep": "running",
  "breakpoints": [5, 7, 10]
}
```

### GET /api/debugger/trace
Get current trace data.

**Response:**
```json
{
  "trace": [
    {
      "layer": "Conv2D_1",
      "execution_time": 0.002,
      "mean_activation": 0.234
    }
  ]
}
```

### GET /api/debugger/variables
Get current variable data.

**Response:**
```json
{
  "variables": {
    "loss": 0.234,
    "accuracy": 0.89
  }
}
```

### GET /api/debugger/progress
Get current execution progress.

**Response:**
```json
{
  "progress": {
    "forwardPass": 50.0,
    "backwardPass": 0.0,
    "currentPhase": "forward"
  }
}
```

## Configuration

### Dashboard Configuration

Create a `config.yaml` file in the project root:

```yaml
# WebSocket update interval (ms)
websocket_interval: 1000

# Dashboard port
dashboard_port: 8050

# Enable debug logging
debug: false

# CORS settings
cors_origins:
  - "http://localhost:3000"
  - "http://localhost:8050"
```

### Frontend Configuration

Edit `neural/aquarium/src/components/debugger/Debugger.jsx`:

```javascript
// WebSocket configuration
const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:5001';
const DASHBOARD_URL = process.env.REACT_APP_DASHBOARD_URL || 'http://localhost:8050';

// Update intervals
const UPDATE_INTERVAL = 100; // ms
```

## Troubleshooting

### WebSocket Connection Issues

**Problem:** Cannot connect to WebSocket
**Solution:** 
1. Check that dashboard is running with WebSocket support
2. Verify CORS settings allow your origin
3. Check firewall settings

**Problem:** Connection keeps disconnecting
**Solution:**
1. Check network stability
2. Increase timeout values
3. Implement reconnection with exponential backoff

### Performance Issues

**Problem:** UI becomes slow with many trace entries
**Solution:**
1. Implement pagination in trace view
2. Limit trace buffer size
3. Use virtual scrolling for long lists

**Problem:** High memory usage
**Solution:**
1. Clear trace data periodically
2. Implement trace data rotation
3. Reduce update frequency

### Breakpoint Issues

**Problem:** Breakpoints not triggering
**Solution:**
1. Verify breakpoint line numbers match code
2. Check that breakpoint checks are in training loop
3. Ensure debugger state is synchronized

## Examples

See the following files for complete examples:

- `neural/dashboard/debugger_example.py` - Backend integration example
- `neural/aquarium/src/components/debugger/example.jsx` - Frontend usage example
- `tests/test_debugger.py` - Unit tests showing API usage

## Best Practices

1. **Always check debugger state before operations**
   ```python
   if debugger.is_running and not debugger.is_paused:
       # Continue execution
   ```

2. **Use thread-safe operations**
   ```python
   with debugger._lock:
       # Critical section
   ```

3. **Limit trace data size**
   ```python
   MAX_TRACE_ENTRIES = 1000
   if len(debugger.trace_buffer) > MAX_TRACE_ENTRIES:
       debugger.trace_buffer = debugger.trace_buffer[-MAX_TRACE_ENTRIES:]
   ```

4. **Handle WebSocket disconnections gracefully**
   ```javascript
   ws.onclose = () => {
       console.log('Disconnected, reconnecting...');
       setTimeout(connectWebSocket, 3000);
   };
   ```

5. **Provide meaningful layer names**
   ```python
   debugger.update_trace_data({
       'layer': f'Conv2D_block1_layer2',  # Good
       # 'layer': 'layer_5',  # Less useful
   })
   ```

## Security Considerations

1. **Use HTTPS/WSS in production**
2. **Implement authentication for dashboard access**
3. **Sanitize user input in breakpoint management**
4. **Limit CORS origins to trusted domains**
5. **Rate-limit WebSocket messages**
6. **Validate all incoming debug commands**

## Contributing

To contribute improvements to the debugger:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

See `CONTRIBUTING.md` for more details.
