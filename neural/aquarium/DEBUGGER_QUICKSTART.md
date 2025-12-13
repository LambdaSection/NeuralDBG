# Neural Aquarium Debugger - Quick Start Guide

Get up and running with the integrated debugger in 5 minutes!

## Prerequisites

- Python 3.8+
- Node.js 14+ and npm
- Neural DSL installed
- Flask and flask-socketio

## Installation

### 1. Install Backend Dependencies

```bash
# Install Neural DSL with full features
pip install -e ".[full]"

# Or install just the debugger dependencies
pip install flask flask-socketio
```

### 2. Install Frontend Dependencies

```bash
cd neural/aquarium
npm install
```

## Quick Start

### Option 1: Run Example (Recommended for First Time)

```bash
# Terminal 1: Start the debugger backend example
python neural/dashboard/debugger_example.py

# Terminal 2: Start the Aquarium IDE
cd neural/aquarium
npm start
```

Then:
1. Open http://localhost:3000 in your browser
2. Open http://localhost:8050 in another tab
3. Click "Start Training" on the dashboard
4. Watch the debugger in action!

### Option 2: Integrate with Your Code

#### Step 1: Start the Dashboard

```bash
python neural/dashboard/dashboard_with_debugger.py
```

#### Step 2: Add Debugger to Your Training Code

```python
from flask import Flask
from flask_socketio import SocketIO
from neural.dashboard import create_debugger_backend, setup_debugger_routes

# Create Flask app and SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Create debugger backend
debugger = create_debugger_backend(socketio)
setup_debugger_routes(app, socketio, debugger)

# Your training code
def train_model():
    debugger.handle_start()
    
    for layer_idx, layer in enumerate(layers):
        # Check breakpoints
        if debugger.check_breakpoint(layer_idx + 1):
            debugger.set_current_layer(layer.name)
            debugger.handle_pause()
            debugger.wait_if_paused()
        
        # Execute layer
        output = layer(input_data)
        
        # Update debugger
        debugger.update_trace_data({
            'layer': layer.name,
            'execution_time': 0.002,
            'mean_activation': float(output.mean()),
            'output_shape': list(output.shape),
        })
    
    debugger.handle_stop()

# Run the server
if __name__ == "__main__":
    socketio.run(app, host="localhost", port=8050)
```

#### Step 3: Start the Frontend

```bash
cd neural/aquarium
npm start
```

Open http://localhost:3000 and start debugging!

## Basic Usage

### Setting Breakpoints

1. **In the Code Editor**: Click on any line number
2. **Visual Feedback**: Red dot appears next to the line
3. **Remove**: Click the line number again
4. **Clear All**: Click "Clear All" button

### Debugging Controls

1. **Start**: Click the green "Start" button or press F5
2. **Pause**: Click the orange "Pause" button or press F6
3. **Step**: Click "Step" or press F8
4. **Stop**: Click the red "Stop" button or press Shift+F5

### Inspecting Variables

1. **Click the tabs**: Activations, Gradients, or Variables
2. **Expand layers**: Click on any layer to see details
3. **View metrics**: Check mean, std, min, max values
4. **Watch for warnings**: Yellow/red indicators show issues

### Viewing Timeline

1. **Check Progress**: See forward/backward pass progress bars
2. **Layer Timing**: View execution time for each layer
3. **Identify Bottlenecks**: Long bars indicate slow layers
4. **Monitor FLOPs**: See computational cost per layer

## Example Workflow

### 1. Set Breakpoints

```neural
network MyCNN {
  input: (28, 28, 1)
  layers:
    Conv2D(32, (3, 3), "relu")      # Line 4 - Set breakpoint here
    MaxPooling2D((2, 2))             # Line 5
    Flatten()                        # Line 6 - And here
    Dense(128, "relu")               # Line 7
    Dense(10, "softmax")             # Line 8
  loss: "categorical_crossentropy"
  optimizer: Adam(learning_rate=0.001)
}
```

Click on lines 4 and 6 to set breakpoints.

### 2. Start Debugging

Click "Start" or press F5.

### 3. Execution Pauses at First Breakpoint

When execution reaches line 4 (Conv2D):
- Current line highlighted in yellow
- Debugger pauses automatically
- Status shows "Paused"

### 4. Inspect State

In the Variable Inspector:
- Click "Activations" tab
- Expand "Conv2D" layer
- View activation statistics

### 5. Continue or Step

Options:
- **Continue (F5)**: Run until next breakpoint
- **Step (F8)**: Execute next layer
- **Step Over (F10)**: Skip to next line
- **Stop (Shift+F5)**: End debugging session

## Common Patterns

### Pattern 1: Debug Dead Neurons

```python
# In your code
debugger.update_trace_data({
    'layer': 'Conv2D_1',
    'dead_ratio': 0.45,  # 45% of neurons are dead!
})
```

In the UI:
1. Go to Activations tab
2. Expand the layer
3. See "Dead Neurons: 45%" in orange/red

### Pattern 2: Monitor Gradient Flow

```python
# In backward hook
debugger.update_trace_data({
    'layer': 'Dense_1',
    'grad_norm': 0.0001,  # Vanishing gradient!
})
```

In the UI:
1. Go to Gradients tab
2. Check gradient norms
3. Small values indicate vanishing gradients

### Pattern 3: Track Training Metrics

```python
# In training loop
debugger.update_variables({
    'loss': current_loss,
    'accuracy': current_accuracy,
    'learning_rate': current_lr,
})
```

In the UI:
1. Go to Variables tab
2. See real-time metric updates

## Troubleshooting

### Problem: Can't Connect to Dashboard

**Solution:**
```bash
# Check if dashboard is running
curl http://localhost:8050/api/debugger/status

# If not running, start it
python neural/dashboard/dashboard_with_debugger.py
```

### Problem: Breakpoints Not Working

**Solution:**
1. Ensure you're calling `debugger.check_breakpoint(line_number)`
2. Verify line numbers match your code
3. Check that debugger is in running state

### Problem: No Data in Inspector

**Solution:**
1. Verify you're calling `debugger.update_trace_data()`
2. Check WebSocket connection (look for console errors)
3. Ensure data is in correct format

### Problem: UI is Slow

**Solution:**
1. Reduce update frequency
2. Limit trace buffer size
3. Clear old data periodically

```python
# Limit buffer size
MAX_ENTRIES = 1000
if len(debugger.trace_buffer) > MAX_ENTRIES:
    debugger.trace_buffer = debugger.trace_buffer[-MAX_ENTRIES:]
```

## Keyboard Shortcuts Reference

| Shortcut | Action |
|----------|--------|
| F5 | Start/Continue |
| F6 | Pause |
| Shift+F5 | Stop |
| F8 | Step |
| F10 | Step Over |
| F11 | Step Into |
| Shift+F11 | Step Out |

## Tips & Tricks

### Tip 1: Use Meaningful Layer Names

```python
# Good
debugger.update_trace_data({
    'layer': 'Conv2D_block1_layer1',
    ...
})

# Less helpful
debugger.update_trace_data({
    'layer': 'layer_5',
    ...
})
```

### Tip 2: Set Strategic Breakpoints

Set breakpoints at:
- First layer (check input data)
- After pooling layers (check dimension reduction)
- Before output layer (check final representations)
- Problem layers (where training struggles)

### Tip 3: Monitor Both Passes

Track both forward and backward:
```python
# Forward pass
debugger.update_trace_data({
    'layer': 'Conv2D_1',
    'mean_activation': ...,
})

# Backward pass
debugger.update_trace_data({
    'layer': 'Conv2D_1_grad',
    'grad_norm': ...,
})
```

### Tip 4: Use Progress Updates

Keep users informed:
```python
for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(batches):
        progress = (batch_idx / len(batches)) * 100
        debugger.update_execution_progress(
            forward_pass=progress,
            phase='forward'
        )
```

### Tip 5: Handle Exceptions Gracefully

```python
try:
    # Your training code
    train_model()
except Exception as e:
    debugger.handle_stop()
    raise
```

## Next Steps

Now that you're up and running:

1. **Read the Integration Guide**: See `DEBUGGER_INTEGRATION.md` for advanced integration patterns

2. **Explore Features**: Check `DEBUGGER_FEATURES.md` for complete feature documentation

3. **View Examples**: Look at `neural/dashboard/debugger_example.py` for complete examples

4. **Join the Community**: Get help on GitHub or Discord

5. **Contribute**: Help improve the debugger! See `CONTRIBUTING.md`

## Quick Reference

### Import Statement
```python
from neural.dashboard import create_debugger_backend, setup_debugger_routes
```

### Basic Setup
```python
app = Flask(__name__)
socketio = SocketIO(app)
debugger = create_debugger_backend(socketio)
setup_debugger_routes(app, socketio, debugger)
```

### Essential Methods
```python
debugger.handle_start()
debugger.handle_pause()
debugger.handle_stop()
debugger.check_breakpoint(line_number)
debugger.update_trace_data({...})
debugger.update_variables({...})
debugger.update_execution_progress(...)
```

## Resources

- **Documentation**: `neural/aquarium/src/components/debugger/README.md`
- **Examples**: `neural/dashboard/debugger_example.py`
- **Tests**: `tests/test_debugger.py`
- **GitHub**: https://github.com/Lemniscate-world/Neural
- **Issues**: https://github.com/Lemniscate-world/Neural/issues

Happy debugging! 🐛🔍
