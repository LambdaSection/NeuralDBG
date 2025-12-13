# Integrated Debugger Component

The Integrated Debugger provides a comprehensive debugging interface for neural network development, embedding the NeuralDbg dashboard with advanced debugging controls.

## Features

### 1. Debug Controls
- **Start/Pause/Stop**: Basic execution control
- **Step Over**: Execute the next layer without entering
- **Step Into**: Step into layer details
- **Step Out**: Complete current layer and pause
- **Continue**: Resume execution until next breakpoint

### 2. Breakpoint Management
- Set/remove breakpoints by clicking line numbers in the code editor
- Visual indicators for active breakpoints
- Syntax highlighting for different layer types
- Current execution line indicator
- List of all active breakpoints

### 3. Variable Inspector
- **Layer Activations**: View activation statistics for each layer
  - Mean, std, min, max activation values
  - Output tensor shapes
  - Dead neuron detection
  - Anomaly detection
- **Gradients**: Inspect gradient flow during backpropagation
  - Gradient norm, mean, std
  - Gradient min/max values
- **Variables**: Inspect custom variables and state

### 4. Execution Timeline
- Visual timeline of forward and backward passes
- Real-time progress indicators
- Layer execution times
- FLOPs count per layer
- Interactive layer highlighting

### 5. NeuralDbg Dashboard Integration
- Embedded dashboard via iframe
- Real-time data synchronization
- WebSocket-based communication
- Resource monitoring
- Architecture visualization

## Components

### Debugger.jsx
Main component that orchestrates all debugging functionality.

**Props:**
- `code` (string): Neural DSL code to debug
- `onChange` (function): Callback when code changes

**Features:**
- WebSocket connection to dashboard backend
- State management for debugging session
- Event handlers for debug controls
- Real-time data updates

### DebugControls.jsx
Toolbar with debugging control buttons.

**Props:**
- `debuggerState` (object): Current debugger state
- `onStart`, `onPause`, `onStep`, `onStop`, etc.: Control handlers

**Features:**
- Visual feedback for button states
- Keyboard shortcuts support (F5, F10, F11, etc.)
- Disabled state management

### BreakpointManager.jsx
Code editor with breakpoint support.

**Props:**
- `code` (string): Code to display
- `breakpoints` (Set): Active breakpoints
- `currentLine` (number): Currently executing line
- `onToggleBreakpoint`, `onClearAll`: Breakpoint handlers

**Features:**
- Syntax highlighting for layer types
- Click-to-toggle breakpoints
- Current line highlighting
- Auto-scroll to current line
- Breakpoint list with layer names

### VariableInspector.jsx
Inspector panel for variables, activations, and gradients.

**Props:**
- `variables` (object): Custom variables
- `traceData` (array): Layer execution trace
- `currentLayer` (number/string): Current layer

**Features:**
- Tabbed interface (Activations/Gradients/Variables)
- Expandable layer details
- Statistical metrics display
- Warning indicators for anomalies
- Formatted number display

### ExecutionTimeline.jsx
Visual timeline of execution progress.

**Props:**
- `traceData` (array): Layer execution trace
- `progress` (object): Execution progress data
- `currentLayer` (number/string): Current layer

**Features:**
- Canvas-based timeline visualization
- Forward/backward pass progress bars
- Layer color coding by type
- Execution time display
- FLOPs statistics
- Interactive legend

## Usage

### Basic Setup

```jsx
import { Debugger } from './components/debugger';

function App() {
  const [code, setCode] = useState(`
network MyCNN {
  input: (28, 28, 1)
  layers:
    Conv2D(32, (3, 3), "relu")
    MaxPooling2D((2, 2))
    Flatten()
    Dense(128, "relu")
    Dense(10, "softmax")
  loss: "categorical_crossentropy"
  optimizer: Adam(learning_rate=0.001)
}
  `);

  return (
    <Debugger
      code={code}
      onChange={setCode}
    />
  );
}
```

### Backend Setup

The debugger requires the NeuralDbg dashboard to be running with WebSocket support:

```bash
# Start the enhanced dashboard with debugger backend
python neural/dashboard/dashboard_with_debugger.py
```

Or integrate into your own Flask application:

```python
from neural.dashboard.debugger_backend import create_debugger_backend, setup_debugger_routes
from flask import Flask
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

debugger = create_debugger_backend(socketio)
setup_debugger_routes(app, socketio, debugger)

if __name__ == "__main__":
    socketio.run(app, host="localhost", port=8050)
```

### Programmatic Control

```python
# Update trace data
debugger.update_trace_data({
    'layer': 'Conv2D_1',
    'execution_time': 0.002,
    'mean_activation': 0.234,
    'output_shape': [32, 28, 28, 32],
    'flops': 1234567,
})

# Update execution progress
debugger.update_execution_progress(
    forward_pass=50.0,
    backward_pass=0.0,
    phase='forward'
)

# Check for breakpoints
if debugger.check_breakpoint(line_number):
    debugger.set_current_layer('Conv2D_1')
    debugger.handle_pause()
    debugger.wait_if_paused()
```

## WebSocket Protocol

### Client → Server

**Start debugging:**
```json
{
  "command": "start"
}
```

**Add breakpoint:**
```json
{
  "command": "add_breakpoint",
  "lineNumber": 5
}
```

**Step execution:**
```json
{
  "command": "step"
}
```

### Server → Client

**State change:**
```json
{
  "type": "state_change",
  "isRunning": true,
  "isPaused": false,
  "currentLayer": "Conv2D_1",
  "currentStep": "running"
}
```

**Trace update:**
```json
{
  "type": "trace_update",
  "trace": [
    {
      "layer": "Conv2D_1",
      "execution_time": 0.002,
      "mean_activation": 0.234,
      "output_shape": [32, 28, 28, 32]
    }
  ]
}
```

**Progress update:**
```json
{
  "type": "execution_progress",
  "progress": {
    "forwardPass": 50.0,
    "backwardPass": 0.0,
    "currentPhase": "forward"
  }
}
```

## Styling

The debugger uses a dark theme consistent with modern IDEs:
- Background: `#1e1e1e`
- Panels: `#252526`
- Borders: `#3e3e42`
- Text: `#cccccc`
- Accents: Various colors for different states

All components are fully responsive and support scrolling for long content.

## Keyboard Shortcuts

- **F5**: Start/Continue debugging
- **F6**: Pause execution
- **Shift+F5**: Stop debugging
- **F8**: Step
- **F10**: Step Over
- **F11**: Step Into
- **Shift+F11**: Step Out

## Dependencies

### Frontend
- React 16.8+
- WebSocket API (built-in)

### Backend
- Flask
- flask-socketio (optional, for WebSocket support)
- python 3.8+

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Debugger                            │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              DebugControls                            │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌─────────────────────────┬─────────────────────────────┐  │
│  │  BreakpointManager      │   VariableInspector         │  │
│  │  - Code Editor          │   - Activations             │  │
│  │  - Line Numbers         │   - Gradients               │  │
│  │  - Breakpoint List      │   - Variables               │  │
│  │                         │                             │  │
│  │  NeuralDbg Dashboard    │   ExecutionTimeline         │  │
│  │  - iframe embed         │   - Timeline Canvas         │  │
│  │  - Architecture viz     │   - Progress Bars           │  │
│  │  - Resource monitoring  │   - Layer Legend            │  │
│  └─────────────────────────┴─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                             │
                    WebSocket Connection
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              DebuggerBackend (Python)                       │
│  - State Management                                         │
│  - Breakpoint Management                                    │
│  - Trace Data Collection                                    │
│  - WebSocket Communication                                  │
└─────────────────────────────────────────────────────────────┘
```

## Future Enhancements

- Variable value editing
- Conditional breakpoints
- Watch expressions
- Call stack visualization
- Memory profiling
- GPU utilization tracking
- Export debug sessions
- Replay debugging
- Multi-model comparison
- Distributed debugging support
