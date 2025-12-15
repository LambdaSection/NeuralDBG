"""
Example of using the debugger backend with a training loop.
"""

import time

import numpy as np

from neural.dashboard.debugger_backend import create_debugger_backend


try:
    from flask import Flask
    from flask_socketio import SocketIO

    from neural.dashboard.debugger_backend import setup_debugger_routes
    
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False
    print("Warning: flask_socketio not available")


def simulate_layer_execution(layer_name, debugger, line_number):
    """Simulate the execution of a neural network layer."""
    
    if debugger.check_breakpoint(line_number):
        print(f"Breakpoint hit at line {line_number}: {layer_name}")
        debugger.set_current_layer(layer_name)
        debugger.handle_pause()
        debugger.wait_if_paused()
    
    debugger.set_current_layer(layer_name)
    
    execution_time = np.random.uniform(0.001, 0.01)
    time.sleep(execution_time)
    
    mean_activation = np.random.uniform(0.0, 1.0)
    std_activation = np.random.uniform(0.0, 0.3)
    max_activation = np.random.uniform(0.5, 2.0)
    min_activation = np.random.uniform(-0.5, 0.0)
    dead_ratio = np.random.uniform(0.0, 0.5)
    
    output_shape = [32, 28, 28, 32] if 'Conv' in layer_name else [32, 128]
    flops = np.random.randint(100000, 10000000)
    
    trace_entry = {
        'layer': layer_name,
        'execution_time': execution_time,
        'mean_activation': mean_activation,
        'std_activation': std_activation,
        'max_activation': max_activation,
        'min_activation': min_activation,
        'output_shape': output_shape,
        'dead_ratio': dead_ratio,
        'flops': flops,
        'anomaly': dead_ratio > 0.3,
    }
    
    debugger.update_trace_data(trace_entry)
    
    if np.random.random() < 0.3:
        grad_norm = np.random.uniform(0.01, 2.0)
        grad_mean = np.random.uniform(-0.1, 0.1)
        grad_std = np.random.uniform(0.0, 0.5)
        
        trace_entry.update({
            'grad_norm': grad_norm,
            'grad_mean': grad_mean,
            'grad_std': grad_std,
        })


def simulate_training_loop(debugger):
    """Simulate a training loop with debugging."""
    
    layers = [
        ('Conv2D_1', 4),
        ('MaxPooling2D_1', 5),
        ('Conv2D_2', 6),
        ('MaxPooling2D_2', 7),
        ('Flatten', 8),
        ('Dense_1', 9),
        ('Dropout', 10),
        ('Dense_2', 11),
    ]
    
    print("Starting simulated training...")
    
    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}/3")
        
        debugger.update_execution_progress(
            forward_pass=0,
            backward_pass=0,
            phase='forward'
        )
        
        total_layers = len(layers)
        
        for idx, (layer_name, line_number) in enumerate(layers):
            if not debugger.is_running:
                print("Training stopped by debugger")
                return
            
            progress = ((idx + 1) / total_layers) * 100
            debugger.update_execution_progress(forward_pass=progress)
            
            print(f"  Forward: {layer_name}")
            simulate_layer_execution(layer_name, debugger, line_number)
        
        debugger.update_execution_progress(
            forward_pass=100,
            backward_pass=0,
            phase='backward'
        )
        
        for idx, (layer_name, line_number) in enumerate(reversed(layers)):
            if not debugger.is_running:
                print("Training stopped by debugger")
                return
            
            progress = ((idx + 1) / total_layers) * 100
            debugger.update_execution_progress(backward_pass=progress)
            
            print(f"  Backward: {layer_name}")
            simulate_layer_execution(f"{layer_name}_grad", debugger, line_number)
        
        debugger.update_execution_progress(
            forward_pass=100,
            backward_pass=100,
            phase='complete'
        )
        
        time.sleep(0.5)
    
    print("\nTraining complete!")
    debugger.handle_stop()


def on_start():
    """Callback when debugging starts."""
    print("Debug session started")


def on_pause():
    """Callback when debugging pauses."""
    print("Debug session paused")


def on_stop():
    """Callback when debugging stops."""
    print("Debug session stopped")


def main():
    """Main function to run the debugger example."""
    
    if not SOCKETIO_AVAILABLE:
        print("This example requires flask-socketio. Install with: pip install flask-socketio")
        return
    
    app = Flask(__name__)
    socketio = SocketIO(app, cors_allowed_origins="*")
    
    debugger = create_debugger_backend(socketio)
    setup_debugger_routes(app, socketio, debugger)
    
    debugger.register_callback('on_start', on_start)
    debugger.register_callback('on_pause', on_pause)
    debugger.register_callback('on_stop', on_stop)
    
    @app.route('/')
    def index():
        return """
        <html>
        <head>
            <title>Debugger Example</title>
        </head>
        <body>
            <h1>Neural Network Debugger Example</h1>
            <p>Open the Aquarium IDE to interact with the debugger.</p>
            <p>Dashboard is running on <a href="http://localhost:8050">http://localhost:8050</a></p>
            <div>
                <button onclick="fetch('/start_training')">Start Training</button>
            </div>
        </body>
        </html>
        """
    
    @app.route('/start_training')
    def start_training():
        """Start the training simulation."""
        import threading
        thread = threading.Thread(target=simulate_training_loop, args=(debugger,))
        thread.daemon = True
        thread.start()
        return "Training started"
    
    print("Starting debugger backend server...")
    print("Open http://localhost:8050 in your browser")
    print("Then click 'Start Training' to begin the simulation")
    
    socketio.run(app, host="localhost", port=8050, debug=False, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()
