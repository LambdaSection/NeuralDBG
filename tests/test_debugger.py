"""
Tests for the debugger backend functionality.
"""

import threading
import time

import pytest

from neural.dashboard.debugger_backend import DebuggerBackend


class TestDebuggerBackend:
    """Test suite for DebuggerBackend."""

    def setup_method(self):
        """Set up test fixtures."""
        self.debugger = DebuggerBackend(socketio=None)

    def test_initial_state(self):
        """Test initial debugger state."""
        assert not self.debugger.is_running
        assert not self.debugger.is_paused
        assert self.debugger.current_layer is None
        assert self.debugger.current_step == 'idle'
        assert len(self.debugger.breakpoints) == 0
        assert len(self.debugger.trace_buffer) == 0

    def test_start_debugging(self):
        """Test starting a debug session."""
        self.debugger.handle_start()
        assert self.debugger.is_running
        assert not self.debugger.is_paused
        assert self.debugger.current_step == 'running'
        assert self.debugger.execution_progress['currentPhase'] == 'forward'

    def test_pause_debugging(self):
        """Test pausing execution."""
        self.debugger.handle_start()
        self.debugger.handle_pause()
        assert self.debugger.is_running
        assert self.debugger.is_paused
        assert self.debugger.current_step == 'paused'

    def test_stop_debugging(self):
        """Test stopping a debug session."""
        self.debugger.handle_start()
        self.debugger.update_trace_data({'layer': 'test', 'execution_time': 0.1})
        self.debugger.handle_stop()
        
        assert not self.debugger.is_running
        assert not self.debugger.is_paused
        assert self.debugger.current_layer is None
        assert self.debugger.current_step == 'idle'
        assert len(self.debugger.trace_buffer) == 0

    def test_step_debugging(self):
        """Test step execution."""
        self.debugger.handle_start()
        self.debugger.handle_pause()
        self.debugger.handle_step()
        assert self.debugger.current_step == 'stepping'

    def test_continue_debugging(self):
        """Test continuing after pause."""
        self.debugger.handle_start()
        self.debugger.handle_pause()
        self.debugger.handle_continue()
        assert not self.debugger.is_paused
        assert self.debugger.current_step == 'running'

    def test_breakpoint_management(self):
        """Test breakpoint add/remove operations."""
        self.debugger.handle_add_breakpoint(5)
        self.debugger.handle_add_breakpoint(10)
        
        assert 5 in self.debugger.breakpoints
        assert 10 in self.debugger.breakpoints
        assert self.debugger.check_breakpoint(5)
        assert self.debugger.check_breakpoint(10)
        assert not self.debugger.check_breakpoint(7)
        
        self.debugger.handle_remove_breakpoint(5)
        assert 5 not in self.debugger.breakpoints
        assert not self.debugger.check_breakpoint(5)
        
        self.debugger.handle_clear_all_breakpoints()
        assert len(self.debugger.breakpoints) == 0

    def test_trace_data_update(self):
        """Test updating trace data."""
        trace_entry = {
            'layer': 'Conv2D_1',
            'execution_time': 0.002,
            'mean_activation': 0.234,
            'output_shape': [32, 28, 28, 32],
        }
        
        self.debugger.update_trace_data(trace_entry)
        assert len(self.debugger.trace_buffer) == 1
        assert self.debugger.trace_buffer[0] == trace_entry

    def test_variable_update(self):
        """Test updating variables."""
        variables = {
            'loss': 0.234,
            'accuracy': 0.89,
        }
        
        self.debugger.update_variables(variables)
        assert self.debugger.variables['loss'] == 0.234
        assert self.debugger.variables['accuracy'] == 0.89

    def test_execution_progress_update(self):
        """Test updating execution progress."""
        self.debugger.update_execution_progress(
            forward_pass=50.0,
            backward_pass=0.0,
            phase='forward'
        )
        
        assert self.debugger.execution_progress['forwardPass'] == 50.0
        assert self.debugger.execution_progress['backwardPass'] == 0.0
        assert self.debugger.execution_progress['currentPhase'] == 'forward'

    def test_set_current_layer(self):
        """Test setting current layer."""
        self.debugger.set_current_layer('Conv2D_1')
        assert self.debugger.current_layer == 'Conv2D_1'

    def test_callback_registration(self):
        """Test callback registration and execution."""
        callback_executed = {'value': False}
        
        def test_callback():
            callback_executed['value'] = True
        
        self.debugger.register_callback('on_start', test_callback)
        self.debugger.handle_start()
        
        assert callback_executed['value']

    def test_wait_if_paused(self):
        """Test wait_if_paused blocking behavior."""
        self.debugger.handle_start()
        self.debugger.handle_pause()
        
        wait_complete = {'value': False}
        
        def wait_thread():
            self.debugger.wait_if_paused()
            wait_complete['value'] = True
        
        thread = threading.Thread(target=wait_thread)
        thread.start()
        
        time.sleep(0.2)
        assert not wait_complete['value']
        
        self.debugger.handle_continue()
        thread.join(timeout=1.0)
        assert wait_complete['value']

    def test_thread_safety(self):
        """Test thread-safe operations."""
        def add_trace():
            for i in range(100):
                self.debugger.update_trace_data({
                    'layer': f'layer_{i}',
                    'execution_time': 0.001,
                })
        
        threads = [threading.Thread(target=add_trace) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(self.debugger.trace_buffer) == 500

    def test_multiple_callbacks(self):
        """Test multiple callbacks for same event."""
        callback_count = {'value': 0}
        
        def callback1():
            callback_count['value'] += 1
        
        def callback2():
            callback_count['value'] += 10
        
        self.debugger.register_callback('on_start', callback1)
        self.debugger.register_callback('on_start', callback2)
        self.debugger.handle_start()
        
        assert callback_count['value'] == 11

    def test_breakpoint_hit_scenario(self):
        """Test a realistic breakpoint scenario."""
        self.debugger.handle_start()
        self.debugger.handle_add_breakpoint(5)
        
        layers = [
            ('Conv2D_1', 4),
            ('MaxPool_1', 5),
            ('Conv2D_2', 6),
        ]
        
        hit_breakpoint = False
        
        for layer_name, line_number in layers:
            if self.debugger.check_breakpoint(line_number):
                self.debugger.set_current_layer(layer_name)
                self.debugger.handle_pause()
                hit_breakpoint = True
                break
        
        assert hit_breakpoint
        assert self.debugger.is_paused
        assert self.debugger.current_layer == 'MaxPool_1'


class TestDebuggerWithSocketIO:
    """Test suite for DebuggerBackend with SocketIO."""

    def setup_method(self):
        """Set up test fixtures."""
        try:
            from flask import Flask
            from flask_socketio import SocketIO
            
            app = Flask(__name__)
            self.socketio = SocketIO(app)
            self.debugger = DebuggerBackend(socketio=self.socketio)
        except ImportError:
            pytest.skip("flask_socketio not available")

    def test_emit_state_change(self):
        """Test emitting state change with SocketIO."""
        self.debugger.handle_start()

    def test_emit_trace_update(self):
        """Test emitting trace update with SocketIO."""
        self.debugger.update_trace_data({'layer': 'test', 'execution_time': 0.1})

    def test_emit_variable_update(self):
        """Test emitting variable update with SocketIO."""
        self.debugger.update_variables({'loss': 0.5})

    def test_emit_progress_update(self):
        """Test emitting progress update with SocketIO."""
        self.debugger.update_execution_progress(forward_pass=50.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
