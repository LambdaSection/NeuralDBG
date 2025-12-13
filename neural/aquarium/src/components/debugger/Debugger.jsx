import React, { useState, useEffect, useRef } from 'react';
import './Debugger.css';
import DebugControls from './DebugControls';
import BreakpointManager from './BreakpointManager';
import VariableInspector from './VariableInspector';
import ExecutionTimeline from './ExecutionTimeline';

const Debugger = ({ code, onChange }) => {
  const [debuggerState, setDebuggerState] = useState({
    isRunning: false,
    isPaused: false,
    currentLayer: null,
    currentStep: 'idle',
    breakpoints: new Set(),
  });

  const [traceData, setTraceData] = useState([]);
  const [variables, setVariables] = useState({});
  const [executionProgress, setExecutionProgress] = useState({
    forwardPass: 0,
    backwardPass: 0,
    currentPhase: 'idle',
  });

  const iframeRef = useRef(null);
  const wsRef = useRef(null);

  // Connect to NeuralDbg dashboard WebSocket
  useEffect(() => {
    const connectWebSocket = () => {
      try {
        const ws = new WebSocket('ws://localhost:5001');
        
        ws.onopen = () => {
          console.log('Connected to NeuralDbg dashboard');
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            handleDashboardMessage(data);
          } catch (error) {
            console.error('Error parsing WebSocket message:', error);
          }
        };

        ws.onerror = (error) => {
          console.error('WebSocket error:', error);
        };

        ws.onclose = () => {
          console.log('Disconnected from NeuralDbg dashboard');
          // Attempt to reconnect after 3 seconds
          setTimeout(connectWebSocket, 3000);
        };

        wsRef.current = ws;
      } catch (error) {
        console.error('Error connecting to WebSocket:', error);
      }
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  // Handle messages from the dashboard
  const handleDashboardMessage = (data) => {
    if (data.type === 'trace_update') {
      setTraceData(data.trace);
    } else if (data.type === 'variable_update') {
      setVariables(data.variables);
    } else if (data.type === 'execution_progress') {
      setExecutionProgress(data.progress);
    } else if (data.type === 'state_change') {
      setDebuggerState(prev => ({
        ...prev,
        isRunning: data.isRunning,
        isPaused: data.isPaused,
        currentLayer: data.currentLayer,
        currentStep: data.currentStep,
      }));
    }
  };

  // Send control commands to the dashboard
  const sendCommand = (command, data = {}) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        command,
        ...data,
      }));
    } else {
      console.warn('WebSocket not connected');
    }
  };

  // Debug control handlers
  const handleStart = () => {
    sendCommand('start');
    setDebuggerState(prev => ({
      ...prev,
      isRunning: true,
      isPaused: false,
      currentStep: 'running',
    }));
  };

  const handlePause = () => {
    sendCommand('pause');
    setDebuggerState(prev => ({
      ...prev,
      isPaused: true,
    }));
  };

  const handleStep = () => {
    sendCommand('step');
  };

  const handleStop = () => {
    sendCommand('stop');
    setDebuggerState(prev => ({
      ...prev,
      isRunning: false,
      isPaused: false,
      currentLayer: null,
      currentStep: 'idle',
    }));
    setTraceData([]);
    setVariables({});
    setExecutionProgress({
      forwardPass: 0,
      backwardPass: 0,
      currentPhase: 'idle',
    });
  };

  const handleStepOver = () => {
    sendCommand('step_over');
  };

  const handleStepInto = () => {
    sendCommand('step_into');
  };

  const handleStepOut = () => {
    sendCommand('step_out');
  };

  const handleContinue = () => {
    sendCommand('continue');
    setDebuggerState(prev => ({
      ...prev,
      isPaused: false,
    }));
  };

  // Breakpoint management
  const handleToggleBreakpoint = (lineNumber) => {
    const newBreakpoints = new Set(debuggerState.breakpoints);
    if (newBreakpoints.has(lineNumber)) {
      newBreakpoints.delete(lineNumber);
      sendCommand('remove_breakpoint', { lineNumber });
    } else {
      newBreakpoints.add(lineNumber);
      sendCommand('add_breakpoint', { lineNumber });
    }
    setDebuggerState(prev => ({
      ...prev,
      breakpoints: newBreakpoints,
    }));
  };

  const handleClearAllBreakpoints = () => {
    sendCommand('clear_all_breakpoints');
    setDebuggerState(prev => ({
      ...prev,
      breakpoints: new Set(),
    }));
  };

  return (
    <div className="debugger-container">
      <div className="debugger-header">
        <h2>Integrated Debugger</h2>
        <div className="debugger-status">
          <span className={`status-indicator ${debuggerState.isRunning ? 'running' : ''} ${debuggerState.isPaused ? 'paused' : ''}`} />
          <span className="status-text">
            {debuggerState.isPaused ? 'Paused' : debuggerState.isRunning ? 'Running' : 'Idle'}
          </span>
        </div>
      </div>

      <DebugControls
        debuggerState={debuggerState}
        onStart={handleStart}
        onPause={handlePause}
        onStep={handleStep}
        onStop={handleStop}
        onStepOver={handleStepOver}
        onStepInto={handleStepInto}
        onStepOut={handleStepOut}
        onContinue={handleContinue}
      />

      <div className="debugger-main">
        <div className="debugger-left">
          <div className="dashboard-container">
            <h3>NeuralDbg Dashboard</h3>
            <iframe
              ref={iframeRef}
              src="http://localhost:8050"
              title="NeuralDbg Dashboard"
              className="dashboard-iframe"
              sandbox="allow-same-origin allow-scripts allow-forms"
            />
          </div>

          <BreakpointManager
            code={code}
            breakpoints={debuggerState.breakpoints}
            currentLine={debuggerState.currentLayer}
            onToggleBreakpoint={handleToggleBreakpoint}
            onClearAll={handleClearAllBreakpoints}
          />
        </div>

        <div className="debugger-right">
          <VariableInspector
            variables={variables}
            traceData={traceData}
            currentLayer={debuggerState.currentLayer}
          />

          <ExecutionTimeline
            traceData={traceData}
            progress={executionProgress}
            currentLayer={debuggerState.currentLayer}
          />
        </div>
      </div>
    </div>
  );
};

export default Debugger;
