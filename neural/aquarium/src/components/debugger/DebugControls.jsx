import React from 'react';
import './DebugControls.css';

const DebugControls = ({
  debuggerState,
  onStart,
  onPause,
  onStep,
  onStop,
  onStepOver,
  onStepInto,
  onStepOut,
  onContinue,
}) => {
  const { isRunning, isPaused } = debuggerState;

  return (
    <div className="debug-controls">
      <div className="control-group primary-controls">
        <button
          className="control-button start"
          onClick={onStart}
          disabled={isRunning}
          title="Start Debugging (F5)"
        >
          <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
            <path d="M3 2l10 6-10 6z" />
          </svg>
          <span>Start</span>
        </button>

        <button
          className="control-button pause"
          onClick={onPause}
          disabled={!isRunning || isPaused}
          title="Pause Execution (F6)"
        >
          <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
            <path d="M4 2h3v12H4zm5 0h3v12H9z" />
          </svg>
          <span>Pause</span>
        </button>

        <button
          className="control-button stop"
          onClick={onStop}
          disabled={!isRunning}
          title="Stop Debugging (Shift+F5)"
        >
          <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
            <rect x="3" y="3" width="10" height="10" />
          </svg>
          <span>Stop</span>
        </button>

        <div className="control-divider" />

        <button
          className="control-button continue"
          onClick={onContinue}
          disabled={!isPaused}
          title="Continue (F5)"
        >
          <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
            <path d="M3 2l8 6-8 6zm9 0v12h2V2z" />
          </svg>
          <span>Continue</span>
        </button>
      </div>

      <div className="control-group step-controls">
        <button
          className="control-button step-over"
          onClick={onStepOver}
          disabled={!isPaused}
          title="Step Over (F10)"
        >
          <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
            <path d="M8 3l4 4-4 4V8H2V6h6z" />
            <path d="M14 3v10h-2V3z" />
          </svg>
          <span>Step Over</span>
        </button>

        <button
          className="control-button step-into"
          onClick={onStepInto}
          disabled={!isPaused}
          title="Step Into (F11)"
        >
          <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
            <path d="M8 2l4 4-4 4V7H4v5H2V6h6z" />
          </svg>
          <span>Step Into</span>
        </button>

        <button
          className="control-button step-out"
          onClick={onStepOut}
          disabled={!isPaused}
          title="Step Out (Shift+F11)"
        >
          <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
            <path d="M8 14l-4-4 4-4v3h4V4h2v6H8z" />
          </svg>
          <span>Step Out</span>
        </button>

        <button
          className="control-button step"
          onClick={onStep}
          disabled={!isPaused}
          title="Step (F8)"
        >
          <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
            <path d="M3 2l10 6-10 6z" />
          </svg>
          <span>Step</span>
        </button>
      </div>
    </div>
  );
};

export default DebugControls;
