import React, { useRef, useEffect } from 'react';
import './BreakpointManager.css';

const BreakpointManager = ({
  code,
  breakpoints,
  currentLine,
  onToggleBreakpoint,
  onClearAll,
}) => {
  const editorRef = useRef(null);
  const lines = code ? code.split('\n') : [];

  useEffect(() => {
    if (editorRef.current && currentLine !== null) {
      const lineElement = editorRef.current.querySelector(`[data-line="${currentLine}"]`);
      if (lineElement) {
        lineElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }
  }, [currentLine]);

  const handleLineClick = (lineNumber) => {
    onToggleBreakpoint(lineNumber);
  };

  const getLayerType = (line) => {
    const layerMatch = line.match(/^\s*(\w+)\(/);
    return layerMatch ? layerMatch[1] : null;
  };

  return (
    <div className="breakpoint-manager">
      <div className="breakpoint-header">
        <h3>Code Editor</h3>
        <div className="breakpoint-actions">
          <span className="breakpoint-count">
            {breakpoints.size} breakpoint{breakpoints.size !== 1 ? 's' : ''}
          </span>
          <button
            className="clear-button"
            onClick={onClearAll}
            disabled={breakpoints.size === 0}
            title="Clear All Breakpoints"
          >
            Clear All
          </button>
        </div>
      </div>

      <div className="code-editor" ref={editorRef}>
        <div className="line-numbers">
          {lines.map((_, index) => {
            const lineNumber = index + 1;
            const hasBreakpoint = breakpoints.has(lineNumber);
            const isCurrentLine = currentLine === lineNumber;

            return (
              <div
                key={lineNumber}
                className={`line-number ${hasBreakpoint ? 'has-breakpoint' : ''} ${isCurrentLine ? 'current-line' : ''}`}
                data-line={lineNumber}
                onClick={() => handleLineClick(lineNumber)}
                title={hasBreakpoint ? 'Remove breakpoint' : 'Add breakpoint'}
              >
                <span className="line-num">{lineNumber}</span>
                {hasBreakpoint && (
                  <span className="breakpoint-indicator">●</span>
                )}
                {isCurrentLine && (
                  <span className="current-line-indicator">▶</span>
                )}
              </div>
            );
          })}
        </div>

        <div className="code-lines">
          {lines.map((line, index) => {
            const lineNumber = index + 1;
            const isCurrentLine = currentLine === lineNumber;
            const layerType = getLayerType(line);

            return (
              <div
                key={lineNumber}
                className={`code-line ${isCurrentLine ? 'current-line' : ''} ${layerType ? 'layer-line' : ''}`}
                data-line={lineNumber}
              >
                <pre>
                  <code className={layerType ? `layer-${layerType.toLowerCase()}` : ''}>
                    {line || '\n'}
                  </code>
                </pre>
              </div>
            );
          })}
        </div>
      </div>

      <div className="breakpoint-list">
        <h4>Active Breakpoints</h4>
        {breakpoints.size === 0 ? (
          <div className="no-breakpoints">No breakpoints set</div>
        ) : (
          <ul>
            {Array.from(breakpoints).sort((a, b) => a - b).map((lineNum) => {
              const line = lines[lineNum - 1];
              const layerType = getLayerType(line);
              return (
                <li key={lineNum} className="breakpoint-item">
                  <span className="breakpoint-line">Line {lineNum}</span>
                  {layerType && (
                    <span className="breakpoint-layer">{layerType}</span>
                  )}
                  <button
                    className="remove-breakpoint"
                    onClick={() => onToggleBreakpoint(lineNum)}
                    title="Remove breakpoint"
                  >
                    ×
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </div>
  );
};

export default BreakpointManager;
