import React, { useEffect, useRef } from 'react';
import './ExecutionTimeline.css';

const ExecutionTimeline = ({ traceData, progress, currentLayer }) => {
  const canvasRef = useRef(null);
  const timelineRef = useRef(null);

  useEffect(() => {
    if (canvasRef.current && traceData && traceData.length > 0) {
      drawTimeline();
    }
  }, [traceData, progress, currentLayer]);

  const drawTimeline = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    if (!traceData || traceData.length === 0) return;

    const padding = 40;
    const timelineHeight = 60;
    const barHeight = 20;
    const forwardY = padding + 20;
    const backwardY = forwardY + timelineHeight + 30;

    const totalTime = traceData.reduce(
      (sum, entry) => sum + (entry.execution_time || 0),
      0
    );

    let currentX = padding;

    ctx.font = '12px Arial';
    ctx.fillStyle = '#cccccc';
    ctx.fillText('Forward Pass', padding, forwardY - 5);
    ctx.fillText('Backward Pass', padding, backwardY - 5);

    traceData.forEach((entry, index) => {
      const executionTime = entry.execution_time || 0;
      const barWidth = totalTime > 0 ? (executionTime / totalTime) * (width - 2 * padding) : 0;

      const isCurrentLayer = currentLayer === entry.layer || currentLayer === index + 1;

      ctx.fillStyle = isCurrentLayer ? '#ffeb3b' : getColorForLayer(entry.layer, index);
      ctx.fillRect(currentX, forwardY, barWidth, barHeight);

      ctx.strokeStyle = '#3e3e42';
      ctx.strokeRect(currentX, forwardY, barWidth, barHeight);

      if (barWidth > 30) {
        ctx.fillStyle = '#1e1e1e';
        ctx.font = 'bold 10px Arial';
        const text = entry.layer || `L${index + 1}`;
        const textWidth = ctx.measureText(text).width;
        ctx.fillText(text, currentX + (barWidth - textWidth) / 2, forwardY + barHeight / 2 + 3);
      }

      currentX += barWidth;
    });

    currentX = padding;
    traceData.slice().reverse().forEach((entry, index) => {
      const executionTime = entry.execution_time || 0;
      const barWidth = totalTime > 0 ? (executionTime / totalTime) * (width - 2 * padding) : 0;

      const isCurrentLayer = currentLayer === entry.layer || currentLayer === traceData.length - index;

      ctx.fillStyle = isCurrentLayer ? '#ffeb3b' : getColorForLayer(entry.layer, index, 0.7);
      ctx.fillRect(currentX, backwardY, barWidth, barHeight);

      ctx.strokeStyle = '#3e3e42';
      ctx.strokeRect(currentX, backwardY, barWidth, barHeight);

      if (barWidth > 30) {
        ctx.fillStyle = '#1e1e1e';
        ctx.font = 'bold 10px Arial';
        const text = entry.layer || `L${traceData.length - index}`;
        const textWidth = ctx.measureText(text).width;
        ctx.fillText(text, currentX + (barWidth - textWidth) / 2, backwardY + barHeight / 2 + 3);
      }

      currentX += barWidth;
    });

    const forwardProgress = progress.forwardPass || 0;
    const backwardProgress = progress.backwardPass || 0;

    if (forwardProgress > 0 && forwardProgress <= 100) {
      const progressX = padding + ((width - 2 * padding) * forwardProgress) / 100;
      ctx.strokeStyle = '#4caf50';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(progressX, forwardY - 5);
      ctx.lineTo(progressX, forwardY + barHeight + 5);
      ctx.stroke();
    }

    if (backwardProgress > 0 && backwardProgress <= 100) {
      const progressX = padding + ((width - 2 * padding) * backwardProgress) / 100;
      ctx.strokeStyle = '#ff9800';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(progressX, backwardY - 5);
      ctx.lineTo(progressX, backwardY + barHeight + 5);
      ctx.stroke();
    }
  };

  const getColorForLayer = (layerName, index, alpha = 1) => {
    const colors = [
      `rgba(255, 107, 107, ${alpha})`,
      `rgba(77, 171, 247, ${alpha})`,
      `rgba(81, 207, 102, ${alpha})`,
      `rgba(255, 212, 59, ${alpha})`,
      `rgba(151, 117, 250, ${alpha})`,
      `rgba(255, 138, 101, ${alpha})`,
    ];

    if (layerName) {
      if (layerName.toLowerCase().includes('conv')) return colors[0];
      if (layerName.toLowerCase().includes('dense') || layerName.toLowerCase().includes('linear')) return colors[1];
      if (layerName.toLowerCase().includes('lstm') || layerName.toLowerCase().includes('gru')) return colors[2];
      if (layerName.toLowerCase().includes('dropout') || layerName.toLowerCase().includes('batch')) return colors[3];
      if (layerName.toLowerCase().includes('pool') || layerName.toLowerCase().includes('flatten')) return colors[4];
    }

    return colors[index % colors.length];
  };

  const formatTime = (seconds) => {
    if (seconds < 0.001) {
      return `${(seconds * 1000000).toFixed(2)} µs`;
    } else if (seconds < 1) {
      return `${(seconds * 1000).toFixed(2)} ms`;
    } else {
      return `${seconds.toFixed(2)} s`;
    }
  };

  const totalExecutionTime = traceData
    ? traceData.reduce((sum, entry) => sum + (entry.execution_time || 0), 0)
    : 0;

  return (
    <div className="execution-timeline">
      <div className="timeline-header">
        <h3>Execution Timeline</h3>
        <div className="timeline-stats">
          <div className="stat">
            <span className="stat-label">Total Time:</span>
            <span className="stat-value">{formatTime(totalExecutionTime)}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Phase:</span>
            <span className={`stat-value phase-${progress.currentPhase}`}>
              {progress.currentPhase || 'idle'}
            </span>
          </div>
        </div>
      </div>

      <div className="timeline-canvas-container" ref={timelineRef}>
        <canvas
          ref={canvasRef}
          width={800}
          height={200}
          className="timeline-canvas"
        />
      </div>

      <div className="timeline-progress">
        <div className="progress-bar-container">
          <div className="progress-label">
            <span>Forward Pass</span>
            <span>{progress.forwardPass?.toFixed(1) || 0}%</span>
          </div>
          <div className="progress-bar">
            <div
              className="progress-fill forward"
              style={{ width: `${progress.forwardPass || 0}%` }}
            />
          </div>
        </div>

        <div className="progress-bar-container">
          <div className="progress-label">
            <span>Backward Pass</span>
            <span>{progress.backwardPass?.toFixed(1) || 0}%</span>
          </div>
          <div className="progress-bar">
            <div
              className="progress-fill backward"
              style={{ width: `${progress.backwardPass || 0}%` }}
            />
          </div>
        </div>
      </div>

      <div className="timeline-legend">
        <h4>Layer Execution Details</h4>
        {traceData && traceData.length > 0 ? (
          <div className="legend-list">
            {traceData.map((entry, index) => {
              const isCurrentLayer = currentLayer === entry.layer || currentLayer === index + 1;
              return (
                <div
                  key={index}
                  className={`legend-item ${isCurrentLayer ? 'current' : ''}`}
                >
                  <div
                    className="legend-color"
                    style={{ backgroundColor: getColorForLayer(entry.layer, index) }}
                  />
                  <span className="legend-name">{entry.layer || `Layer ${index + 1}`}</span>
                  <span className="legend-time">{formatTime(entry.execution_time || 0)}</span>
                  {entry.flops && (
                    <span className="legend-flops">
                      {(entry.flops / 1e6).toFixed(2)} MFLOPs
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        ) : (
          <div className="no-data">No execution data available</div>
        )}
      </div>
    </div>
  );
};

export default ExecutionTimeline;
