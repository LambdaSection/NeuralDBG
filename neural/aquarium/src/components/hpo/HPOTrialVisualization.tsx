import React, { useEffect, useRef } from 'react';
import { HPOTrial } from '../../types/hpo';
import './HPOTrialVisualization.css';

interface HPOTrialVisualizationProps {
  trials: any[];
  isRunning: boolean;
  onTrialUpdate: (trialData: any) => void;
}

const HPOTrialVisualization: React.FC<HPOTrialVisualizationProps> = ({
  trials,
  isRunning,
  onTrialUpdate
}) => {
  const chartCanvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (trials.length > 0 && chartCanvasRef.current) {
      drawOptimizationChart();
    }
  }, [trials]);

  const drawOptimizationChart = () => {
    const canvas = chartCanvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const padding = 40;

    ctx.clearRect(0, 0, width, height);

    if (trials.length === 0) {
      ctx.fillStyle = '#666';
      ctx.font = '14px Arial';
      ctx.textAlign = 'center';
      ctx.fillText('No trials yet', width / 2, height / 2);
      return;
    }

    const values = trials.map(t => t.value);
    const minValue = Math.min(...values);
    const maxValue = Math.max(...values);
    const valueRange = maxValue - minValue || 1;

    ctx.strokeStyle = '#444';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();

    ctx.fillStyle = '#888';
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Trial Number', width / 2, height - 10);
    
    ctx.save();
    ctx.translate(15, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Loss', 0, 0);
    ctx.restore();

    const xStep = (width - 2 * padding) / (trials.length - 1 || 1);
    const yScale = (height - 2 * padding) / valueRange;

    ctx.strokeStyle = '#4A90E2';
    ctx.fillStyle = '#4A90E2';
    ctx.lineWidth = 2;
    ctx.beginPath();

    trials.forEach((trial, idx) => {
      const x = padding + idx * xStep;
      const y = height - padding - (trial.value - minValue) * yScale;

      if (idx === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }

      ctx.fillRect(x - 3, y - 3, 6, 6);
    });

    ctx.stroke();

    let bestValue = trials[0].value;
    ctx.strokeStyle = '#50E3C2';
    ctx.lineWidth = 2;
    ctx.beginPath();

    trials.forEach((trial, idx) => {
      if (trial.value < bestValue) {
        bestValue = trial.value;
      }
      const x = padding + idx * xStep;
      const y = height - padding - (bestValue - minValue) * yScale;

      if (idx === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });

    ctx.stroke();
  };

  const getBestTrial = () => {
    if (trials.length === 0) return null;
    return trials.reduce((best, trial) => 
      trial.value < best.value ? trial : best
    );
  };

  const bestTrial = getBestTrial();

  return (
    <div className="hpo-trial-visualization">
      <div className="visualization-header">
        <h3>Optimization Progress</h3>
        {isRunning && (
          <div className="status-indicator">
            <span className="spinner"></span>
            Running...
          </div>
        )}
      </div>

      <div className="chart-container">
        <canvas 
          ref={chartCanvasRef} 
          width={600} 
          height={350}
          className="optimization-chart"
        />
        <div className="chart-legend">
          <div className="legend-item">
            <span className="legend-color" style={{ backgroundColor: '#4A90E2' }}></span>
            <span>Trial Loss</span>
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ backgroundColor: '#50E3C2' }}></span>
            <span>Best Loss</span>
          </div>
        </div>
      </div>

      <div className="trials-summary">
        <div className="summary-card">
          <div className="summary-label">Total Trials</div>
          <div className="summary-value">{trials.length}</div>
        </div>
        <div className="summary-card">
          <div className="summary-label">Best Loss</div>
          <div className="summary-value">
            {bestTrial ? bestTrial.value.toFixed(4) : 'N/A'}
          </div>
        </div>
        <div className="summary-card">
          <div className="summary-label">Best Trial</div>
          <div className="summary-value">
            {bestTrial ? `#${bestTrial.number}` : 'N/A'}
          </div>
        </div>
      </div>

      <div className="trials-table-container">
        <h4>Trial History</h4>
        <div className="trials-table">
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Loss</th>
                <th>Accuracy</th>
                <th>Parameters</th>
                <th>State</th>
              </tr>
            </thead>
            <tbody>
              {trials.length === 0 ? (
                <tr>
                  <td colSpan={5} style={{ textAlign: 'center', padding: '20px' }}>
                    No trials executed yet
                  </td>
                </tr>
              ) : (
                trials.map((trial, idx) => (
                  <tr key={idx} className={trial === bestTrial ? 'best-trial' : ''}>
                    <td>{trial.number}</td>
                    <td>{trial.value.toFixed(4)}</td>
                    <td>{trial.accuracy ? trial.accuracy.toFixed(4) : 'N/A'}</td>
                    <td>
                      <div className="params-cell">
                        {trial.params && Object.entries(trial.params).slice(0, 2).map(([k, v]: [string, any]) => (
                          <span key={k} className="param-tag">
                            {k}: {typeof v === 'number' ? v.toFixed(3) : v}
                          </span>
                        ))}
                        {trial.params && Object.keys(trial.params).length > 2 && (
                          <span className="param-tag">+{Object.keys(trial.params).length - 2}</span>
                        )}
                      </div>
                    </td>
                    <td>
                      <span className={`state-badge state-${trial.state || 'complete'}`}>
                        {trial.state || 'COMPLETE'}
                      </span>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {bestTrial && (
        <div className="best-params-section">
          <h4>Best Parameters</h4>
          <div className="params-grid">
            {bestTrial.params && Object.entries(bestTrial.params).map(([key, value]: [string, any]) => (
              <div key={key} className="param-item">
                <span className="param-key">{key}</span>
                <span className="param-value">
                  {typeof value === 'number' ? value.toFixed(6) : value}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default HPOTrialVisualization;
