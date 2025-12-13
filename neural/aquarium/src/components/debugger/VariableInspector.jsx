import React, { useState } from 'react';
import './VariableInspector.css';

const VariableInspector = ({ variables, traceData, currentLayer }) => {
  const [selectedTab, setSelectedTab] = useState('activations');
  const [expandedLayers, setExpandedLayers] = useState(new Set());
  const [selectedVariable, setSelectedVariable] = useState(null);

  const toggleLayer = (layerName) => {
    const newExpanded = new Set(expandedLayers);
    if (newExpanded.has(layerName)) {
      newExpanded.delete(layerName);
    } else {
      newExpanded.add(layerName);
    }
    setExpandedLayers(newExpanded);
  };

  const formatTensorShape = (shape) => {
    if (Array.isArray(shape)) {
      return `[${shape.join(', ')}]`;
    }
    return String(shape);
  };

  const formatNumber = (num) => {
    if (typeof num === 'number') {
      if (Math.abs(num) < 0.01 && num !== 0) {
        return num.toExponential(4);
      }
      return num.toFixed(4);
    }
    return String(num);
  };

  const renderActivations = () => {
    if (!traceData || traceData.length === 0) {
      return <div className="no-data">No activation data available</div>;
    }

    return (
      <div className="activations-list">
        {traceData.map((entry, index) => {
          const layerName = entry.layer || `Layer ${index}`;
          const isExpanded = expandedLayers.has(layerName);
          const isCurrent = currentLayer === layerName || currentLayer === index + 1;

          return (
            <div
              key={index}
              className={`layer-entry ${isCurrent ? 'current' : ''}`}
            >
              <div
                className="layer-header"
                onClick={() => toggleLayer(layerName)}
              >
                <span className="expand-icon">{isExpanded ? '▼' : '▶'}</span>
                <span className="layer-name">{layerName}</span>
                {isCurrent && <span className="current-badge">Current</span>}
              </div>

              {isExpanded && (
                <div className="layer-details">
                  {entry.mean_activation !== undefined && (
                    <div className="detail-row">
                      <span className="detail-label">Mean Activation:</span>
                      <span className="detail-value">
                        {formatNumber(entry.mean_activation)}
                      </span>
                    </div>
                  )}

                  {entry.std_activation !== undefined && (
                    <div className="detail-row">
                      <span className="detail-label">Std Activation:</span>
                      <span className="detail-value">
                        {formatNumber(entry.std_activation)}
                      </span>
                    </div>
                  )}

                  {entry.max_activation !== undefined && (
                    <div className="detail-row">
                      <span className="detail-label">Max Activation:</span>
                      <span className="detail-value">
                        {formatNumber(entry.max_activation)}
                      </span>
                    </div>
                  )}

                  {entry.min_activation !== undefined && (
                    <div className="detail-row">
                      <span className="detail-label">Min Activation:</span>
                      <span className="detail-value">
                        {formatNumber(entry.min_activation)}
                      </span>
                    </div>
                  )}

                  {entry.output_shape && (
                    <div className="detail-row">
                      <span className="detail-label">Output Shape:</span>
                      <span className="detail-value shape">
                        {formatTensorShape(entry.output_shape)}
                      </span>
                    </div>
                  )}

                  {entry.dead_ratio !== undefined && (
                    <div className="detail-row">
                      <span className="detail-label">Dead Neurons:</span>
                      <span className={`detail-value ${entry.dead_ratio > 0.3 ? 'warning' : ''}`}>
                        {(entry.dead_ratio * 100).toFixed(2)}%
                      </span>
                    </div>
                  )}

                  {entry.anomaly && (
                    <div className="detail-row anomaly">
                      <span className="detail-label">⚠ Anomaly:</span>
                      <span className="detail-value">Detected</span>
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    );
  };

  const renderGradients = () => {
    if (!traceData || traceData.length === 0) {
      return <div className="no-data">No gradient data available</div>;
    }

    const layersWithGradients = traceData.filter(
      entry => entry.grad_norm !== undefined || entry.grad_mean !== undefined
    );

    if (layersWithGradients.length === 0) {
      return <div className="no-data">No gradient data available</div>;
    }

    return (
      <div className="gradients-list">
        {layersWithGradients.map((entry, index) => {
          const layerName = entry.layer || `Layer ${index}`;
          const isExpanded = expandedLayers.has(layerName);
          const isCurrent = currentLayer === layerName;

          return (
            <div
              key={index}
              className={`layer-entry ${isCurrent ? 'current' : ''}`}
            >
              <div
                className="layer-header"
                onClick={() => toggleLayer(layerName)}
              >
                <span className="expand-icon">{isExpanded ? '▼' : '▶'}</span>
                <span className="layer-name">{layerName}</span>
                {isCurrent && <span className="current-badge">Current</span>}
              </div>

              {isExpanded && (
                <div className="layer-details">
                  {entry.grad_norm !== undefined && (
                    <div className="detail-row">
                      <span className="detail-label">Gradient Norm:</span>
                      <span className="detail-value">
                        {formatNumber(entry.grad_norm)}
                      </span>
                    </div>
                  )}

                  {entry.grad_mean !== undefined && (
                    <div className="detail-row">
                      <span className="detail-label">Gradient Mean:</span>
                      <span className="detail-value">
                        {formatNumber(entry.grad_mean)}
                      </span>
                    </div>
                  )}

                  {entry.grad_std !== undefined && (
                    <div className="detail-row">
                      <span className="detail-label">Gradient Std:</span>
                      <span className="detail-value">
                        {formatNumber(entry.grad_std)}
                      </span>
                    </div>
                  )}

                  {entry.grad_max !== undefined && (
                    <div className="detail-row">
                      <span className="detail-label">Gradient Max:</span>
                      <span className="detail-value">
                        {formatNumber(entry.grad_max)}
                      </span>
                    </div>
                  )}

                  {entry.grad_min !== undefined && (
                    <div className="detail-row">
                      <span className="detail-label">Gradient Min:</span>
                      <span className="detail-value">
                        {formatNumber(entry.grad_min)}
                      </span>
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    );
  };

  const renderVariables = () => {
    if (!variables || Object.keys(variables).length === 0) {
      return <div className="no-data">No variables available</div>;
    }

    return (
      <div className="variables-list">
        {Object.entries(variables).map(([name, value]) => (
          <div
            key={name}
            className="variable-entry"
            onClick={() => setSelectedVariable(name)}
          >
            <div className="variable-header">
              <span className="variable-name">{name}</span>
              <span className="variable-type">
                {typeof value === 'object' && value !== null
                  ? Array.isArray(value)
                    ? 'array'
                    : 'object'
                  : typeof value}
              </span>
            </div>
            <div className="variable-value">
              {typeof value === 'object'
                ? JSON.stringify(value, null, 2)
                : String(value)}
            </div>
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="variable-inspector">
      <div className="inspector-header">
        <h3>Variable Inspector</h3>
      </div>

      <div className="inspector-tabs">
        <button
          className={`tab ${selectedTab === 'activations' ? 'active' : ''}`}
          onClick={() => setSelectedTab('activations')}
        >
          Layer Activations
        </button>
        <button
          className={`tab ${selectedTab === 'gradients' ? 'active' : ''}`}
          onClick={() => setSelectedTab('gradients')}
        >
          Gradients
        </button>
        <button
          className={`tab ${selectedTab === 'variables' ? 'active' : ''}`}
          onClick={() => setSelectedTab('variables')}
        >
          Variables
        </button>
      </div>

      <div className="inspector-content">
        {selectedTab === 'activations' && renderActivations()}
        {selectedTab === 'gradients' && renderGradients()}
        {selectedTab === 'variables' && renderVariables()}
      </div>
    </div>
  );
};

export default VariableInspector;
