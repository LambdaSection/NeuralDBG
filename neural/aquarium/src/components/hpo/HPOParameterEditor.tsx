import React from 'react';
import { HPOParameter, HPOParameterType } from '../../types/hpo';
import './HPOParameterEditor.css';

interface HPOParameterEditorProps {
  parameter: HPOParameter;
  onUpdate: (updates: Partial<HPOParameter>) => void;
  onRemove: () => void;
}

const HPOParameterEditor: React.FC<HPOParameterEditorProps> = ({ 
  parameter, 
  onUpdate, 
  onRemove 
}) => {
  const handleTypeChange = (type: HPOParameterType) => {
    const updates: Partial<HPOParameter> = { type };
    if (type === 'categorical' || type === 'choice') {
      updates.values = parameter.values || ['option1', 'option2'];
      updates.min = undefined;
      updates.max = undefined;
      updates.step = undefined;
    } else {
      updates.values = undefined;
      updates.min = parameter.min || 0;
      updates.max = parameter.max || 100;
      updates.step = parameter.step || 1;
    }
    onUpdate(updates);
  };

  const handleValuesChange = (valuesStr: string) => {
    try {
      const values = valuesStr.split(',').map(v => {
        const trimmed = v.trim();
        const num = Number(trimmed);
        return isNaN(num) ? trimmed : num;
      });
      onUpdate({ values });
    } catch (e) {
      console.error('Invalid values format');
    }
  };

  return (
    <div className="hpo-parameter-editor">
      <div className="parameter-header">
        <input
          type="text"
          value={parameter.name}
          onChange={(e) => onUpdate({ name: e.target.value })}
          placeholder="Parameter name (e.g., dense_units)"
          className="parameter-name-input"
        />
        <button onClick={onRemove} className="btn-remove" title="Remove parameter">
          ✕
        </button>
      </div>

      <div className="parameter-config">
        <div className="config-row">
          <label>Type</label>
          <select
            value={parameter.type}
            onChange={(e) => handleTypeChange(e.target.value as HPOParameterType)}
            className="param-select"
          >
            <option value="range">Range</option>
            <option value="log_range">Log Range</option>
            <option value="choice">Choice</option>
            <option value="categorical">Categorical</option>
          </select>
        </div>

        <div className="config-row">
          <label>Layer Type</label>
          <select
            value={parameter.layerType}
            onChange={(e) => onUpdate({ layerType: e.target.value })}
            className="param-select"
          >
            <option value="Dense">Dense</option>
            <option value="Conv2D">Conv2D</option>
            <option value="LSTM">LSTM</option>
            <option value="Dropout">Dropout</option>
            <option value="BatchNormalization">BatchNormalization</option>
            <option value="Optimizer">Optimizer</option>
          </select>
        </div>

        <div className="config-row">
          <label>Parameter Name</label>
          <select
            value={parameter.paramName}
            onChange={(e) => onUpdate({ paramName: e.target.value })}
            className="param-select"
          >
            {parameter.layerType === 'Dense' && (
              <>
                <option value="units">units</option>
                <option value="activation">activation</option>
              </>
            )}
            {parameter.layerType === 'Conv2D' && (
              <>
                <option value="filters">filters</option>
                <option value="kernel_size">kernel_size</option>
                <option value="activation">activation</option>
              </>
            )}
            {parameter.layerType === 'LSTM' && (
              <>
                <option value="units">units</option>
                <option value="num_layers">num_layers</option>
              </>
            )}
            {parameter.layerType === 'Dropout' && (
              <option value="rate">rate</option>
            )}
            {parameter.layerType === 'BatchNormalization' && (
              <option value="momentum">momentum</option>
            )}
            {parameter.layerType === 'Optimizer' && (
              <>
                <option value="learning_rate">learning_rate</option>
                <option value="batch_size">batch_size</option>
              </>
            )}
          </select>
        </div>

        {(parameter.type === 'range' || parameter.type === 'log_range') && (
          <>
            <div className="config-row">
              <label>Min</label>
              <input
                type="number"
                value={parameter.min || 0}
                onChange={(e) => onUpdate({ min: parseFloat(e.target.value) })}
                className="param-input"
                step="any"
              />
            </div>
            <div className="config-row">
              <label>Max</label>
              <input
                type="number"
                value={parameter.max || 100}
                onChange={(e) => onUpdate({ max: parseFloat(e.target.value) })}
                className="param-input"
                step="any"
              />
            </div>
            {parameter.type === 'range' && (
              <div className="config-row">
                <label>Step</label>
                <input
                  type="number"
                  value={parameter.step || 1}
                  onChange={(e) => onUpdate({ step: parseFloat(e.target.value) })}
                  className="param-input"
                  step="any"
                />
              </div>
            )}
          </>
        )}

        {(parameter.type === 'categorical' || parameter.type === 'choice') && (
          <div className="config-row">
            <label>Values (comma-separated)</label>
            <input
              type="text"
              value={parameter.values?.join(', ') || ''}
              onChange={(e) => handleValuesChange(e.target.value)}
              placeholder="e.g., 32, 64, 128, 256"
              className="param-input"
            />
          </div>
        )}
      </div>

      <div className="parameter-preview">
        <strong>Preview:</strong>
        <code>{formatParameterPreview(parameter)}</code>
      </div>
    </div>
  );
};

const formatParameterPreview = (param: HPOParameter): string => {
  const layerPrefix = param.layerType === 'Optimizer' ? 'opt_' : `${param.layerType}_`;
  const paramName = `${layerPrefix}${param.paramName}`;
  
  if (param.type === 'range') {
    return `trial.suggest_float("${paramName}", ${param.min}, ${param.max}, step=${param.step})`;
  } else if (param.type === 'log_range') {
    return `trial.suggest_float("${paramName}", ${param.min}, ${param.max}, log=True)`;
  } else if (param.type === 'categorical' || param.type === 'choice') {
    return `trial.suggest_categorical("${paramName}", [${param.values?.map(v => typeof v === 'string' ? `"${v}"` : v).join(', ')}])`;
  }
  return '';
};

export default HPOParameterEditor;
