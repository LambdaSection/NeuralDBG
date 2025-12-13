import React, { useState } from 'react';
import { HPOParameter, HPOConfig as HPOConfigType } from '../../types/hpo';
import HPOParameterEditor from './HPOParameterEditor';
import HPOTrialVisualization from './HPOTrialVisualization';
import './HPOConfig.css';

interface HPOConfigProps {
  dslCode: string;
  onExecuteHPO: (config: HPOConfigType) => void;
}

const HPOConfig: React.FC<HPOConfigProps> = ({ dslCode, onExecuteHPO }) => {
  const [parameters, setParameters] = useState<HPOParameter[]>([]);
  const [nTrials, setNTrials] = useState<number>(10);
  const [dataset, setDataset] = useState<string>('MNIST');
  const [backend, setBackend] = useState<string>('pytorch');
  const [device, setDevice] = useState<string>('auto');
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [trials, setTrials] = useState<any[]>([]);

  const addParameter = () => {
    const newParam: HPOParameter = {
      id: Date.now().toString(),
      name: '',
      type: 'range',
      layerType: 'Dense',
      paramName: 'units',
      min: 0,
      max: 100,
      step: 1,
      values: [],
    };
    setParameters([...parameters, newParam]);
  };

  const updateParameter = (id: string, updates: Partial<HPOParameter>) => {
    setParameters(parameters.map(p => p.id === id ? { ...p, ...updates } : p));
  };

  const removeParameter = (id: string) => {
    setParameters(parameters.filter(p => p.id !== id));
  };

  const handleExecuteHPO = async () => {
    const config: HPOConfigType = {
      parameters,
      nTrials,
      dataset,
      backend,
      device,
      dslCode,
    };

    setIsRunning(true);
    setTrials([]);
    
    try {
      const HPOService = (await import('../../services/HPOService')).default;
      
      await HPOService.streamTrials(
        config,
        (trialData) => {
          handleTrialUpdate(trialData);
        },
        (results) => {
          setIsRunning(false);
          console.log('HPO completed:', results);
        },
        (error) => {
          setIsRunning(false);
          console.error('HPO error:', error);
          alert(`HPO failed: ${error.message}`);
        }
      );
    } catch (error) {
      setIsRunning(false);
      console.error('Failed to execute HPO:', error);
      alert('Failed to execute HPO. Make sure the backend server is running on port 5003.');
    }
    
    onExecuteHPO(config);
  };

  const handleTrialUpdate = (trialData: any) => {
    setTrials(prev => [...prev, trialData]);
  };

  return (
    <div className="hpo-config">
      <div className="hpo-header">
        <h2>Hyperparameter Optimization</h2>
        <p>Configure and execute HPO with Optuna backend</p>
      </div>

      <div className="hpo-content">
        <div className="hpo-config-panel">
          <div className="config-section">
            <h3>General Settings</h3>
            <div className="settings-grid">
              <div className="setting-row">
                <label>Number of Trials</label>
                <input
                  type="number"
                  value={nTrials}
                  onChange={(e) => setNTrials(parseInt(e.target.value))}
                  min={1}
                  max={1000}
                  className="setting-input"
                />
              </div>
              <div className="setting-row">
                <label>Dataset</label>
                <select
                  value={dataset}
                  onChange={(e) => setDataset(e.target.value)}
                  className="setting-input"
                >
                  <option value="MNIST">MNIST</option>
                  <option value="CIFAR10">CIFAR-10</option>
                  <option value="CIFAR100">CIFAR-100</option>
                </select>
              </div>
              <div className="setting-row">
                <label>Backend</label>
                <select
                  value={backend}
                  onChange={(e) => setBackend(e.target.value)}
                  className="setting-input"
                >
                  <option value="pytorch">PyTorch</option>
                  <option value="tensorflow">TensorFlow</option>
                </select>
              </div>
              <div className="setting-row">
                <label>Device</label>
                <select
                  value={device}
                  onChange={(e) => setDevice(e.target.value)}
                  className="setting-input"
                >
                  <option value="auto">Auto</option>
                  <option value="cpu">CPU</option>
                  <option value="cuda">CUDA (GPU)</option>
                </select>
              </div>
            </div>
          </div>

          <div className="config-section">
            <div className="section-header">
              <h3>HPO Parameters</h3>
              <button onClick={addParameter} className="btn btn-primary">
                + Add Parameter
              </button>
            </div>
            <div className="parameters-list">
              {parameters.length === 0 ? (
                <div className="empty-state">
                  <p>No HPO parameters defined.</p>
                  <p>Click "Add Parameter" to start configuring hyperparameter search.</p>
                </div>
              ) : (
                parameters.map(param => (
                  <HPOParameterEditor
                    key={param.id}
                    parameter={param}
                    onUpdate={(updates) => updateParameter(param.id, updates)}
                    onRemove={() => removeParameter(param.id)}
                  />
                ))
              )}
            </div>
          </div>

          <div className="config-section">
            <button
              onClick={handleExecuteHPO}
              disabled={isRunning || parameters.length === 0}
              className="btn btn-execute"
            >
              {isRunning ? '⏳ Running HPO...' : '🚀 Execute HPO'}
            </button>
          </div>
        </div>

        <div className="hpo-visualization-panel">
          <HPOTrialVisualization
            trials={trials}
            isRunning={isRunning}
            onTrialUpdate={handleTrialUpdate}
          />
        </div>
      </div>
    </div>
  );
};

export default HPOConfig;
