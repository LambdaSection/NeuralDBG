import React, { useState } from 'react';
import { NeuralDSLMonacoEditor } from '../NeuralDSLMonacoEditor';
import * as monaco from 'monaco-editor';

const exampleCode = `network TransformerModel {
  input: (None, 512)
  layers:
    Transformer(num_heads=8, d_model=512, num_layers=6)
    Dense(units=256, activation="relu")
    Output(units=1000, activation="softmax")
  loss: "categorical_crossentropy"
  optimizer: Adam(learning_rate=HPO(log_range(0.0001, 0.01)))
  hpo: {
    n_trials: 50,
    direction: "maximize",
    metric: "val_accuracy"
  }
}`;

interface ValidationStats {
  totalErrors: number;
  totalWarnings: number;
  lastValidation: Date | null;
}

export const WithParserBackendExample: React.FC = () => {
  const [code, setCode] = useState(exampleCode);
  const [errors, setErrors] = useState<monaco.editor.IMarker[]>([]);
  const [stats, setStats] = useState<ValidationStats>({
    totalErrors: 0,
    totalWarnings: 0,
    lastValidation: null
  });

  const handleValidation = (markers: monaco.editor.IMarker[]) => {
    setErrors(markers);
    setStats({
      totalErrors: markers.filter(m => m.severity === monaco.MarkerSeverity.Error).length,
      totalWarnings: markers.filter(m => m.severity === monaco.MarkerSeverity.Warning).length,
      lastValidation: new Date()
    });
  };

  return (
    <div style={{ display: 'flex', height: '100vh' }}>
      <div style={{ flex: 1 }}>
        <NeuralDSLMonacoEditor
          value={code}
          onChange={setCode}
          onValidation={handleValidation}
          theme="dark"
          parserEndpoint="http://localhost:5000/api/parse"
        />
      </div>
      <div style={{ 
        width: '350px', 
        background: '#252526', 
        color: '#cccccc',
        padding: '20px',
        overflowY: 'auto',
        borderLeft: '1px solid #3c3c3c'
      }}>
        <h2 style={{ marginTop: 0, color: '#fff' }}>Validation Results</h2>
        
        <div style={{ marginBottom: '20px' }}>
          <div style={{ marginBottom: '10px' }}>
            <strong>Statistics</strong>
          </div>
          <div style={{ fontSize: '14px' }}>
            <div style={{ color: '#f48771', marginBottom: '5px' }}>
              Errors: {stats.totalErrors}
            </div>
            <div style={{ color: '#cca700', marginBottom: '5px' }}>
              Warnings: {stats.totalWarnings}
            </div>
            <div style={{ fontSize: '12px', color: '#888', marginTop: '10px' }}>
              Last validated: {stats.lastValidation?.toLocaleTimeString() || 'Never'}
            </div>
          </div>
        </div>

        <div>
          <div style={{ marginBottom: '10px' }}>
            <strong>Issues</strong>
          </div>
          {errors.length === 0 ? (
            <div style={{ color: '#89d185', fontSize: '14px' }}>
              ✓ No issues found
            </div>
          ) : (
            <div style={{ fontSize: '13px' }}>
              {errors.map((error, idx) => (
                <div 
                  key={idx}
                  style={{ 
                    marginBottom: '12px',
                    padding: '10px',
                    background: '#1e1e1e',
                    borderRadius: '4px',
                    borderLeft: `3px solid ${
                      error.severity === monaco.MarkerSeverity.Error 
                        ? '#f48771' 
                        : '#cca700'
                    }`
                  }}
                >
                  <div style={{ 
                    color: error.severity === monaco.MarkerSeverity.Error 
                      ? '#f48771' 
                      : '#cca700',
                    fontWeight: 'bold',
                    marginBottom: '5px'
                  }}>
                    {error.severity === monaco.MarkerSeverity.Error ? 'ERROR' : 'WARNING'}
                  </div>
                  <div style={{ marginBottom: '5px' }}>
                    Line {error.startLineNumber}, Col {error.startColumn}
                  </div>
                  <div style={{ color: '#d4d4d4' }}>
                    {error.message}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
