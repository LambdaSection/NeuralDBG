import React, { useState } from 'react';
import { NeuralDSLMonacoEditor } from '../NeuralDSLMonacoEditor';
import * as monaco from 'monaco-editor';

const defaultCode = `network MNISTClassifier {
  input: (28, 28, 1)
  layers:
    Conv2D(filters=32, kernel_size=(3, 3), activation="relu")
    MaxPooling2D(pool_size=(2, 2))
    Conv2D(filters=64, kernel_size=(3, 3), activation="relu")
    MaxPooling2D(pool_size=(2, 2))
    Flatten()
    Dense(units=128, activation="relu")
    Dropout(rate=0.5)
    Output(units=10, activation="softmax")
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate=0.001)
  metrics: ["accuracy"]
  training: {
    epochs: 15,
    batch_size: 64,
    validation_split: 0.2
  }
}`;

export const BasicUsageExample: React.FC = () => {
  const [code, setCode] = useState(defaultCode);
  const [errors, setErrors] = useState<monaco.editor.IMarker[]>([]);

  return (
    <div style={{ width: '100%', height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <div style={{ padding: '20px', background: '#1e1e1e', color: '#fff' }}>
        <h1>Neural DSL Editor - Basic Usage</h1>
        <div style={{ marginTop: '10px' }}>
          {errors.length > 0 ? (
            <div style={{ color: '#ff6b6b' }}>
              <strong>Errors: {errors.length}</strong>
              <ul style={{ margin: '5px 0', paddingLeft: '20px' }}>
                {errors.slice(0, 5).map((error, idx) => (
                  <li key={idx}>
                    Line {error.startLineNumber}: {error.message}
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            <div style={{ color: '#51cf66' }}>
              <strong>✓ No errors detected</strong>
            </div>
          )}
        </div>
      </div>
      <div style={{ flex: 1 }}>
        <NeuralDSLMonacoEditor
          value={code}
          onChange={setCode}
          onValidation={setErrors}
          theme="dark"
        />
      </div>
    </div>
  );
};
