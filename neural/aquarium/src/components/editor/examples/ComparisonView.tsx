import React, { useState } from 'react';
import { NeuralDSLMonacoEditor } from '../NeuralDSLMonacoEditor';

const beforeCode = `network SimpleModel {
  input: (28, 28)
  layers:
    Dense(128)
    Dense(64)
    Output(10)
  loss: "mse"
  optimizer: "SGD"
}`;

const afterCode = `network ImprovedModel {
  input: (28, 28, 1)
  layers:
    Conv2D(filters=32, kernel_size=(3, 3), activation="relu")
    MaxPooling2D(pool_size=(2, 2))
    Flatten()
    Dense(units=128, activation="relu")
    Dropout(rate=0.3)
    Output(units=10, activation="softmax")
  loss: "categorical_crossentropy"
  optimizer: Adam(learning_rate=0.001)
  metrics: ["accuracy"]
  training: {
    epochs: 20,
    batch_size: 32
  }
}`;

export const ComparisonViewExample: React.FC = () => {
  const [leftCode, setLeftCode] = useState(beforeCode);
  const [rightCode, setRightCode] = useState(afterCode);

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <div style={{ 
        padding: '20px', 
        background: '#1e1e1e', 
        color: '#fff',
        borderBottom: '1px solid #3c3c3c'
      }}>
        <h1 style={{ margin: 0 }}>Side-by-Side Comparison</h1>
        <p style={{ margin: '10px 0 0 0', color: '#888' }}>
          Compare different versions of your Neural DSL models
        </p>
      </div>
      <div style={{ display: 'flex', flex: 1 }}>
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', borderRight: '1px solid #3c3c3c' }}>
          <div style={{ 
            padding: '10px 20px', 
            background: '#252526', 
            color: '#cccccc',
            borderBottom: '1px solid #3c3c3c',
            fontWeight: 'bold'
          }}>
            Before (Simple Model)
          </div>
          <div style={{ flex: 1 }}>
            <NeuralDSLMonacoEditor
              value={leftCode}
              onChange={setLeftCode}
              theme="dark"
            />
          </div>
        </div>
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <div style={{ 
            padding: '10px 20px', 
            background: '#252526', 
            color: '#cccccc',
            borderBottom: '1px solid #3c3c3c',
            fontWeight: 'bold'
          }}>
            After (Improved Model)
          </div>
          <div style={{ flex: 1 }}>
            <NeuralDSLMonacoEditor
              value={rightCode}
              onChange={setRightCode}
              theme="dark"
            />
          </div>
        </div>
      </div>
    </div>
  );
};
