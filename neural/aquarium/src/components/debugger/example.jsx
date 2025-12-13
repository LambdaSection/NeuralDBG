import React, { useState } from 'react';
import { Debugger } from './index';

const DebuggerExample = () => {
  const [code, setCode] = useState(`network MnistCNN {
  input: (28, 28, 1)
  
  layers:
    Conv2D(32, (3, 3), "relu")
    MaxPooling2D((2, 2))
    Conv2D(64, (3, 3), "relu")
    MaxPooling2D((2, 2))
    Flatten()
    Dense(128, "relu")
    Dropout(0.5)
    Dense(10, "softmax")
  
  loss: "categorical_crossentropy"
  optimizer: Adam(learning_rate=0.001)
  
  train {
    epochs: 10
    batch_size: 32
    validation_split: 0.2
  }
}`);

  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <Debugger code={code} onChange={setCode} />
    </div>
  );
};

export default DebuggerExample;
