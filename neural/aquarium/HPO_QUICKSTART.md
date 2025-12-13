# HPO Configuration Interface - Quick Start Guide

## Overview

The HPO (Hyperparameter Optimization) configuration interface allows you to visually define and execute hyperparameter optimization for your Neural DSL models using Optuna backend.

## Quick Start

### 1. Start the HPO Server

**Windows:**
```bash
cd neural\aquarium
start-hpo-server.bat
```

**Unix/Linux/Mac:**
```bash
cd neural/aquarium
chmod +x start-hpo-server.sh
./start-hpo-server.sh
```

The server will start on `http://localhost:5003`

### 2. Use the HPO Interface

The HPO interface is integrated into Neural Aquarium. Import and use it in your React components:

```typescript
import { HPOConfig } from './components/hpo';

function MyComponent() {
  const dslCode = `
    network MyModel {
      input: (None, 28, 28, 1)
      layers:
        Dense(units=128, activation=relu)
        Dropout(rate=0.5)
        Output(units=10, activation=softmax)
      loss: categorical_crossentropy
      optimizer: Adam(learning_rate=0.001)
    }
  `;

  const handleExecuteHPO = (config) => {
    console.log('HPO Config:', config);
  };

  return <HPOConfig dslCode={dslCode} onExecuteHPO={handleExecuteHPO} />;
}
```

### 3. Configure HPO Parameters

1. Click **"Add Parameter"** to create a new HPO parameter
2. Fill in the details:
   - **Name**: Descriptive name (e.g., "dense_units")
   - **Type**: Choose from:
     - **Range**: Numeric range with step (e.g., 0.1 to 0.9, step 0.1)
     - **Log Range**: Logarithmic scale (e.g., 0.0001 to 0.1)
     - **Categorical**: Discrete choices (e.g., [32, 64, 128, 256])
     - **Choice**: Similar to categorical
   - **Layer Type**: Select the layer (Dense, Conv2D, LSTM, Dropout, Optimizer)
   - **Parameter Name**: Choose the parameter to optimize
   - **Bounds/Values**: Set min/max for ranges or comma-separated values for categorical

3. The preview shows the generated Optuna code

### 4. Configure General Settings

- **Number of Trials**: How many combinations to try (default: 10)
- **Dataset**: Choose MNIST, CIFAR-10, or CIFAR-100
- **Backend**: PyTorch or TensorFlow
- **Device**: Auto, CPU, or CUDA (GPU)

### 5. Execute HPO

1. Click **"Execute HPO"** button
2. Watch real-time progress:
   - Optimization chart shows trial loss and best loss
   - Summary cards display trial count, best loss, best trial number
   - Trial history table shows all attempts with parameters
3. Review results:
   - Best parameters are highlighted in green
   - Full parameter grid displayed at bottom

## Parameter Type Examples

### Range Parameter (Dropout Rate)
```typescript
{
  type: 'range',
  layerType: 'Dropout',
  paramName: 'rate',
  min: 0.3,
  max: 0.7,
  step: 0.1
}
```
Generates: `trial.suggest_float("Dropout_rate", 0.3, 0.7, step=0.1)`

### Log Range Parameter (Learning Rate)
```typescript
{
  type: 'log_range',
  layerType: 'Optimizer',
  paramName: 'learning_rate',
  min: 0.0001,
  max: 0.01
}
```
Generates: `trial.suggest_float("opt_learning_rate", 0.0001, 0.01, log=True)`

### Categorical Parameter (Layer Units)
```typescript
{
  type: 'categorical',
  layerType: 'Dense',
  paramName: 'units',
  values: [64, 128, 256, 512]
}
```
Generates: `trial.suggest_categorical("Dense_units", [64, 128, 256, 512])`

## API Endpoints

The HPO server exposes the following endpoints:

- `POST /api/hpo/execute` - Execute HPO synchronously
- `GET /api/hpo/stream` - Stream trial results (Server-Sent Events)
- `GET /api/hpo/study/<id>` - Get study status
- `POST /api/hpo/study/<id>/stop` - Stop running study
- `POST /api/hpo/validate-dsl` - Validate DSL code
- `GET /api/hpo/parameter-suggestions` - Get parameter templates

## Common Use Cases

### 1. Optimize Dense Layer Units
```
Type: Categorical
Layer: Dense
Parameter: units
Values: 32, 64, 128, 256
```

### 2. Optimize Dropout Rate
```
Type: Range
Layer: Dropout
Parameter: rate
Min: 0.2, Max: 0.8, Step: 0.1
```

### 3. Optimize Learning Rate
```
Type: Log Range
Layer: Optimizer
Parameter: learning_rate
Min: 0.0001, Max: 0.1
```

### 4. Optimize Batch Size
```
Type: Categorical
Layer: Optimizer
Parameter: batch_size
Values: 16, 32, 64, 128
```

## Viewing Results

After HPO execution completes:

1. **Optimization Chart**: Shows loss progression and best loss over trials
2. **Summary Cards**: Display key metrics (total trials, best loss, best trial)
3. **Trial History**: Table with all trials, their parameters, and results
4. **Best Parameters**: Grid showing optimal parameter values

## Tips

- Start with 5-10 trials for quick testing
- Use log_range for learning rates (they vary exponentially)
- Use categorical for discrete choices like layer units
- Monitor the chart to see if optimization is improving
- Best trial is highlighted in green in the table
- Check the preview to understand how parameters are sampled

## Troubleshooting

**"Make sure the backend server is running on port 5003"**
- Start the HPO server using the commands above
- Check if port 5003 is not blocked

**"HPO execution failed"**
- Verify your DSL code is valid
- Check that the dataset and backend are compatible
- Try with fewer trials first

**"Charts not updating"**
- Refresh the page
- Check browser console for errors
- Verify the server is streaming events

## Integration Example

Here's a complete example integrating HPO into a Neural Aquarium component:

```typescript
import React, { useState } from 'react';
import { HPOConfig } from './components/hpo';
import HPOService from './services/HPOService';

function ModelOptimizer() {
  const [results, setResults] = useState(null);
  
  const dslCode = `
    network OptimizedModel {
      input: (None, 28, 28, 1)
      layers:
        Conv2D(filters=32, kernel_size=(3, 3), activation=relu)
        MaxPooling2D(pool_size=(2, 2))
        Flatten()
        Dense(units=128, activation=relu)
        Dropout(rate=0.5)
        Output(units=10, activation=softmax)
      loss: categorical_crossentropy
      optimizer: Adam(learning_rate=0.001)
    }
  `;

  const handleExecuteHPO = async (config) => {
    try {
      const hpoResults = await HPOService.executeHPO(config);
      setResults(hpoResults);
      console.log('Best params:', hpoResults.best_params);
    } catch (error) {
      console.error('HPO failed:', error);
    }
  };

  return (
    <div>
      <h1>Model Optimizer</h1>
      <HPOConfig 
        dslCode={dslCode}
        onExecuteHPO={handleExecuteHPO}
      />
      {results && (
        <div className="results">
          <h2>Optimization Complete!</h2>
          <pre>{JSON.stringify(results.best_params, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}
```

## Next Steps

- Explore the full documentation in `HPO_IMPLEMENTATION.md`
- Check `neural/aquarium/src/components/hpo/README.md` for component details
- Review the backend API in `neural/aquarium/api/hpo_api.py`
- Integrate HPO into your Neural DSL workflow
