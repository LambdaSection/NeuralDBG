# HPO Configuration Interface

A comprehensive UI for configuring and executing hyperparameter optimization with Optuna backend integration.

## Features

### Visual HPO Parameter Definition
- **Parameter Types**: Support for range, log_range, choice, and categorical parameters
- **Dropdowns**: Easy selection of layer types and parameter names
- **Input Fields**: Configurable parameter bounds (min, max, step)
- **Preview**: Real-time preview of Optuna trial.suggest_* code

### Trial Visualization
- **Real-time Charts**: Canvas-based optimization progress visualization
- **Trial Loss**: Line chart showing loss per trial
- **Best Loss**: Cumulative best loss tracking
- **Summary Cards**: Total trials, best loss, best trial number
- **Trial History Table**: Detailed view of all trials with parameters and states

### Integration with neural/hpo/hpo.py
- Direct integration with existing HPO module
- Automatic DSL code modification with HPO annotations
- Support for all Optuna parameter types
- Multi-objective optimization support

### One-Click HPO Execution
- Simple configuration interface
- Backend selection (PyTorch/TensorFlow)
- Dataset selection (MNIST, CIFAR-10, CIFAR-100)
- Device selection (Auto, CPU, CUDA)
- Configurable number of trials

## Components

### HPOConfig.tsx
Main component that orchestrates the HPO configuration and execution.

**Props:**
- `dslCode: string` - The Neural DSL code to optimize
- `onExecuteHPO: (config: HPOConfig) => void` - Callback when HPO is executed

**State:**
- Parameter list with add/update/remove operations
- General settings (trials, dataset, backend, device)
- Trial results and visualization data

### HPOParameterEditor.tsx
Component for editing individual HPO parameters.

**Props:**
- `parameter: HPOParameter` - The parameter configuration
- `onUpdate: (updates: Partial<HPOParameter>) => void` - Update callback
- `onRemove: () => void` - Remove callback

**Features:**
- Dynamic parameter type switching
- Context-aware parameter name selection based on layer type
- Value validation and formatting
- Real-time preview generation

### HPOTrialVisualization.tsx
Component for visualizing optimization progress and results.

**Props:**
- `trials: any[]` - Array of trial results
- `isRunning: boolean` - Whether optimization is running
- `onTrialUpdate: (trialData: any) => void` - Callback for trial updates

**Features:**
- Canvas-based chart rendering
- Trial history table with sorting and filtering
- Best parameters display
- State badges (COMPLETE, RUNNING, PRUNED, FAIL)

## API Integration

### HPOService.ts
Service class for communicating with the HPO backend API.

**Methods:**
- `executeHPO(config: HPOConfig): Promise<HPOStudyResults>` - Execute HPO synchronously
- `streamTrials(...)` - Stream trial results via Server-Sent Events
- `getStudyStatus(studyId: string)` - Get current study status
- `stopStudy(studyId: string)` - Stop a running study

### Backend API (hpo_api.py)
Flask API providing HPO execution endpoints.

**Endpoints:**
- `POST /api/hpo/execute` - Execute HPO with given configuration
- `GET /api/hpo/stream` - Stream trial results in real-time
- `GET /api/hpo/study/<study_id>` - Get study status
- `POST /api/hpo/study/<study_id>/stop` - Stop study
- `POST /api/hpo/validate-dsl` - Validate DSL code
- `GET /api/hpo/parameter-suggestions` - Get parameter suggestions

## Usage Example

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

  const handleExecuteHPO = async (config) => {
    const results = await HPOService.executeHPO(config);
    console.log('Best parameters:', results.best_params);
  };

  return (
    <HPOConfig 
      dslCode={dslCode}
      onExecuteHPO={handleExecuteHPO}
    />
  );
}
```

## Parameter Configuration Example

```typescript
const parameter: HPOParameter = {
  id: '1',
  name: 'dense_units',
  type: 'categorical',
  layerType: 'Dense',
  paramName: 'units',
  values: [32, 64, 128, 256]
};
```

This generates:
```python
trial.suggest_categorical("Dense_units", [32, 64, 128, 256])
```

## Styling

All components use CSS modules for scoped styling with a dark theme:
- Primary color: #667eea (purple-blue gradient)
- Accent color: #50E3C2 (teal)
- Background: #1a1a1a (dark gray)
- Text: #e0e0e0 (light gray)

## Dependencies

- React 18+
- TypeScript 4.9+
- Axios (for API calls)
- Neural DSL HPO module (neural/hpo/hpo.py)
- Optuna (backend optimization)
