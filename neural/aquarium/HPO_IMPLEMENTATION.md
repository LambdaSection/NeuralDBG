# HPO Configuration Interface Implementation

## Overview

This document describes the implementation of the Hyperparameter Optimization (HPO) configuration interface in Neural Aquarium. The interface provides a visual way to define HPO parameters, execute optimization with Optuna backend, and visualize trial results in real-time.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Neural Aquarium UI                       │
│  ┌───────────────────────────────────────────────────────┐  │
│  │            HPO Configuration Panel                    │  │
│  │  - Parameter Editor                                   │  │
│  │  - General Settings (trials, dataset, backend)       │  │
│  │  - Execute Button                                     │  │
│  └───────────────────────────────────────────────────────┘  │
│                            ↓                                 │
│  ┌───────────────────────────────────────────────────────┐  │
│  │         HPO Trial Visualization Panel                 │  │
│  │  - Real-time Charts                                   │  │
│  │  - Trial History Table                                │  │
│  │  - Best Parameters Display                            │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ↓
                      HTTP/SSE API
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  HPO API Server (Flask)                     │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  - /api/hpo/execute (POST)                            │  │
│  │  - /api/hpo/stream (GET - SSE)                        │  │
│  │  - /api/hpo/validate-dsl (POST)                       │  │
│  │  - /api/hpo/parameter-suggestions (GET)               │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              neural/hpo/hpo.py (Optuna Backend)             │
│  - optimize_and_return()                                    │
│  - create_dynamic_model()                                   │
│  - train_model()                                            │
└─────────────────────────────────────────────────────────────┘
```

## Components

### Frontend Components

#### 1. HPOConfig.tsx
**Location**: `neural/aquarium/src/components/hpo/HPOConfig.tsx`

Main orchestration component that manages:
- HPO parameter list (add, update, remove)
- General settings (trials, dataset, backend, device)
- Trial execution and results management
- Integration with HPOService for API communication

**Key Features**:
- Dynamic parameter management with unique IDs
- Real-time trial streaming via Server-Sent Events
- Error handling and user feedback
- State management for running optimization

#### 2. HPOParameterEditor.tsx
**Location**: `neural/aquarium/src/components/hpo/HPOParameterEditor.tsx`

Individual parameter configuration component:
- Parameter type selection (range, log_range, categorical, choice)
- Layer type selection (Dense, Conv2D, LSTM, Dropout, etc.)
- Context-aware parameter name dropdown
- Dynamic field rendering based on parameter type
- Real-time Optuna code preview

**Supported Parameters**:
- **Dense**: units, activation
- **Conv2D**: filters, kernel_size, activation
- **LSTM**: units, num_layers
- **Dropout**: rate
- **BatchNormalization**: momentum
- **Optimizer**: learning_rate, batch_size

#### 3. HPOTrialVisualization.tsx
**Location**: `neural/aquarium/src/components/hpo/HPOTrialVisualization.tsx`

Trial visualization and monitoring component:
- Canvas-based optimization progress chart
- Trial loss and best loss tracking
- Summary cards (total trials, best loss, best trial)
- Detailed trial history table
- Best parameters display grid

**Chart Features**:
- Dual-line chart (trial loss + best loss)
- Auto-scaling based on value range
- Axis labels and legends
- Responsive canvas rendering

### Backend Components

#### 4. hpo_api.py
**Location**: `neural/aquarium/api/hpo_api.py`

Flask API server providing HPO execution endpoints:

**Endpoints**:

1. **POST /api/hpo/execute**
   - Execute HPO with given configuration
   - Returns study results synchronously
   - Integrates with neural/hpo/hpo.py

2. **GET /api/hpo/stream**
   - Stream trial results via Server-Sent Events
   - Real-time trial updates during optimization
   - Completion event with final results

3. **GET /api/hpo/study/<study_id>**
   - Get status of a running/completed study
   - Returns trial history and best parameters

4. **POST /api/hpo/study/<study_id>/stop**
   - Stop a running optimization study
   - Graceful termination

5. **POST /api/hpo/validate-dsl**
   - Validate DSL code before execution
   - Extract HPO parameters from DSL
   - Return layer count and parameter list

6. **GET /api/hpo/parameter-suggestions**
   - Get suggested parameter configurations
   - Layer-specific parameter templates
   - Common value ranges and types

**Key Functions**:
- `convert_hpo_config_to_dsl()`: Converts UI config to DSL with HPO annotations
- Study management with active_studies dictionary
- Error handling and exception propagation

#### 5. HPOService.ts
**Location**: `neural/aquarium/src/services/HPOService.ts`

TypeScript service class for API communication:
- `executeHPO()`: Synchronous HPO execution
- `streamTrials()`: Real-time trial streaming with callbacks
- `getStudyStatus()`: Study status monitoring
- `stopStudy()`: Study termination

**Features**:
- Axios-based HTTP requests
- EventSource for Server-Sent Events
- Error handling and retry logic
- Type-safe API responses

### Type Definitions

#### 6. hpo.ts
**Location**: `neural/aquarium/src/types/hpo.ts`

TypeScript type definitions:

```typescript
export type HPOParameterType = 'range' | 'log_range' | 'choice' | 'categorical';

export interface HPOParameter {
  id: string;
  name: string;
  type: HPOParameterType;
  layerType: string;
  paramName: string;
  min?: number;
  max?: number;
  step?: number;
  values?: any[];
}

export interface HPOConfig {
  parameters: HPOParameter[];
  nTrials: number;
  dataset: string;
  backend: string;
  device: string;
  dslCode: string;
}

export interface HPOTrial {
  number: number;
  values: Record<string, any>;
  value: number | number[];
  state: 'RUNNING' | 'COMPLETE' | 'PRUNED' | 'FAIL';
}

export interface HPOStudyResults {
  best_trial: HPOTrial;
  best_params: Record<string, any>;
  trials: HPOTrial[];
}
```

## Integration with neural/hpo/hpo.py

The UI integrates seamlessly with the existing HPO module:

1. **DSL Annotation**: UI-configured parameters are injected into DSL code as HPO annotations
2. **Parser Integration**: ModelTransformer.parse_network_with_hpo() extracts parameters
3. **Optuna Execution**: optimize_and_return() executes the optimization
4. **Result Processing**: Best parameters are returned and displayed

**Example DSL with HPO**:
```python
network HPOModel {
    input: (None, 28, 28, 1)
    layers:
        Dense(units={"hpo": {"type": "categorical", "values": [64, 128, 256]}}, activation=relu)
        Dropout(rate={"hpo": {"type": "range", "start": 0.3, "end": 0.7, "step": 0.1}})
        Output(units=10, activation=softmax)
    loss: categorical_crossentropy
    optimizer: Adam(learning_rate={"hpo": {"type": "log_range", "start": 0.0001, "end": 0.01}})
}
```

## Styling

All components follow a consistent dark theme:

**Color Palette**:
- Primary: `#667eea` (purple-blue gradient)
- Accent: `#50E3C2` (teal)
- Background: `#1a1a1a` (dark gray)
- Secondary Background: `#252525`
- Border: `#333`
- Text: `#e0e0e0` (light gray)
- Muted Text: `#888`

**CSS Files**:
- `HPOConfig.css`: Main layout and configuration panel
- `HPOParameterEditor.css`: Parameter editor styling
- `HPOTrialVisualization.css`: Visualization panel and charts

## Usage

### Starting the HPO Server

**Windows**:
```bash
cd neural/aquarium
start-hpo-server.bat
```

**Unix/Linux/Mac**:
```bash
cd neural/aquarium
chmod +x start-hpo-server.sh
./start-hpo-server.sh
```

**Python Direct**:
```bash
python neural/aquarium/start-hpo-server.py --port 5003 --debug
```

### Using the UI

1. **Define Parameters**:
   - Click "Add Parameter"
   - Select parameter type (range, log_range, categorical)
   - Choose layer type and parameter name
   - Configure bounds or values
   - Preview shows Optuna code

2. **Configure Settings**:
   - Set number of trials (1-1000)
   - Select dataset (MNIST, CIFAR-10, CIFAR-100)
   - Choose backend (PyTorch, TensorFlow)
   - Select device (Auto, CPU, CUDA)

3. **Execute HPO**:
   - Click "Execute HPO"
   - Monitor real-time trial progress
   - View optimization chart
   - Check trial history table
   - Review best parameters

4. **Analyze Results**:
   - Best loss and accuracy displayed
   - Parameter values shown in grid
   - Trial history with all attempts
   - State badges for trial status

## API Examples

### Execute HPO
```bash
curl -X POST http://localhost:5003/api/hpo/execute \
  -H "Content-Type: application/json" \
  -d '{
    "dslCode": "network MyModel { ... }",
    "parameters": [...],
    "nTrials": 10,
    "dataset": "MNIST",
    "backend": "pytorch",
    "device": "auto"
  }'
```

### Stream Trials
```bash
curl -N http://localhost:5003/api/hpo/stream?config={...}
```

### Validate DSL
```bash
curl -X POST http://localhost:5003/api/hpo/validate-dsl \
  -H "Content-Type: application/json" \
  -d '{"dslCode": "network MyModel { ... }"}'
```

## Dependencies

**Frontend**:
- React 18+
- TypeScript 4.9+
- Axios (HTTP client)

**Backend**:
- Flask (web framework)
- Flask-CORS (CORS support)
- Optuna (optimization)
- PyTorch/TensorFlow (backends)

## Testing

### Manual Testing Checklist

- [ ] Add/remove HPO parameters
- [ ] Change parameter types and verify field updates
- [ ] Execute HPO with different configurations
- [ ] Verify real-time trial updates
- [ ] Check chart rendering and updates
- [ ] Test with different datasets and backends
- [ ] Verify error handling (server down, invalid DSL)
- [ ] Check best parameters display
- [ ] Validate DSL code before execution
- [ ] Test stop functionality for running studies

## Future Enhancements

1. **Multi-objective Optimization**: Support multiple metrics (loss, accuracy, precision)
2. **Pruning Strategies**: Integrate Optuna pruners (Median, Hyperband)
3. **Study Persistence**: Save and load optimization studies
4. **Parallel Execution**: Run multiple trials in parallel
5. **Advanced Visualizations**: Parameter importance, slice plots, contour plots
6. **Export Results**: Save best parameters to file or apply to DSL
7. **Comparison View**: Compare multiple HPO runs
8. **Budget Management**: Time-based or trial-based budgets

## Troubleshooting

**Issue**: HPO server not starting
- **Solution**: Check if port 5003 is available, ensure Flask is installed

**Issue**: Trials not updating in UI
- **Solution**: Verify backend server is running, check browser console for errors

**Issue**: Invalid DSL error
- **Solution**: Use validate-dsl endpoint to check syntax before execution

**Issue**: Out of memory during optimization
- **Solution**: Reduce batch size, use smaller model, or enable CPU mode

**Issue**: Charts not rendering
- **Solution**: Check canvas support in browser, verify trial data format

## File Structure

```
neural/aquarium/
├── src/
│   ├── components/
│   │   └── hpo/
│   │       ├── HPOConfig.tsx
│   │       ├── HPOConfig.css
│   │       ├── HPOParameterEditor.tsx
│   │       ├── HPOParameterEditor.css
│   │       ├── HPOTrialVisualization.tsx
│   │       ├── HPOTrialVisualization.css
│   │       ├── index.ts
│   │       └── README.md
│   ├── services/
│   │   └── HPOService.ts
│   └── types/
│       ├── hpo.ts
│       └── index.ts
├── api/
│   └── hpo_api.py
├── start-hpo-server.py
├── start-hpo-server.bat
├── start-hpo-server.sh
└── HPO_IMPLEMENTATION.md
```

## Conclusion

The HPO configuration interface provides a comprehensive, user-friendly way to configure and execute hyperparameter optimization directly from the Neural Aquarium UI. With visual parameter definition, real-time trial monitoring, and seamless integration with the existing HPO module, users can efficiently optimize their neural network models without writing code.
