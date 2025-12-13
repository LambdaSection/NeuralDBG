# Neural DSL Monaco Editor

A Monaco-based code editor component for Neural DSL with comprehensive language support including syntax highlighting, IntelliSense, error diagnostics, autocomplete, bracket matching, and code folding.

## Features

### 1. Syntax Highlighting
- **Layer Types**: Dense, Conv2D, LSTM, GRU, etc. with distinct highlighting
- **Keywords**: network, input, layers, optimizer, loss, metrics, training, hpo
- **Optimizers**: Adam, SGD, RMSprop, etc.
- **Activation Functions**: relu, sigmoid, tanh, softmax, etc.
- **HPO Types**: range, log_range, choice, categorical
- **Comments**: Single-line (//) and multi-line (/* */)
- **Numbers**: Integer and floating-point literals
- **Strings**: Double and single-quoted strings with escape sequences
- **Operators**: Device specification (@), multiplication (*), parameter assignment (:, =)

### 2. IntelliSense & Autocomplete
- **Context-aware Suggestions**: Different completions based on cursor position
- **Layer Autocomplete**: All supported layer types with parameter hints
- **Parameter Suggestions**: Context-specific parameters for each layer type
- **Activation Functions**: Complete list of supported activations
- **Optimizers**: All optimizers with default parameters
- **Loss Functions**: Comprehensive loss function suggestions
- **HPO Functions**: Hyperparameter optimization function templates
- **Snippet Support**: Multi-line templates for common patterns

### 3. Error Diagnostics
- **Real-time Validation**: Errors and warnings as you type (500ms debounce)
- **Basic Validation**: 
  - Unclosed braces, brackets, and parentheses
  - Unclosed string literals
  - Missing required sections (input, layers)
  - Invalid syntax patterns
  - Missing layer parameters
- **Parser Integration**: Optional backend parser endpoint for advanced validation
- **Severity Levels**: Error, Warning, Info markers
- **Inline Diagnostics**: Error messages on hover
- **Glyph Margin**: Visual indicators for errors
- **Minimap Annotations**: Error positions in minimap

### 4. Editor Features
- **Bracket Matching**: Automatic highlighting of matching brackets
- **Bracket Pair Colorization**: Different colors for nested bracket pairs
- **Auto-closing Pairs**: Automatic closing of brackets, quotes, and braces
- **Code Folding**: 
  - Fold network definitions, layer sections, and nested blocks
  - Indentation-based folding strategy
  - Region markers support (#region/#endregion)
- **Auto-indentation**: Smart indentation on new lines
- **Line Numbers**: Visible and active line highlighting
- **Minimap**: Code overview minimap
- **Word Wrap**: Configurable word wrapping
- **Parameter Hints**: Function parameter information
- **Quick Suggestions**: Automatic suggestion triggering

### 5. Theming
- **Dark Theme**: VS Code dark theme optimized for Neural DSL
- **Light Theme**: VS Code light theme optimized for Neural DSL
- **Custom Token Colors**: Carefully chosen colors for each token type
- **Editor Colors**: Background, selection, cursor, line highlight, etc.

## Usage

### Basic Usage

```tsx
import React, { useState } from 'react';
import { NeuralDSLMonacoEditor } from './components/editor';

function App() {
  const [code, setCode] = useState(`network MyModel {
  input: (None, 28, 28)
  layers:
    Dense(128, activation="relu")
    Dropout(rate=0.2)
    Output(units=10, activation="softmax")
  loss: "categorical_crossentropy"
  optimizer: "Adam"
}`);

  const [errors, setErrors] = useState([]);

  return (
    <NeuralDSLMonacoEditor
      value={code}
      onChange={setCode}
      onValidation={setErrors}
      height="600px"
      theme="dark"
    />
  );
}
```

### With Parser Backend

```tsx
<NeuralDSLMonacoEditor
  value={code}
  onChange={setCode}
  onValidation={(markers) => {
    console.log('Validation errors:', markers);
  }}
  parserEndpoint="http://localhost:5000/api/parse"
  theme="dark"
/>
```

### Read-only Mode

```tsx
<NeuralDSLMonacoEditor
  value={code}
  readOnly={true}
  theme="light"
/>
```

## Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `value` | `string` | `''` | Initial code content |
| `onChange` | `(value: string) => void` | - | Callback when code changes |
| `onValidation` | `(errors: IMarker[]) => void` | - | Callback when validation completes |
| `height` | `string` | `'600px'` | Editor height (CSS value) |
| `theme` | `'light' \| 'dark'` | `'dark'` | Editor theme |
| `readOnly` | `boolean` | `false` | Read-only mode |
| `parserEndpoint` | `string` | `'/api/parse'` | Backend parser endpoint for validation |

## Parser Endpoint Format

The editor can integrate with a backend parser for advanced validation. The endpoint should accept POST requests with the following format:

### Request
```json
{
  "code": "network MyModel { ... }"
}
```

### Response
```json
{
  "success": false,
  "errors": [
    {
      "line": 5,
      "column": 10,
      "message": "Invalid parameter 'units': expected integer",
      "severity": "error",
      "endLine": 5,
      "endColumn": 15
    }
  ],
  "warnings": [
    {
      "line": 2,
      "column": 1,
      "message": "Missing optimizer definition",
      "severity": "warning"
    }
  ]
}
```

## Keyboard Shortcuts

- **Ctrl+Space**: Trigger autocomplete
- **Ctrl+Shift+F**: Format document
- **Ctrl+F**: Find
- **Ctrl+H**: Replace
- **Ctrl+/**: Toggle line comment
- **Shift+Alt+F**: Format selection
- **Ctrl+]**: Indent line
- **Ctrl+[**: Outdent line
- **Ctrl+Shift+K**: Delete line
- **Alt+Up/Down**: Move line up/down
- **Shift+Alt+Up/Down**: Copy line up/down

## Neural DSL Syntax Examples

### Basic Network
```neural
network MNISTClassifier {
  input: (28, 28, 1)
  layers:
    Conv2D(filters=32, kernel_size=(3, 3), activation="relu")
    MaxPooling2D(pool_size=(2, 2))
    Flatten()
    Dense(units=128, activation="relu")
    Dropout(rate=0.5)
    Output(units=10, activation="softmax")
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate=0.001)
  metrics: ["accuracy"]
}
```

### With HPO
```neural
network OptimizedModel {
  input: (32, 32, 3)
  layers:
    Conv2D(
      filters=HPO(choice(32, 64, 128)),
      kernel_size=(3, 3),
      activation="relu"
    )
    Dense(
      units=HPO(range(64, 256)),
      activation=HPO(categorical("relu", "tanh"))
    )
    Output(units=10, activation="softmax")
  optimizer: Adam(learning_rate=HPO(log_range(0.0001, 0.01)))
  hpo: {
    n_trials: 100,
    direction: "maximize"
  }
}
```

### With Device Specification
```neural
network DistributedModel {
  input: (224, 224, 3)
  layers:
    Conv2D(filters=64, kernel_size=(3, 3)) @"GPU:0"
    Dense(units=512) @"GPU:1"
    Output(units=1000, activation="softmax") @"GPU:0"
}
```

### Layer Multiplication
```neural
network DeepNetwork {
  input: (28, 28, 1)
  layers:
    Dense(128, activation="relu") * 3
    Dropout(rate=0.2)
    Output(units=10, activation="softmax")
}
```

## Customization

### Custom Theme

You can modify the theme by editing `theme.ts`:

```typescript
import { NeuralDSLTheme } from './theme';

// Modify colors
NeuralDSLTheme.dark.rules.push({
  token: 'custom-token',
  foreground: 'FF5733'
});
```

### Custom Completions

Add custom completion providers by extending `CompletionProvider`:

```typescript
import { CompletionProvider } from './completionProvider';

class CustomCompletionProvider extends CompletionProvider {
  // Override or add methods
}
```

### Custom Validation

Extend the `DiagnosticsProvider` for custom validation rules:

```typescript
import { DiagnosticsProvider } from './diagnosticsProvider';

class CustomDiagnosticsProvider extends DiagnosticsProvider {
  // Add custom validation logic
}
```

## Architecture

```
editor/
├── NeuralDSLMonacoEditor.tsx   # Main editor component
├── languageConfig.ts            # Monarch language definition & config
├── theme.ts                     # Dark and light themes
├── completionProvider.ts        # IntelliSense & autocomplete
├── diagnosticsProvider.ts       # Error detection & validation
├── index.ts                     # Public exports
├── package.json                 # Dependencies
└── README.md                    # Documentation
```

## Dependencies

- `monaco-editor`: ^0.44.0 - Core Monaco editor
- `react`: ^18.2.0 - React framework
- `react-dom`: ^18.2.0 - React DOM renderer

## Browser Support

- Chrome/Edge: Latest 2 versions
- Firefox: Latest 2 versions
- Safari: Latest 2 versions

## Performance

- **Lazy Loading**: Monaco editor can be loaded asynchronously
- **Debounced Validation**: 500ms debounce for validation
- **Efficient Updates**: Only updates when value changes
- **Web Workers**: Monaco uses web workers for syntax highlighting

## License

MIT

## Contributing

See the main Neural DSL repository for contribution guidelines.
