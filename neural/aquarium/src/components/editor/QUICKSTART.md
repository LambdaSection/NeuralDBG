# Quick Start Guide - Neural DSL Monaco Editor

Get started with the Neural DSL Monaco Editor in 5 minutes.

## Installation

```bash
# Install dependencies
npm install monaco-editor react react-dom

# Install dev dependencies (optional, for TypeScript)
npm install --save-dev @types/react @types/react-dom typescript
```

## Basic Setup

### 1. Copy the Editor Component

Copy the entire `editor/` directory into your React project:

```bash
cp -r neural/aquarium/src/components/editor your-project/src/components/
```

### 2. Configure Your Build Tool

#### Using Webpack

```bash
npm install --save-dev monaco-editor-webpack-plugin
```

Update `webpack.config.js`:

```javascript
const MonacoWebpackPlugin = require('monaco-editor-webpack-plugin');

module.exports = {
  plugins: [
    new MonacoWebpackPlugin({
      languages: []  // We use a custom language
    })
  ]
};
```

#### Using Vite

```bash
npm install --save-dev vite-plugin-monaco-editor
```

Update `vite.config.ts`:

```typescript
import monacoEditorPlugin from 'vite-plugin-monaco-editor';

export default {
  plugins: [monacoEditorPlugin({})]
};
```

### 3. Use the Editor

```tsx
import React, { useState } from 'react';
import { NeuralDSLMonacoEditor } from './components/editor';

function App() {
  const [code, setCode] = useState(`network MyModel {
  input: (None, 28, 28)
  layers:
    Dense(128, activation="relu")
    Output(units=10, activation="softmax")
  loss: "categorical_crossentropy"
}`);

  return (
    <div style={{ height: '100vh' }}>
      <NeuralDSLMonacoEditor
        value={code}
        onChange={setCode}
        theme="dark"
      />
    </div>
  );
}

export default App;
```

## Features at a Glance

### Syntax Highlighting
All Neural DSL syntax is automatically highlighted with appropriate colors.

### Autocomplete
Press `Ctrl+Space` to see suggestions:
- Layer types (Dense, Conv2D, LSTM, etc.)
- Parameters for each layer
- Activation functions
- Optimizers and loss functions
- HPO functions

### Error Detection
Errors appear inline as you type:
- Red underlines for syntax errors
- Yellow underlines for warnings
- Hover over errors for details

### Bracket Matching
Click on any bracket to see its matching pair highlighted.

### Code Folding
Click the arrows in the left margin to fold/unfold code blocks.

## Common Use Cases

### Read-Only Display

```tsx
<NeuralDSLMonacoEditor
  value={code}
  readOnly={true}
  theme="light"
/>
```

### With Validation Callback

```tsx
function MyEditor() {
  const [errors, setErrors] = useState([]);

  return (
    <>
      <div>Errors: {errors.length}</div>
      <NeuralDSLMonacoEditor
        value={code}
        onChange={setCode}
        onValidation={setErrors}
      />
    </>
  );
}
```

### With Backend Parser

```tsx
<NeuralDSLMonacoEditor
  value={code}
  onChange={setCode}
  parserEndpoint="http://localhost:5000/api/parse"
/>
```

## Keyboard Shortcuts

- **Ctrl+Space**: Trigger autocomplete
- **Ctrl+F**: Find
- **Ctrl+H**: Replace
- **Ctrl+/**: Toggle comment
- **Ctrl+Shift+F**: Format document
- **Alt+Up/Down**: Move line
- **Shift+Alt+Up/Down**: Copy line

## Next Steps

- Read [README.md](./README.md) for detailed features
- Check [INTEGRATION.md](./INTEGRATION.md) for advanced integration
- Browse [examples/](./examples/) for working code samples
- Customize themes in [theme.ts](./theme.ts)
- Add custom snippets in [utils/snippets.ts](./utils/snippets.ts)

## Troubleshooting

### Monaco doesn't load
- Ensure webpack/vite plugin is installed
- Check browser console for errors
- Verify `monaco-editor` is in dependencies, not devDependencies

### No syntax highlighting
- Check that language registration happens before editor creation
- Verify no errors in browser console
- Try clearing cache and rebuilding

### Autocomplete not working
- Press `Ctrl+Space` to manually trigger
- Check that `suggest` options are enabled
- Verify completion provider is registered

## Need Help?

- See [README.md](./README.md) for full documentation
- Check [INTEGRATION.md](./INTEGRATION.md) for integration issues
- Review [examples/](./examples/) for working implementations
- Read [IMPLEMENTATION.md](./IMPLEMENTATION.md) for architecture details

## Demo Examples

The `examples/` directory contains three ready-to-run demos:

1. **BasicUsage.tsx** - Simple editor with error display
2. **WithParserBackend.tsx** - Editor with validation sidebar
3. **ComparisonView.tsx** - Side-by-side comparison view

Try them out to see the editor in action!
