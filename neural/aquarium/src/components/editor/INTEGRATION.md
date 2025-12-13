# Neural DSL Monaco Editor - Integration Guide

This guide explains how to integrate the Monaco-based Neural DSL editor into your application.

## Prerequisites

- Node.js 16+ and npm/yarn
- React 16.8+ (for hooks support)
- TypeScript 4.5+ (recommended)

## Installation

### 1. Install Dependencies

```bash
npm install monaco-editor react react-dom
# or
yarn add monaco-editor react react-dom
```

### 2. Install Dev Dependencies (if using TypeScript)

```bash
npm install --save-dev @types/react @types/react-dom typescript
# or
yarn add --dev @types/react @types/react-dom typescript
```

### 3. Configure Webpack (if not using Create React App)

Monaco Editor requires special webpack configuration. Add the `monaco-editor-webpack-plugin`:

```bash
npm install --save-dev monaco-editor-webpack-plugin
```

Update your `webpack.config.js`:

```javascript
const MonacoWebpackPlugin = require('monaco-editor-webpack-plugin');

module.exports = {
  // ... other config
  plugins: [
    new MonacoWebpackPlugin({
      languages: [], // We're using a custom language
      features: [
        'bracketMatching',
        'clipboard',
        'codeAction',
        'codelens',
        'colorDetector',
        'comment',
        'contextmenu',
        'coreCommands',
        'cursorUndo',
        'find',
        'folding',
        'fontZoom',
        'format',
        'gotoError',
        'gotoLine',
        'hover',
        'inPlaceReplace',
        'inspectTokens',
        'linesOperations',
        'links',
        'multicursor',
        'parameterHints',
        'quickCommand',
        'quickOutline',
        'referenceSearch',
        'rename',
        'smartSelect',
        'snippets',
        'suggest',
        'toggleHighContrast',
        'toggleTabFocusMode',
        'transpose',
        'wordHighlighter',
        'wordOperations',
        'wordPartOperations'
      ]
    })
  ]
};
```

### 4. Configure Vite (Alternative to Webpack)

If you're using Vite, install the plugin:

```bash
npm install --save-dev vite-plugin-monaco-editor
```

Update `vite.config.ts`:

```typescript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import monacoEditorPlugin from 'vite-plugin-monaco-editor';

export default defineConfig({
  plugins: [
    react(),
    monacoEditorPlugin({})
  ]
});
```

## Basic Integration

### Simple React Component

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

## Advanced Integration

### With Backend Parser

```tsx
import React, { useState, useCallback } from 'react';
import { NeuralDSLMonacoEditor } from './components/editor';
import * as monaco from 'monaco-editor';

function EditorWithValidation() {
  const [code, setCode] = useState('');
  const [errors, setErrors] = useState<monaco.editor.IMarker[]>([]);
  const [isValidating, setIsValidating] = useState(false);

  const handleValidation = useCallback((markers: monaco.editor.IMarker[]) => {
    setErrors(markers);
    setIsValidating(false);
  }, []);

  const handleChange = useCallback((newCode: string) => {
    setCode(newCode);
    setIsValidating(true);
  }, []);

  return (
    <div>
      <div style={{ padding: '10px', background: '#f0f0f0' }}>
        <span>Status: {isValidating ? 'Validating...' : 'Ready'}</span>
        <span style={{ marginLeft: '20px' }}>
          Errors: {errors.filter(e => e.severity === monaco.MarkerSeverity.Error).length}
        </span>
      </div>
      <NeuralDSLMonacoEditor
        value={code}
        onChange={handleChange}
        onValidation={handleValidation}
        parserEndpoint="http://localhost:5000/api/parse"
        height="calc(100vh - 50px)"
      />
    </div>
  );
}
```

### Multiple Editors (Tabs)

```tsx
import React, { useState } from 'react';
import { NeuralDSLMonacoEditor } from './components/editor';

interface Tab {
  id: string;
  title: string;
  content: string;
}

function TabbedEditor() {
  const [tabs, setTabs] = useState<Tab[]>([
    { id: '1', title: 'Model 1', content: 'network Model1 {...}' },
    { id: '2', title: 'Model 2', content: 'network Model2 {...}' }
  ]);
  const [activeTab, setActiveTab] = useState('1');

  const updateTabContent = (id: string, content: string) => {
    setTabs(tabs.map(tab => 
      tab.id === id ? { ...tab, content } : tab
    ));
  };

  const activeContent = tabs.find(t => t.id === activeTab)?.content || '';

  return (
    <div>
      <div style={{ display: 'flex', borderBottom: '1px solid #ccc' }}>
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              padding: '10px 20px',
              border: 'none',
              background: activeTab === tab.id ? '#fff' : '#f0f0f0',
              borderBottom: activeTab === tab.id ? '2px solid blue' : 'none'
            }}
          >
            {tab.title}
          </button>
        ))}
      </div>
      <NeuralDSLMonacoEditor
        value={activeContent}
        onChange={(content) => updateTabContent(activeTab, content)}
        height="calc(100vh - 50px)"
      />
    </div>
  );
}
```

### With File Operations

```tsx
import React, { useState } from 'react';
import { NeuralDSLMonacoEditor } from './components/editor';

function EditorWithFileOps() {
  const [code, setCode] = useState('');
  const [filename, setFilename] = useState('untitled.neural');

  const handleSave = () => {
    const blob = new Blob([code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
  };

  const handleLoad = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const content = e.target?.result as string;
        setCode(content);
        setFilename(file.name);
      };
      reader.readAsText(file);
    }
  };

  return (
    <div>
      <div style={{ padding: '10px', background: '#f0f0f0' }}>
        <input
          type="file"
          accept=".neural"
          onChange={handleLoad}
          style={{ marginRight: '10px' }}
        />
        <button onClick={handleSave}>Save</button>
        <span style={{ marginLeft: '10px' }}>{filename}</span>
      </div>
      <NeuralDSLMonacoEditor
        value={code}
        onChange={setCode}
        height="calc(100vh - 50px)"
      />
    </div>
  );
}
```

## Backend Parser Integration

### Flask Backend Example

```python
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
sys.path.append('../../../')
from neural.parser.parser import NeuralParser

app = Flask(__name__)
CORS(app)

parser = NeuralParser()

@app.route('/api/parse', methods=['POST'])
def parse_code():
    data = request.json
    code = data.get('code', '')
    
    try:
        result = parser.parse(code)
        return jsonify({
            'success': True,
            'errors': [],
            'warnings': []
        })
    except Exception as e:
        error_info = parse_error(str(e))
        return jsonify({
            'success': False,
            'errors': [error_info],
            'warnings': []
        })

def parse_error(error_msg):
    import re
    match = re.search(r'line (\d+), column (\d+)', error_msg)
    if match:
        return {
            'line': int(match.group(1)),
            'column': int(match.group(2)),
            'message': error_msg,
            'severity': 'error'
        }
    return {
        'line': 1,
        'column': 1,
        'message': error_msg,
        'severity': 'error'
    }

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

### Express Backend Example

```javascript
const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');

const app = express();
app.use(cors());
app.use(express.json());

app.post('/api/parse', (req, res) => {
  const { code } = req.body;
  
  // Call Python parser
  const python = spawn('python', ['-c', `
import sys
sys.path.append('../../../../')
from neural.parser.parser import NeuralParser

parser = NeuralParser()
code = """${code.replace(/"/g, '\\"')}"""

try:
    result = parser.parse(code)
    print('{"success": true, "errors": [], "warnings": []}')
except Exception as e:
    print('{"success": false, "errors": [{"line": 1, "column": 1, "message": "' + str(e) + '", "severity": "error"}], "warnings": []}')
`]);

  let output = '';
  python.stdout.on('data', (data) => {
    output += data.toString();
  });

  python.on('close', () => {
    try {
      const result = JSON.parse(output);
      res.json(result);
    } catch (e) {
      res.json({
        success: false,
        errors: [{
          line: 1,
          column: 1,
          message: 'Parser error',
          severity: 'error'
        }]
      });
    }
  });
});

app.listen(5000, () => {
  console.log('Parser server running on port 5000');
});
```

## Styling

### Custom Theme

```tsx
import { NeuralDSLTheme } from './components/editor/theme';

// Modify existing theme
NeuralDSLTheme.dark.colors['editor.background'] = '#000000';

// Or create entirely new theme
const customTheme = {
  base: 'vs-dark' as const,
  inherit: true,
  rules: [
    { token: 'keyword', foreground: 'FF0000' },
    // ... more rules
  ],
  colors: {
    'editor.background': '#1a1a1a',
    // ... more colors
  }
};
```

### CSS Customization

```css
/* Override editor container */
.neural-dsl-editor-container {
  border: 2px solid #007acc;
  border-radius: 8px;
  overflow: hidden;
}

/* Custom scrollbar */
.monaco-editor .monaco-scrollable-element > .scrollbar {
  background: rgba(0, 0, 0, 0.4);
}

/* Custom error styling */
.neural-dsl-error-inline {
  background-color: rgba(255, 0, 0, 0.5) !important;
}
```

## Performance Optimization

### Lazy Loading Monaco

```tsx
import React, { lazy, Suspense } from 'react';

const LazyEditor = lazy(() => 
  import('./components/editor').then(module => ({
    default: module.NeuralDSLMonacoEditor
  }))
);

function App() {
  return (
    <Suspense fallback={<div>Loading editor...</div>}>
      <LazyEditor value="" onChange={() => {}} />
    </Suspense>
  );
}
```

### Debounced Validation

The editor already includes debounced validation (500ms), but you can adjust it:

```typescript
// In diagnosticsProvider.ts
this.validationTimeout = setTimeout(async () => {
  // validation logic
}, 1000); // Change to 1 second
```

## Troubleshooting

### Monaco fails to load

**Problem**: `Cannot find module 'monaco-editor'`

**Solution**: Ensure webpack or vite plugin is properly configured. Check that `monaco-editor` is in `dependencies`, not `devDependencies`.

### Syntax highlighting doesn't work

**Problem**: Text appears without colors

**Solution**: Verify that the language registration happens before editor creation. Check browser console for errors.

### Parser endpoint errors

**Problem**: Validation always fails

**Solution**: 
1. Verify backend is running: `curl http://localhost:5000/api/parse`
2. Check CORS is enabled on backend
3. Verify endpoint URL is correct in component props

### Performance issues with large files

**Problem**: Editor lags with files over 1000 lines

**Solution**:
1. Disable minimap: `minimap: { enabled: false }`
2. Reduce validation frequency
3. Consider lazy loading sections of large files

## Examples

See the `examples/` directory for complete working examples:
- `BasicUsage.tsx` - Simple editor integration
- `WithParserBackend.tsx` - Editor with validation panel
- `ComparisonView.tsx` - Side-by-side diff view

## API Reference

See `README.md` for complete API documentation.
