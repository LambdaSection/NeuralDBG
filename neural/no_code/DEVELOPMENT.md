# No-Code Interface Development Guide

This guide covers development of the Neural DSL No-Code Interface, including architecture, components, and contribution guidelines.

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (React)                     │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ React Flow  │  │ Layer Palette│  │   Properties  │  │
│  │   Canvas    │  │  & Search    │  │     Panel     │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
│           │               │                  │           │
│           └───────────────┴──────────────────┘           │
│                           │                              │
│                    Fetch API Calls                       │
└────────────────────────────┼────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Flask REST API │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
    ┌────▼────┐      ┌──────▼──────┐      ┌────▼────┐
    │ Shape   │      │    Code     │      │  Model  │
    │Validator│      │  Generator  │      │ Storage │
    └─────────┘      └─────────────┘      └─────────┘
```

### Components

#### Frontend (React)

**Location**: `neural/no_code/static/app.jsx`

**Main Components**:
- `App`: Root component managing state and routing
- `LayerPalette`: Categorized, searchable layer list
- `PropertiesPanel`: Layer parameter editor
- `ValidationPanel`: Real-time validation feedback
- `ExportModal`: Code export interface
- `TemplateModal`: Template selection gallery
- `Tutorial`: Interactive tutorial system
- `CustomNode`: React Flow node component

**State Management**:
- React hooks (useState, useEffect, useCallback)
- React Flow's useNodesState and useEdgesState
- Local state for UI interactions

#### Backend (Flask)

**Location**: `neural/no_code/app.py`

**Endpoints**:
- `GET /` - Serve main HTML page
- `GET /api/layers` - Get layer categories
- `GET /api/templates` - Get model templates
- `GET /api/tutorial` - Get tutorial steps
- `POST /api/validate` - Validate model
- `POST /api/generate-code` - Generate code
- `POST /api/save` - Save model
- `GET /api/load/<name>` - Load model
- `GET /api/models` - List saved models

#### Configuration

**Location**: `neural/no_code/config.py`

**Config Classes**:
- `Config`: Base configuration
- `DevelopmentConfig`: Dev environment settings
- `ProductionConfig`: Production settings
- `TestingConfig`: Test environment settings

## Adding New Features

### Adding a New Layer Type

1. **Update Backend** (`app.py`):

```python
LAYER_CATEGORIES = {
    "YourCategory": [
        {
            "name": "YourLayer",
            "params": {
                "param1": default_value,
                "param2": default_value
            }
        }
    ]
}
```

2. **Ensure Shape Propagation** (`neural/shape_propagation/shape_propagator.py`):

```python
def propagate_your_layer(self, input_shape, layer_params):
    # Calculate output shape
    return output_shape
```

3. **Add Code Generation** (if needed):

Update `neural/code_generation/` for backend-specific code.

### Adding a New Template

Update `app.py`:

```python
MODEL_TEMPLATES = {
    "your_template": {
        "name": "Your Template Name",
        "description": "Brief description",
        "input_shape": [None, height, width, channels],
        "layers": [
            {
                "type": "LayerType",
                "params": {...}
            },
            # More layers...
        ]
    }
}
```

### Adding Tutorial Steps

Update `app.py`:

```python
TUTORIAL_STEPS = [
    {
        "id": "step_id",
        "title": "Step Title",
        "content": "Step description and instructions",
        "target": "css-selector-or-id",
    },
    # More steps...
]
```

### Adding UI Features

1. **Edit `static/app.jsx`** for React components
2. **Edit `static/styles.css`** for styling
3. **Add API endpoints** in `app.py` if needed
4. **Update state management** in component

Example - Adding a new panel:

```jsx
function YourNewPanel({ data }) {
    const [state, setState] = useState(initialState);
    
    useEffect(() => {
        // Setup
    }, [dependency]);
    
    return (
        <div className="your-panel">
            {/* Your UI */}
        </div>
    );
}
```

## Development Workflow

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/your-org/neural-dsl.git
cd neural-dsl

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dashboard]"
pip install -r requirements-dev.txt

# Run in development mode
export FLASK_ENV=development  # On Windows: set FLASK_ENV=development
python neural/no_code/app.py
```

### Making Changes

1. **Create a feature branch**:
```bash
git checkout -b feature/your-feature-name
```

2. **Make changes** to code

3. **Test locally**:
```bash
# Run tests
pytest tests/test_no_code_interface.py -v

# Manual testing
python neural/no_code/app.py
# Open http://localhost:8051
```

4. **Commit changes**:
```bash
git add .
git commit -m "feat: add your feature description"
```

5. **Push and create PR**:
```bash
git push origin feature/your-feature-name
```

### Testing

**Unit Tests**:
```bash
pytest tests/test_no_code_interface.py -v
```

**API Tests**:
```bash
# Start server
python neural/no_code/app.py

# Run API tests
pytest tests/test_no_code_interface.py::test_validate_simple_model -v
```

**Integration Tests**:
```bash
python neural/no_code/examples/example_usage.py
```

**Manual Testing Checklist**:
- [ ] Drag and drop layers
- [ ] Connect layers
- [ ] Edit properties
- [ ] Validate model
- [ ] Load templates
- [ ] Export code
- [ ] Save/load models
- [ ] Run tutorial

## Code Style

### Python (Backend)

Follow PEP 8:
```bash
# Check style
ruff check neural/no_code/

# Format code
ruff check --fix neural/no_code/
```

### JavaScript (Frontend)

- Use functional components
- Use hooks for state management
- Destructure props
- Use arrow functions
- Keep components small and focused

Example:
```jsx
const MyComponent = ({ prop1, prop2 }) => {
    const [state, setState] = useState(initialValue);
    
    useEffect(() => {
        // Side effects
    }, [dependencies]);
    
    const handleEvent = useCallback(() => {
        // Event handler
    }, [dependencies]);
    
    return (
        <div>
            {/* JSX */}
        </div>
    );
};
```

### CSS

- Use class-based styling
- Follow BEM naming convention when appropriate
- Keep selectors specific
- Use CSS variables for colors/spacing
- Mobile-first responsive design

## Debugging

### Backend Debugging

1. **Enable debug mode**:
```python
# In config.py or environment
DEBUG = True
```

2. **Add breakpoints**:
```python
import pdb; pdb.set_trace()
```

3. **Check logs**:
```bash
# Flask logs to console
python neural/no_code/app.py
```

### Frontend Debugging

1. **Browser DevTools** (F12):
- Console for errors
- Network tab for API calls
- React DevTools extension

2. **Add console logs**:
```jsx
console.log('State:', state);
console.log('Props:', props);
```

3. **React DevTools**:
- Inspect component tree
- View props and state
- Profile performance

## Performance Optimization

### Backend

- **Caching**: Cache layer metadata and templates
- **Async Processing**: Use async for long-running operations
- **Database**: Consider SQLite for models (if needed)
- **Rate Limiting**: Add rate limiting for API endpoints

### Frontend

- **Memoization**: Use `useMemo` and `useCallback`
- **Lazy Loading**: Load templates on demand
- **Debouncing**: Debounce search and validation
- **Code Splitting**: Split large components

Example:
```jsx
const expensiveValue = useMemo(() => {
    return computeExpensiveValue(a, b);
}, [a, b]);

const handleChange = useCallback(
    debounce((value) => {
        // Handle change
    }, 300),
    []
);
```

## Security

### API Security

- **Input Validation**: Validate all user inputs
- **CORS**: Configure CORS properly
- **Rate Limiting**: Prevent abuse
- **File Size Limits**: Limit upload sizes

### Frontend Security

- **XSS Prevention**: Sanitize user inputs
- **CSRF Protection**: Use CSRF tokens
- **Content Security Policy**: Configure CSP headers

## Deployment

### Development

```bash
python neural/no_code/launcher.py --ui react --debug
```

### Production

```bash
# Set environment
export FLASK_ENV=production
export CORS_ORIGINS=https://yourdomain.com

# Use production server
gunicorn -w 4 -b 0.0.0.0:8051 neural.no_code.app:app
```

### Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . /app

RUN pip install -e ".[dashboard]"

EXPOSE 8051
CMD ["python", "neural/no_code/app.py"]
```

## Contributing

### Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Update documentation
6. Submit PR with description

### Code Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests pass
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
- [ ] Performance considered
- [ ] Security reviewed

## Troubleshooting

### Common Issues

**Issue**: Flask app won't start
- Check port 8051 is available
- Verify dependencies installed
- Check Python version >= 3.8

**Issue**: React components not updating
- Clear browser cache
- Check console for errors
- Verify state management

**Issue**: Validation failing
- Check shape propagator implementation
- Verify layer parameters
- Review error messages

**Issue**: Code generation errors
- Ensure backend generators available
- Check layer compatibility
- Review model structure

## Resources

- [React Flow Documentation](https://reactflow.dev/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Neural DSL Documentation](../../README.md)
- [Shape Propagation Guide](../shape_propagation/README.md)
- [Code Generation Guide](../code_generation/README.md)

## Contact

For questions or issues:
- GitHub Issues: [Create an issue](https://github.com/your-org/neural-dsl/issues)
- Email: maintainer@example.com
- Discord: Join our community

## License

MIT License - See LICENSE.md for details
