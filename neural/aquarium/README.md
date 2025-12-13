# Neural Aquarium

Neural Aquarium is a comprehensive web-based development environment for Neural DSL, featuring both an AI-powered assistant sidebar and a settings & preferences panel.

## Features

### AI Assistant Sidebar

AI-powered web interface for building Neural DSL models through natural language conversation.

- **Natural Language Interface**: Describe neural networks in plain language
- **Real-time DSL Generation**: Instantly generate Neural DSL code from descriptions
- **Multi-Language Support**: Interact with the AI in 12+ languages
- **Code Review & Editing**: Review and manually edit generated DSL code
- **Conversational Refinement**: Iteratively improve models through chat
- **Apply to Workspace**: Apply generated models to your workspace

### Settings & Preferences Panel

Comprehensive IDE settings panel with persistent configuration storage.

#### 1. Editor Settings
- **Themes**: Choose between Light, Dark, or Custom themes
- **Custom Theme Editor**: Full color customization with color pickers for:
  - Background, Foreground, Selection
  - Comment, Keyword, String, Number, Function, Operator
  - Cursor color
- **Font Configuration**: Size (8-32px) and Family selection
- **Editor Behavior**:
  - Line numbers
  - Word wrap
  - Auto indent
  - Bracket matching
  - Highlight active line
- **Tab Settings**: Tab size (2-8) and spaces vs tabs

#### 2. Keybindings
Fully customizable keyboard shortcuts for:
- **File Operations**: Save, Save As, Open, New File, Close Tab
- **Search & Navigate**: Find, Replace, Go to Line, Command Palette
- **Editing**: Comment Line, Indent, Outdent, Duplicate Line, Delete Line, Move Line Up/Down
- **View**: Toggle Terminal, Toggle Sidebar
- **Model Operations**: Run Model, Debug Model, Compile Model

#### 3. Python Interpreter
- **Interpreter Selection**: Browse or specify Python executable path
- **Default Interpreter**: Set default interpreter command (python, python3, etc.)
- **Virtual Environments**: Configure venv or conda environments
- **Interpreter Detection**: Auto-detect installed Python interpreters
- **Interpreter Testing**: Test selected interpreter

#### 4. Backend Configuration
- **Default Backend**: Choose between TensorFlow, PyTorch, or ONNX
- **Auto-detection**: Automatically detect available backends
- **Backend Preferences**: Enable/disable specific backends
- **Backend Status**: Check which backends are installed

#### 5. Auto-save Settings
- **Enable/Disable**: Toggle auto-save functionality
- **Save Interval**: Configure interval (5-300 seconds)
- **Additional Options**:
  - Auto-save on focus lost
  - Auto-save on window change

#### 6. Extensions & Plugins
- **Extension Management**:
  - View installed extensions
  - Enable/disable extensions
  - Install from marketplace or file
  - Auto-update configuration
- **Plugin Configuration**:
  - Set marketplace URL
  - Auto-install dependencies
  - Configure individual plugins

#### 7. Advanced Settings
- **User Interface**:
  - Sidebar width (150-500px)
  - Panel height (100-500px)
  - Toggle Minimap, Breadcrumbs, Status Bar, Activity Bar
- **Terminal Configuration**:
  - Default shell selection
  - Font size (8-24px)
  - Cursor style (Block, Underline, Bar)
- **NeuralDbg Integration**:
  - Auto-launch settings
  - Port and host configuration
- **Maintenance**:
  - Clear cache
  - Reset all settings

## Architecture

### AI Assistant (React + TypeScript Frontend)

- `src/components/ai/` - AI assistant components
  - `AIAssistantSidebar.tsx` - Main sidebar container
  - `ChatInterface.tsx` - Chat UI with message history
  - `MessageBubble.tsx` - Individual message rendering
  - `DSLCodeViewer.tsx` - DSL code display and editing
  - `LanguageSelector.tsx` - Language selection dropdown

### Backend (Python + Flask)

- `backend/api.py` - REST API server
- Integrates with:
  - `neural/ai/ai_assistant.py` - AI assistant core logic
  - `neural/ai/natural_language_processor.py` - NLP and DSL generation
  - `neural/ai/multi_language.py` - Multi-language translation

### Settings Panel (Python)

```
neural/aquarium/
├── aquarium_ide.py              # Main IDE application
├── README.md                     # This file
└── src/
    └── components/
        └── settings/
            ├── __init__.py       # Package exports
            ├── config_manager.py # Configuration persistence
            ├── settings_panel.py # Settings UI panel
            ├── theme_manager.py  # Theme management
            ├── keybinding_manager.py  # Keybinding handling
            └── extension_manager.py   # Extension management
```

## Setup

### AI Assistant

#### Frontend

```bash
cd neural/aquarium
npm install
npm start
```

The React app will run on `http://localhost:3000`

#### Backend

```bash
cd neural/aquarium/backend
pip install -r requirements.txt
python api.py
```

The Flask API will run on `http://localhost:5000`

#### Full Stack Development

Terminal 1 (Backend):
```bash
cd neural/aquarium/backend
python api.py
```

Terminal 2 (Frontend):
```bash
cd neural/aquarium
npm start
```

### Settings Panel / IDE

```bash
python neural/aquarium/aquarium_ide.py
```

The IDE will start on `http://localhost:8052` by default.

## Configuration Storage

All settings are persistently stored in `~/.aquarium/config.json`.

### Configuration Structure

```json
{
  "editor": {
    "theme": "dark",
    "custom_theme": { ... },
    "font_size": 14,
    "font_family": "Consolas, Monaco, monospace",
    "line_numbers": true,
    "word_wrap": false,
    "tab_size": 4,
    "insert_spaces": true,
    "auto_indent": true,
    "bracket_matching": true,
    "highlight_active_line": true
  },
  "keybindings": {
    "save": "Ctrl+S",
    "open": "Ctrl+O",
    ...
  },
  "python": {
    "interpreter_path": "",
    "default_interpreter": "python",
    "virtual_env_path": "",
    "conda_env": "",
    "use_system_python": true
  },
  "backend": {
    "default": "tensorflow",
    "auto_detect": true,
    "preferences": {
      "tensorflow": true,
      "pytorch": true,
      "onnx": true
    }
  },
  "autosave": {
    "enabled": true,
    "interval": 30,
    "on_focus_lost": true,
    "on_window_change": true
  },
  "extensions": {
    "enabled": [],
    "disabled": [],
    "auto_update": true,
    "check_updates_on_startup": true
  },
  "plugins": {
    "installed": [],
    "marketplace_url": "https://neural-dsl.org/plugins",
    "auto_install_dependencies": true
  },
  "ui": {
    "sidebar_width": 250,
    "panel_height": 200,
    "show_minimap": true,
    "show_breadcrumbs": true,
    "show_status_bar": true,
    "show_activity_bar": true
  },
  "terminal": {
    "shell": "powershell",
    "font_size": 12,
    "cursor_style": "block"
  },
  "neuraldbg": {
    "auto_launch": false,
    "port": 8050,
    "host": "localhost"
  }
}
```

## Usage

### AI Assistant

1. **Start a Conversation**: Click on the AI Assistant sidebar (right side)
2. **Describe Your Model**: e.g., "Create a CNN for MNIST with 10 classes"
3. **Review Generated DSL**: The DSL code appears in the code viewer
4. **Refine Your Model**: Add layers or modify settings through chat
5. **Edit Code**: Click "Edit" to manually adjust the DSL
6. **Apply Model**: Click "Apply" to use the model in your workspace

#### Example Prompts

- "Create a CNN for image classification with 10 classes"
- "Add a dense layer with 128 units"
- "Add dropout with rate 0.5"
- "Set optimizer to Adam with learning rate 0.001"
- "Add a convolutional layer with 64 filters and 3x3 kernel"
- "Add max pooling with pool size 2"

#### Multi-Language Support

The assistant supports these languages:
- English (en)
- French (fr)
- Spanish (es)
- German (de)
- Italian (it)
- Portuguese (pt)
- Russian (ru)
- Chinese Simplified (zh-cn)
- Japanese (ja)
- Korean (ko)
- Arabic (ar)
- Hindi (hi)

Select your language from the dropdown at the top of the sidebar.

### Programmatic Usage

#### Running Aquarium IDE

```python
from neural.aquarium import AquariumIDE

# Create and run IDE
ide = AquariumIDE()
ide.run(debug=True, port=8052)
```

#### Using Configuration Manager

```python
from neural.aquarium.src.components.settings import ConfigManager

# Initialize config manager
config = ConfigManager()

# Get configuration values
theme = config.get('editor', 'theme')
keybindings = config.get('keybindings')

# Set configuration values
config.set('editor', 'theme', 'light')
config.update_section('editor', {
    'font_size': 16,
    'theme': 'dark'
})

# Reset to defaults
config.reset_to_defaults()

# Export/Import configuration
config.export_config('my_config.json')
config.import_config('my_config.json')
```

#### Using Theme Manager

```python
from neural.aquarium.src.components.settings import ThemeManager

# Get theme colors
theme_colors = ThemeManager.get_theme('dark')
custom_colors = ThemeManager.get_theme('custom', {
    'background': '#000000',
    'foreground': '#ffffff'
})

# Apply theme to editor
editor_style = ThemeManager.apply_theme_to_editor(theme_colors)

# Generate CSS
css = ThemeManager.generate_theme_css(theme_colors)
```

#### Using Keybinding Manager

```python
from neural.aquarium.src.components.settings import KeybindingManager

# Initialize manager
kb_manager = KeybindingManager()

# Register keybindings
kb_manager.register_keybinding('save', 'Ctrl+S')
kb_manager.register_handler('save', lambda: print('Saving...'))

# Parse shortcuts
parsed = kb_manager.parse_shortcut('Ctrl+Shift+S')
# Returns: {'ctrl': True, 'shift': True, 'alt': False, 'meta': False, 'key': 'S'}

# Validate shortcuts
is_valid = kb_manager.validate_shortcut('Ctrl+S')  # True
```

#### Using Extension Manager

```python
from neural.aquarium.src.components.settings import ExtensionManager
from pathlib import Path

# Initialize manager
ext_manager = ExtensionManager(Path.home() / '.aquarium')

# Install extension
ext_manager.install_extension('path/to/extension.json')

# Enable/disable extensions
ext_manager.enable_extension('extension-id')
ext_manager.disable_extension('extension-id')

# Get extensions
enabled = ext_manager.get_enabled_extensions()
all_exts = ext_manager.get_all_extensions()

# Uninstall extension
ext_manager.uninstall_extension('extension-id')
```

## API Endpoints

### POST /api/ai/chat
Send a message to the AI assistant.

**Request:**
```json
{
  "user_input": "Create a CNN for MNIST",
  "context": {
    "current_model": "...",
    "conversation_history": [...]
  },
  "language": "en"
}
```

**Response:**
```json
{
  "response": "I've created a CNN model...",
  "dsl_code": "network MyModel { ... }",
  "intent": "create_model",
  "success": true,
  "language": "en"
}
```

### GET /api/ai/current-model
Get the current model DSL.

### POST /api/ai/reset
Reset the AI assistant state.

### POST /api/ai/translate
Translate text between languages.

### POST /api/ai/detect-language
Detect the language of input text.

### GET /api/ai/supported-languages
Get list of supported languages.

## Technology Stack

### Frontend
- React 18
- TypeScript
- Axios (API client)
- React Markdown (message rendering)

### Backend
- Flask 2.3+
- Flask-CORS
- Neural DSL AI modules

### Settings Panel
- Python 3.8+
- Dash (web framework)
- Bootstrap components

## Key Classes

- **ConfigManager**: Manages persistent configuration storage in `~/.aquarium/config.json`
- **SettingsPanel**: Creates the settings UI with all tabs and controls
- **ThemeManager**: Handles editor themes (light, dark, custom)
- **KeybindingManager**: Manages keyboard shortcuts and their handlers
- **ExtensionManager**: Handles installation and management of extensions
- **Extension**: Represents an individual extension

## Integration with Neural DSL

The AI assistant integrates with existing Neural DSL modules:
- `neural/ai/ai_assistant.py` - Main assistant logic
- `neural/ai/natural_language_processor.py` - Intent extraction and DSL generation
- `neural/ai/multi_language.py` - Language detection and translation

The settings panel integrates with:
- **NeuralDbg**: Configure debugger port and auto-launch settings
- **Neural DSL Compiler**: Set default backend (TensorFlow/PyTorch/ONNX)
- **Python Interpreter**: Configure virtual environments and interpreters
- **Extension Marketplace**: Browse and install extensions

Generated DSL code is compatible with the Neural DSL compiler and can be used with:
- `neural compile model.neural --backend tensorflow`
- `neural run model.neural --data dataset.yaml`
- `neural visualize model.neural`

## Development

### Running Tests
```bash
npm test  # Frontend tests
pytest tests/  # Backend/Python tests
```

### Building for Production
```bash
npm run build
```

### Code Style
- Frontend: Follows React/TypeScript best practices
- Backend/Settings: Follows PEP 8, uses type hints

## Troubleshooting

### Backend not connecting
- Ensure Flask server is running on port 5000
- Check CORS settings if running on different ports

### Translation not working
- Install optional language dependencies:
  ```bash
  pip install langdetect googletrans==4.0.0rc1
  # or
  pip install deep-translator
  ```

### DSL code not generating
- Check browser console for errors
- Verify backend API is accessible
- Check backend logs for Python errors

## Future Enhancements

Planned features:
- Monaco Editor integration for advanced code editing
- Real-time syntax highlighting for Neural DSL
- Code completion and IntelliSense
- Git integration
- Collaborative editing
- Cloud synchronization of settings
- Custom extension development API

## License

Part of the Neural DSL project. See main LICENSE.md for details.
