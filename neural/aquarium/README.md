# Neural Aquarium - AI Assistant Sidebar

AI-powered web interface for building Neural DSL models through natural language conversation.

## Features

- **Natural Language Interface**: Describe neural networks in plain language
- **Real-time DSL Generation**: Instantly generate Neural DSL code from descriptions
- **Multi-Language Support**: Interact with the AI in 12+ languages
- **Code Review & Editing**: Review and manually edit generated DSL code
- **Conversational Refinement**: Iteratively improve models through chat
- **Apply to Workspace**: Apply generated models to your workspace

## Architecture

### Frontend (React + TypeScript)
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

## Setup

### Frontend

```bash
cd neural/aquarium
npm install
npm start
```

The React app will run on `http://localhost:3000`

### Backend

```bash
cd neural/aquarium/backend
pip install -r requirements.txt
python api.py
```

The Flask API will run on `http://localhost:5000`

### Full Stack Development

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

## Usage

1. **Start a Conversation**: Click on the AI Assistant sidebar (right side)
2. **Describe Your Model**: e.g., "Create a CNN for MNIST with 10 classes"
3. **Review Generated DSL**: The DSL code appears in the code viewer
4. **Refine Your Model**: Add layers or modify settings through chat
5. **Edit Code**: Click "Edit" to manually adjust the DSL
6. **Apply Model**: Click "Apply" to use the model in your workspace

## Example Prompts

- "Create a CNN for image classification with 10 classes"
- "Add a dense layer with 128 units"
- "Add dropout with rate 0.5"
- "Set optimizer to Adam with learning rate 0.001"
- "Add a convolutional layer with 64 filters and 3x3 kernel"
- "Add max pooling with pool size 2"

## Multi-Language Support

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

## Development

### Running Tests
```bash
npm test
```

### Building for Production
```bash
npm run build
```

### Code Style
- Frontend: Follows React/TypeScript best practices
- Backend: Follows PEP 8, uses type hints

## Integration with Neural DSL

The AI assistant integrates with existing Neural DSL modules:
- `neural/ai/ai_assistant.py` - Main assistant logic
- `neural/ai/natural_language_processor.py` - Intent extraction and DSL generation
- `neural/ai/multi_language.py` - Language detection and translation

Generated DSL code is compatible with the Neural DSL compiler and can be used with:
- `neural compile model.neural --backend tensorflow`
- `neural run model.neural --data dataset.yaml`
- `neural visualize model.neural`

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

## License

Part of the Neural DSL project. See main LICENSE.md for details.
