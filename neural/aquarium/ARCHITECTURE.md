# Neural Aquarium Architecture

## Overview

Neural Aquarium is a full-stack web application that provides an AI-powered interface for building Neural DSL models through natural language conversation.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React)                         │
│  ┌────────────────────────────────────────────────────────┐ │
│  │         AIAssistantSidebar Component                   │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐│ │
│  │  │   Language   │  │     Chat     │  │  DSL Code    ││ │
│  │  │   Selector   │  │  Interface   │  │   Viewer     ││ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘│ │
│  └────────────────────────────────────────────────────────┘ │
│                            │                                 │
│                            │ HTTP/REST                       │
│                            ▼                                 │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              AIService (API Client)                    │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                             │
                             │ axios
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                  Backend (Flask API)                         │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                  REST API Endpoints                    │ │
│  │  /api/ai/chat                                          │ │
│  │  /api/ai/current-model                                 │ │
│  │  /api/ai/reset                                         │ │
│  │  /api/ai/translate                                     │ │
│  │  /api/ai/detect-language                              │ │
│  └────────────────────────────────────────────────────────┘ │
│                            │                                 │
│                            ▼                                 │
│  ┌────────────────────────────────────────────────────────┐ │
│  │            Neural AI Integration Layer                 │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐│ │
│  │  │      AI      │  │     NLP      │  │Multi-Language││ │
│  │  │  Assistant   │  │  Processor   │  │   Support    ││ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘│ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Component Breakdown

### Frontend Components

#### AIAssistantSidebar
- **Purpose**: Main container for the AI assistant
- **State Management**: Manages conversation state, DSL code, language selection
- **Integration**: Communicates with backend via AIService

#### ChatInterface
- **Purpose**: Manages chat UI and message input
- **Features**: 
  - Message history display
  - Input form with suggestions
  - Loading indicators
  - Keyboard shortcuts (Enter to send)

#### MessageBubble
- **Purpose**: Renders individual chat messages
- **Features**:
  - Markdown rendering for formatted text
  - Timestamp display
  - Language indicator
  - User/Assistant role styling

#### DSLCodeViewer
- **Purpose**: Displays and edits generated DSL code
- **Features**:
  - Syntax highlighting
  - Edit mode
  - Copy to clipboard
  - Download as file
  - Apply to workspace

#### LanguageSelector
- **Purpose**: Language selection dropdown
- **Supported Languages**: 12+ languages including EN, FR, ES, DE, IT, PT, RU, ZH-CN, JA, KO, AR, HI

### Backend Components

#### Flask API (api.py)
- **Purpose**: REST API server
- **Endpoints**: 8 endpoints for chat, translation, language detection, etc.
- **Integration**: Bridges frontend and Neural AI modules

#### Neural AI Modules

##### NeuralAIAssistant (neural/ai/ai_assistant.py)
- Main AI assistant logic
- Orchestrates NLP and DSL generation
- Supports LLM integration (OpenAI, Anthropic, Ollama)
- Fallback to rule-based processing

##### NaturalLanguageProcessor (neural/ai/natural_language_processor.py)
- Intent extraction (create_model, add_layer, etc.)
- Parameter extraction (layer sizes, activations, etc.)
- Pattern matching for layer types, optimizers, losses

##### DSLGenerator (neural/ai/natural_language_processor.py)
- Generates Neural DSL code from intents
- Maintains model state
- Produces valid DSL syntax

##### MultiLanguageSupport (neural/ai/multi_language.py)
- Language detection
- Translation (Google Translate, DeepTranslator)
- 12+ language support

## Data Flow

### Chat Message Flow

```
1. User types message in ChatInterface
   ↓
2. ChatInterface calls onSendMessage handler
   ↓
3. AIAssistantSidebar.handleSendMessage called
   ↓
4. AIService.chat() sends POST to /api/ai/chat
   ↓
5. Backend processes with MultiLanguageSupport (if needed)
   ↓
6. NeuralAIAssistant.chat() processes request
   ↓
7. NaturalLanguageProcessor extracts intent
   ↓
8. DSLGenerator generates DSL code
   ↓
9. Response sent back to frontend
   ↓
10. AIAssistantSidebar updates messages and DSL code
    ↓
11. MessageBubble renders response
    ↓
12. DSLCodeViewer displays generated DSL
```

### DSL Code Application Flow

```
1. User clicks "Apply" in DSLCodeViewer
   ↓
2. onApply handler called
   ↓
3. AIAssistantSidebar.handleApplyDSL called
   ↓
4. onDSLApplied callback propagates to App
   ↓
5. App updates appliedDSL state
   ↓
6. Model workspace displays applied DSL
```

## Technology Stack

### Frontend
- **Framework**: React 18 with TypeScript
- **HTTP Client**: Axios
- **Markdown**: React Markdown
- **Styling**: CSS with CSS-in-JS patterns
- **Build Tool**: Create React App

### Backend
- **Framework**: Flask 2.3+
- **CORS**: Flask-CORS
- **Python Version**: 3.8+
- **Integration**: Neural DSL AI modules

## State Management

### Frontend State
- **AIAssistantSidebar**:
  - `messages`: ChatMessage[]
  - `currentDSL`: string
  - `isLoading`: boolean
  - `selectedLanguage`: LanguageCode
  - `isEditing`: boolean
  - `editedDSL`: string

- **App**:
  - `currentDSL`: string (from AI)
  - `appliedDSL`: string (applied to workspace)

### Backend State
- **ai_assistant**: NeuralAIAssistant instance (singleton)
- **multi_lang_support**: MultiLanguageSupport instance
- Each assistant maintains:
  - Current model state
  - DSL generator state
  - Conversation context

## API Communication

### Request Format
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

### Response Format
```json
{
  "response": "I've created a CNN model...",
  "dsl_code": "network MyModel { ... }",
  "intent": "create_model",
  "success": true,
  "language": "en"
}
```

## Error Handling

### Frontend
- Try-catch blocks in async operations
- Error messages displayed in chat
- Loading states prevent duplicate requests
- Graceful degradation for API failures

### Backend
- Exception handling in all endpoints
- Proper HTTP status codes
- Detailed error logging
- Fallback responses for failures

## Security Considerations

- CORS configured for local development
- Input validation on all endpoints
- No authentication required (local use)
- For production: Add JWT auth, rate limiting, input sanitization

## Performance Optimizations

- Message history limited to last 5 for context
- Debounced API calls (prevented by loading state)
- Lazy loading of components
- Memoization in React components
- Connection pooling in Flask

## Future Enhancements

1. **WebSocket Support**: Real-time streaming responses
2. **Syntax Highlighting**: Better DSL code display
3. **Model Visualization**: Visual representation of network
4. **History Management**: Save/load conversations
5. **Model Templates**: Pre-built model templates
6. **Collaborative Editing**: Multi-user support
7. **Advanced LLM Integration**: GPT-4, Claude, Gemini
8. **Voice Input**: Speech-to-text for hands-free operation
