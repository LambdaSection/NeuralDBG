import React, { useState, useEffect, useRef } from 'react';
import { AIService } from '../../services/AIService';
import { ChatMessage, AIResponse, LanguageCode } from '../../types/ai';
import ChatInterface from './ChatInterface';
import DSLCodeViewer from './DSLCodeViewer';
import LanguageSelector from './LanguageSelector';
import './AIAssistantSidebar.css';

interface AIAssistantSidebarProps {
  onDSLGenerated?: (dslCode: string) => void;
  onDSLApplied?: (dslCode: string) => void;
}

const AIAssistantSidebar: React.FC<AIAssistantSidebarProps> = ({
  onDSLGenerated,
  onDSLApplied,
}) => {
  const [isOpen, setIsOpen] = useState<boolean>(true);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [currentDSL, setCurrentDSL] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [selectedLanguage, setSelectedLanguage] = useState<LanguageCode>('en');
  const [isEditing, setIsEditing] = useState<boolean>(false);
  const [editedDSL, setEditedDSL] = useState<string>('');
  const aiService = useRef(new AIService());

  useEffect(() => {
    const welcomeMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'assistant',
      content: 'Hello! I can help you create neural networks using natural language. Try saying: "Create a CNN for image classification with 10 classes"',
      timestamp: new Date(),
    };
    setMessages([welcomeMessage]);
  }, []);

  const handleSendMessage = async (userInput: string) => {
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: userInput,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const context = {
        current_model: currentDSL || undefined,
        conversation_history: messages.slice(-5),
      };

      const response: AIResponse = await aiService.current.chat(
        userInput,
        context,
        selectedLanguage
      );

      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.response,
        timestamp: new Date(),
        metadata: {
          intent: response.intent,
          language: response.language,
        },
      };

      setMessages((prev) => [...prev, assistantMessage]);

      if (response.dsl_code) {
        setCurrentDSL(response.dsl_code);
        setEditedDSL(response.dsl_code);
        if (onDSLGenerated) {
          onDSLGenerated(response.dsl_code);
        }
      }
    } catch (error) {
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `Error: ${error instanceof Error ? error.message : 'Unknown error occurred'}`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleEditDSL = () => {
    setIsEditing(true);
  };

  const handleSaveDSL = () => {
    setCurrentDSL(editedDSL);
    setIsEditing(false);
    if (onDSLGenerated) {
      onDSLGenerated(editedDSL);
    }
  };

  const handleCancelEdit = () => {
    setEditedDSL(currentDSL);
    setIsEditing(false);
  };

  const handleApplyDSL = () => {
    if (onDSLApplied) {
      onDSLApplied(currentDSL);
    }
    const applyMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'assistant',
      content: 'DSL code has been applied to your model!',
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, applyMessage]);
  };

  const handleResetConversation = () => {
    setMessages([]);
    setCurrentDSL('');
    setEditedDSL('');
    aiService.current.reset();
    
    const welcomeMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'assistant',
      content: 'Conversation reset. How can I help you create a neural network?',
      timestamp: new Date(),
    };
    setMessages([welcomeMessage]);
  };

  const handleLanguageChange = (language: LanguageCode) => {
    setSelectedLanguage(language);
    const langMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'assistant',
      content: `Language changed to ${language.toUpperCase()}. You can now speak in your preferred language!`,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, langMessage]);
  };

  return (
    <div className={`ai-assistant-sidebar ${isOpen ? 'open' : 'closed'}`}>
      <div className="sidebar-header">
        <h2>AI Assistant</h2>
        <button
          className="toggle-button"
          onClick={() => setIsOpen(!isOpen)}
          aria-label={isOpen ? 'Close sidebar' : 'Open sidebar'}
        >
          {isOpen ? '→' : '←'}
        </button>
      </div>

      {isOpen && (
        <div className="sidebar-content">
          <LanguageSelector
            selectedLanguage={selectedLanguage}
            onLanguageChange={handleLanguageChange}
          />

          <div className="chat-section">
            <ChatInterface
              messages={messages}
              onSendMessage={handleSendMessage}
              isLoading={isLoading}
            />
          </div>

          {currentDSL && (
            <div className="dsl-section">
              <DSLCodeViewer
                code={isEditing ? editedDSL : currentDSL}
                isEditing={isEditing}
                onEdit={handleEditDSL}
                onSave={handleSaveDSL}
                onCancel={handleCancelEdit}
                onApply={handleApplyDSL}
                onChange={setEditedDSL}
              />
            </div>
          )}

          <div className="sidebar-actions">
            <button
              className="reset-button"
              onClick={handleResetConversation}
            >
              Reset Conversation
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default AIAssistantSidebar;
