import React from 'react';
import ReactMarkdown from 'react-markdown';
import { ChatMessage } from '../../types/ai';
import './MessageBubble.css';

interface MessageBubbleProps {
  message: ChatMessage;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => {
  const formatTime = (date: Date) => {
    return new Date(date).toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <div className={`message-bubble ${message.role}`}>
      <div className="message-content">
        <ReactMarkdown>{message.content}</ReactMarkdown>
      </div>
      <div className="message-metadata">
        <span className="message-time">{formatTime(message.timestamp)}</span>
        {message.metadata?.language && message.metadata.language !== 'en' && (
          <span className="message-language">
            {message.metadata.language.toUpperCase()}
          </span>
        )}
      </div>
    </div>
  );
};

export default MessageBubble;
