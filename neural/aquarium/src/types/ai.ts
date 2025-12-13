export type LanguageCode = 
  | 'en' 
  | 'fr' 
  | 'es' 
  | 'de' 
  | 'it' 
  | 'pt' 
  | 'ru' 
  | 'zh-cn' 
  | 'ja' 
  | 'ko' 
  | 'ar' 
  | 'hi';

export const SUPPORTED_LANGUAGES: Record<LanguageCode, string> = {
  'en': 'English',
  'fr': 'French',
  'es': 'Spanish',
  'de': 'German',
  'it': 'Italian',
  'pt': 'Portuguese',
  'ru': 'Russian',
  'zh-cn': 'Chinese (Simplified)',
  'ja': 'Japanese',
  'ko': 'Korean',
  'ar': 'Arabic',
  'hi': 'Hindi',
};

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  metadata?: {
    intent?: string;
    language?: string;
    dsl_code?: string;
  };
}

export interface AIResponse {
  response: string;
  dsl_code?: string;
  intent: string;
  success: boolean;
  language: string;
}

export interface ConversationContext {
  current_model?: string;
  conversation_history?: ChatMessage[];
}
