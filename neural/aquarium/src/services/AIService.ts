import axios from 'axios';
import { AIResponse, ConversationContext, LanguageCode } from '../types/ai';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

export class AIService {
  private apiUrl: string;

  constructor(apiUrl: string = API_BASE_URL) {
    this.apiUrl = apiUrl;
  }

  async chat(
    userInput: string,
    context?: ConversationContext,
    language: LanguageCode = 'en'
  ): Promise<AIResponse> {
    try {
      const response = await axios.post(`${this.apiUrl}/api/ai/chat`, {
        user_input: userInput,
        context: context || {},
        language: language,
      });

      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw new Error(error.response.data.error || 'AI service error');
      }
      throw new Error('Failed to communicate with AI service');
    }
  }

  async getCurrentModel(): Promise<string> {
    try {
      const response = await axios.get(`${this.apiUrl}/api/ai/current-model`);
      return response.data.model;
    } catch (error) {
      throw new Error('Failed to retrieve current model');
    }
  }

  async reset(): Promise<void> {
    try {
      await axios.post(`${this.apiUrl}/api/ai/reset`);
    } catch (error) {
      console.error('Failed to reset AI assistant:', error);
    }
  }

  async translateText(
    text: string,
    targetLang: LanguageCode,
    sourceLang?: LanguageCode
  ): Promise<string> {
    try {
      const response = await axios.post(`${this.apiUrl}/api/ai/translate`, {
        text,
        target_lang: targetLang,
        source_lang: sourceLang,
      });
      return response.data.translated_text;
    } catch (error) {
      console.error('Translation failed:', error);
      return text;
    }
  }

  async detectLanguage(text: string): Promise<LanguageCode> {
    try {
      const response = await axios.post(`${this.apiUrl}/api/ai/detect-language`, {
        text,
      });
      return response.data.language as LanguageCode;
    } catch (error) {
      console.error('Language detection failed:', error);
      return 'en';
    }
  }
}
