"""
Multi-Language Support for Neural AI

Provides translation and language detection for natural language processing.
"""

import logging
from typing import Optional, Dict
import os

logger = logging.getLogger(__name__)


class LanguageDetector:
    """Detects the language of input text."""
    
    def __init__(self):
        """Initialize language detector."""
        self._detector = None
        self._initialize_detector()
    
    def _initialize_detector(self):
        """Initialize language detection library."""
        try:
            from langdetect import detect, DetectorFactory
            DetectorFactory.seed = 0  # For consistent results
            self._detector = detect
        except ImportError:
            # Fallback to simple heuristic
            self._detector = None
    
    def detect(self, text: str) -> str:
        """
        Detect language of text.
        
        Args:
            text: Input text
            
        Returns:
            Language code (e.g., 'en', 'fr', 'es', 'zh-cn')
        """
        if self._detector:
            try:
                return self._detector(text)
            except:
                pass
        
        # Fallback: simple heuristic
        return self._heuristic_detect(text)
    
    def _heuristic_detect(self, text: str) -> str:
        """Simple heuristic-based language detection."""
        # Check for common non-English characters
        if any(ord(c) > 127 for c in text[:100]):
            # Likely non-English, but we can't determine which
            # Default to English and let translation handle it
            return 'en'
        return 'en'


class Translator:
    """Translates text between languages."""
    
    def __init__(self):
        """Initialize translator."""
        self._translator = None
        self._initialize_translator()
    
    def _initialize_translator(self):
        """Initialize translation library."""
        try:
            from googletrans import Translator
            self._translator = Translator()
        except ImportError:
            # Try alternative
            try:
                from deep_translator import GoogleTranslator
                self._translator = GoogleTranslator
            except ImportError:
                self._translator = None
    
    def translate(self, text: str, target_lang: str = 'en', source_lang: Optional[str] = None) -> str:
        """
        Translate text to target language.
        
        Args:
            text: Text to translate
            target_lang: Target language code (default: 'en')
            source_lang: Source language code (auto-detect if None)
            
        Returns:
            Translated text
        """
        if target_lang == 'en' and not source_lang:
            # If translating to English and source is unknown, try to detect
            detector = LanguageDetector()
            detected = detector.detect(text)
            if detected == 'en':
                return text  # Already English
        
        if not self._translator:
            # No translator available, return original
            return text
        
        try:
            if hasattr(self._translator, 'translate'):
                # googletrans
                result = self._translator.translate(text, dest=target_lang, src=source_lang)
                return result.text
            else:
                # deep_translator
                translator = self._translator(source=source_lang or 'auto', target=target_lang)
                return translator.translate(text)
        except Exception as e:
            # Translation failed, return original
            logger.warning("Translation warning: %s", e)
            return text


class MultiLanguageSupport:
    """
    Multi-language support for Neural AI.
    
    Handles language detection and translation for natural language processing.
    """
    
    SUPPORTED_LANGUAGES = {
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
        'hi': 'Hindi'
    }
    
    def __init__(self):
        """Initialize multi-language support."""
        self.detector = LanguageDetector()
        self.translator = Translator()
    
    def process(self, text: str, target_lang: str = 'en') -> Dict[str, str]:
        """
        Process text: detect language and translate if needed.
        
        Args:
            text: Input text
            target_lang: Target language for processing (default: 'en')
            
        Returns:
            Dictionary with:
            - original: Original text
            - detected_lang: Detected language code
            - translated: Translated text (if needed)
            - final: Final text to use (translated if needed, original if already target)
        """
        detected = self.detector.detect(text)
        
        result = {
            'original': text,
            'detected_lang': detected,
            'translated': text,
            'final': text
        }
        
        # Translate if needed
        if detected != target_lang:
            translated = self.translator.translate(text, target_lang=target_lang, source_lang=detected)
            result['translated'] = translated
            result['final'] = translated
        
        return result
    
    def is_supported(self, lang_code: str) -> bool:
        """Check if language is supported."""
        return lang_code in self.SUPPORTED_LANGUAGES
    
    def get_language_name(self, lang_code: str) -> str:
        """Get human-readable language name."""
        return self.SUPPORTED_LANGUAGES.get(lang_code, lang_code)

