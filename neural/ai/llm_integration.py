"""
LLM Integration Layer for Neural AI

Provides abstraction for different LLM providers (OpenAI, Anthropic, Open-Source).
Supports both API-based and local LLM execution.
"""

from typing import Optional, Dict, List, Any
import os
import json
from neural.exceptions import DependencyError, ConfigurationError


class LLMProvider:
    """Base class for LLM providers."""
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        raise NotImplementedError
    
    def is_available(self) -> bool:
        """Check if provider is available."""
        return False


class OpenAIProvider(LLMProvider):
    """OpenAI GPT-4/GPT-3.5 integration."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key (or from OPENAI_API_KEY env var)
            model: Model to use (gpt-4, gpt-3.5-turbo, etc.)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self._client = None
        
        if self.api_key:
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                pass
    
    def is_available(self) -> bool:
        """Check if OpenAI is available."""
        return self._client is not None
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI."""
        if not self.is_available():
            raise DependencyError(
                dependency='openai',
                feature='OpenAI LLM integration',
                install_hint='pip install openai and set OPENAI_API_KEY environment variable'
            )
        
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            **kwargs
        )
        
        return response.choices[0].message.content
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for Neural DSL generation."""
        return """You are an expert in neural networks and the Neural DSL language.
Your task is to convert natural language descriptions into valid Neural DSL code.

Neural DSL syntax:
- network ModelName { input: (height, width, channels) layers: ... }
- Layers: Conv2D(filters, kernel_size, activation), Dense(units, activation), etc.
- Always provide complete, valid Neural DSL code.

Respond only with the Neural DSL code, no explanations unless asked."""


class AnthropicProvider(LLMProvider):
    """Anthropic Claude integration."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-opus-20240229"):
        """
        Initialize Anthropic provider.
        
        Args:
            api_key: Anthropic API key (or from ANTHROPIC_API_KEY env var)
            model: Model to use
        """
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.model = model
        self._client = None
        
        if self.api_key:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                pass
    
    def is_available(self) -> bool:
        """Check if Anthropic is available."""
        return self._client is not None
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Anthropic Claude."""
        if not self.is_available():
            raise RuntimeError("Anthropic client not available. Install anthropic package and set API key.")
        
        response = self._client.messages.create(
            model=self.model,
            max_tokens=kwargs.get('max_tokens', 1024),
            messages=[
                {"role": "user", "content": f"{self._get_system_prompt()}\n\n{prompt}"}
            ]
        )
        
        return response.content[0].text
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for Neural DSL generation."""
        return """You are an expert in neural networks and the Neural DSL language.
Convert natural language descriptions into valid Neural DSL code.
Respond only with the Neural DSL code."""


class OllamaProvider(LLMProvider):
    """Ollama (local LLM) integration."""
    
    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama provider.
        
        Args:
            model: Ollama model name (llama2, mistral, etc.)
            base_url: Ollama server URL
        """
        self.model = model
        self.base_url = base_url
        self._available = False
        
        # Check if Ollama is available
        try:
            import requests
            response = requests.get(f"{base_url}/api/tags", timeout=2)
            self._available = response.status_code == 200
        except:
            self._available = False
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        return self._available
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Ollama."""
        if not self.is_available():
            raise RuntimeError("Ollama not available. Make sure Ollama is running.")
        
        import requests
        
        full_prompt = f"{self._get_system_prompt()}\n\nUser: {prompt}\nAssistant:"
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                **kwargs
            },
            timeout=30
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Ollama API error: {response.status_code}")
        
        return response.json().get('response', '')
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for Neural DSL generation."""
        return """You are an expert in neural networks and the Neural DSL language.
Convert natural language descriptions into valid Neural DSL code.
Respond only with the Neural DSL code."""


class LLMIntegration:
    """
    Unified LLM integration layer.
    
    Automatically selects available provider or allows manual selection.
    """
    
    def __init__(self, provider: Optional[str] = None):
        """
        Initialize LLM integration.
        
        Args:
            provider: Provider to use ('openai', 'anthropic', 'ollama', or None for auto)
        """
        self.providers = {
            'openai': OpenAIProvider(),
            'anthropic': AnthropicProvider(),
            'ollama': OllamaProvider()
        }
        
        if provider:
            self.current_provider = self.providers.get(provider)
        else:
            # Auto-select first available provider
            self.current_provider = self._auto_select_provider()
    
    def _auto_select_provider(self) -> Optional[LLMProvider]:
        """Auto-select first available provider."""
        # Prefer local (Ollama) for privacy, then API providers
        priority = ['ollama', 'openai', 'anthropic']
        
        for provider_name in priority:
            provider = self.providers[provider_name]
            if provider.is_available():
                return provider
        
        return None
    
    def is_available(self) -> bool:
        """Check if any LLM provider is available."""
        return self.current_provider is not None and self.current_provider.is_available()
    
    def generate_dsl(self, natural_language: str, context: Optional[Dict] = None) -> str:
        """
        Generate Neural DSL from natural language.
        
        Args:
            natural_language: Natural language description
            context: Optional context (current model state, etc.)
            
        Returns:
            Generated Neural DSL code
        """
        if not self.is_available():
            raise RuntimeError("No LLM provider available. Install and configure a provider.")
        
        prompt = self._build_prompt(natural_language, context)
        response = self.current_provider.generate(prompt)
        
        # Extract DSL code from response (might include explanations)
        dsl_code = self._extract_dsl_code(response)
        
        return dsl_code
    
    def _build_prompt(self, natural_language: str, context: Optional[Dict] = None) -> str:
        """Build prompt for LLM."""
        prompt = f"Convert this natural language description into Neural DSL code:\n\n"
        prompt += f"{natural_language}\n\n"
        
        if context:
            prompt += f"Context: Current model has {len(context.get('layers', []))} layers.\n"
        
        prompt += "\nProvide only the Neural DSL code, no explanations."
        
        return prompt
    
    def _extract_dsl_code(self, response: str) -> str:
        """Extract DSL code from LLM response."""
        # Look for code blocks
        import re
        
        # Try to find code in markdown code blocks
        code_block = re.search(r'```(?:neural|python)?\n?(.*?)```', response, re.DOTALL)
        if code_block:
            return code_block.group(1).strip()
        
        # Try to find network { ... } pattern
        network_match = re.search(r'network\s+\w+\s*\{.*?\}', response, re.DOTALL)
        if network_match:
            return network_match.group(0).strip()
        
        # Return as-is if no pattern found
        return response.strip()

