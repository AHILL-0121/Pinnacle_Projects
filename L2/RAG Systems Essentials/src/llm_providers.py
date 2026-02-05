"""
LLM Provider Module

Supports multiple LLM backends:
- Gemini (Google)
- Groq (Fast inference)
- Ollama (Local)

NO OpenAI API usage per requirements.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Generator
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM generation."""
    text: str
    provider: str
    model: str
    usage: Optional[Dict[str, int]] = None
    raw_response: Optional[Any] = None


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.1
    ) -> LLMResponse:
        """Generate text completion."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API provider."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-1.5-flash"
    ):
        self.api_key = api_key
        self.model = model
        self._client = None
        
        if api_key:
            self._initialize()
    
    def _initialize(self):
        """Initialize Gemini client."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(self.model)
            logger.info(f"Initialized Gemini with model: {self.model}")
        except ImportError:
            logger.warning("google-generativeai not installed. Install with: pip install google-generativeai")
            self._client = None
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            self._client = None
    
    @property
    def name(self) -> str:
        return "gemini"
    
    def is_available(self) -> bool:
        return self._client is not None
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.1
    ) -> LLMResponse:
        if not self.is_available():
            raise RuntimeError("Gemini client not initialized")
        
        # Combine system prompt and user prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        try:
            response = self._client.generate_content(
                full_prompt,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": temperature,
                }
            )
            
            return LLMResponse(
                text=response.text,
                provider=self.name,
                model=self.model,
                raw_response=response
            )
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise


class GroqProvider(BaseLLMProvider):
    """Groq API provider for fast inference."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.1-8b-instant"
    ):
        self.api_key = api_key
        self.model = model
        self._client = None
        
        if api_key:
            self._initialize()
    
    def _initialize(self):
        """Initialize Groq client."""
        try:
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
            logger.info(f"Initialized Groq with model: {self.model}")
        except ImportError:
            logger.warning("groq not installed. Install with: pip install groq")
            self._client = None
        except Exception as e:
            logger.error(f"Failed to initialize Groq: {e}")
            self._client = None
    
    @property
    def name(self) -> str:
        return "groq"
    
    def is_available(self) -> bool:
        return self._client is not None
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.1
    ) -> LLMResponse:
        if not self.is_available():
            raise RuntimeError("Groq client not initialized")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return LLMResponse(
                text=response.choices[0].message.content,
                provider=self.name,
                model=self.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                raw_response=response
            )
        except Exception as e:
            logger.error(f"Groq generation failed: {e}")
            raise


class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "mistral"
    ):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self._available = None
    
    @property
    def name(self) -> str:
        return "ollama"
    
    def is_available(self) -> bool:
        """Check if Ollama is running."""
        if self._available is not None:
            return self._available
        
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            self._available = response.status_code == 200
            if self._available:
                logger.info(f"Ollama available at {self.base_url}")
            return self._available
        except Exception:
            self._available = False
            logger.warning(f"Ollama not available at {self.base_url}")
            return False
    
    def list_models(self) -> list:
        """List available Ollama models."""
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        return []
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.1
    ) -> LLMResponse:
        if not self.is_available():
            raise RuntimeError(f"Ollama not available at {self.base_url}")
        
        import requests
        
        # Build request
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120  # Longer timeout for local inference
            )
            response.raise_for_status()
            data = response.json()
            
            return LLMResponse(
                text=data.get("response", ""),
                provider=self.name,
                model=self.model,
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                },
                raw_response=data
            )
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise


class LLMManager:
    """
    Manager for multiple LLM providers.
    
    Provides a unified interface for switching between providers
    and handles fallback logic.
    """
    
    PROVIDERS = {
        "gemini": GeminiProvider,
        "groq": GroqProvider,
        "ollama": OllamaProvider
    }
    
    def __init__(self):
        self._providers: Dict[str, BaseLLMProvider] = {}
        self._active_provider: Optional[str] = None
    
    def register_provider(
        self,
        name: str,
        provider: BaseLLMProvider
    ):
        """Register a provider instance."""
        self._providers[name] = provider
        if self._active_provider is None and provider.is_available():
            self._active_provider = name
    
    def setup_gemini(self, api_key: str, model: str = "gemini-1.5-flash"):
        """Setup Gemini provider."""
        provider = GeminiProvider(api_key=api_key, model=model)
        self.register_provider("gemini", provider)
        return provider.is_available()
    
    def setup_groq(self, api_key: str, model: str = "llama-3.1-8b-instant"):
        """Setup Groq provider."""
        provider = GroqProvider(api_key=api_key, model=model)
        self.register_provider("groq", provider)
        return provider.is_available()
    
    def setup_ollama(self, base_url: str = "http://localhost:11434", model: str = "mistral"):
        """Setup Ollama provider."""
        provider = OllamaProvider(base_url=base_url, model=model)
        self.register_provider("ollama", provider)
        return provider.is_available()
    
    def set_active_provider(self, name: str):
        """Set the active provider."""
        if name not in self._providers:
            raise ValueError(f"Provider '{name}' not registered")
        if not self._providers[name].is_available():
            raise RuntimeError(f"Provider '{name}' is not available")
        self._active_provider = name
        logger.info(f"Active LLM provider: {name}")
    
    def get_active_provider(self) -> BaseLLMProvider:
        """Get the currently active provider."""
        if self._active_provider is None:
            raise RuntimeError("No active LLM provider")
        return self._providers[self._active_provider]
    
    def list_available_providers(self) -> list:
        """List all available providers."""
        return [
            name for name, provider in self._providers.items()
            if provider.is_available()
        ]
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.1
    ) -> LLMResponse:
        """Generate using active provider."""
        provider = self.get_active_provider()
        return provider.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
