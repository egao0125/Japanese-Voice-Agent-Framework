"""Provider registry — maps (category, name) to provider classes with lazy imports."""

from __future__ import annotations

import os
from typing import Any

# Environment variable names for each provider
ENV_VARS: dict[str, str] = {
    "deepgram": "DEEPGRAM_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "elevenlabs": "ELEVENLABS_API_KEY",
    "google": "GOOGLE_API_KEY",
}

# Provider defaults: when swapping providers, apply these config values
PROVIDER_DEFAULTS: dict[str, dict[str, dict[str, Any]]] = {
    "stt": {
        "mock": {"model": ""},
        "deepgram": {"model": "nova-2"},
        "openai": {"model": "whisper-1"},
    },
    "llm": {
        "mock": {"model": ""},
        "anthropic": {"model": "claude-sonnet-4-20250514"},
        "openai": {"model": "gpt-4o-mini"},
    },
    "tts": {
        "mock": {"model": "", "voice_id": ""},
        "elevenlabs": {"model": "eleven_multilingual_v2", "voice_id": ""},
        "openai": {"model": "tts-1", "voice_id": "alloy"},
        "voicevox": {"model": "", "voice_id": "", "speaker_id": 1},
    },
    "vad": {
        "energy": {},
        "silero": {},
    },
}


class ProviderNotAvailable(Exception):
    """Raised when a provider's SDK or API key is not available."""


def _get_builtin(category: str, name: str) -> type:
    """Lazy-import builtin providers."""
    if category == "stt":
        if name == "mock":
            from jvaf.providers.stt import MockSTT
            return MockSTT
        if name == "deepgram":
            from jvaf.providers.stt_deepgram import DeepgramSTT
            return DeepgramSTT
        if name == "openai":
            from jvaf.providers.stt_openai import OpenAISTT
            return OpenAISTT
    elif category == "llm":
        if name == "mock":
            from jvaf.providers.llm import MockLLM
            return MockLLM
        if name == "anthropic":
            from jvaf.providers.llm_anthropic import AnthropicLLM
            return AnthropicLLM
        if name == "openai":
            from jvaf.providers.llm_openai import OpenAILLM
            return OpenAILLM
    elif category == "tts":
        if name == "mock":
            from jvaf.providers.tts import MockTTS
            return MockTTS
        if name == "elevenlabs":
            from jvaf.providers.tts_elevenlabs import ElevenLabsTTS
            return ElevenLabsTTS
        if name == "openai":
            from jvaf.providers.tts_openai import OpenAITTS
            return OpenAITTS
        if name == "voicevox":
            from jvaf.providers.tts_voicevox import VoicevoxTTS
            return VoicevoxTTS
    elif category == "vad":
        if name == "energy":
            from jvaf.providers.vad import EnergyVAD
            return EnergyVAD
        if name == "silero":
            from jvaf.providers.vad_silero import SileroVAD
            return SileroVAD

    raise ProviderNotAvailable(
        f"Unknown provider: {category}/{name}"
    )


# Custom registrations (for user-defined providers)
_CUSTOM: dict[str, dict[str, type]] = {}


def register(category: str, name: str, cls: type) -> None:
    """Register a custom provider class."""
    _CUSTOM.setdefault(category, {})[name] = cls


def get_class(category: str, name: str) -> type:
    """Get provider class by category and name.

    Checks custom registrations first, then builtin providers.
    """
    if category in _CUSTOM and name in _CUSTOM[category]:
        return _CUSTOM[category][name]
    try:
        return _get_builtin(category, name)
    except ImportError as e:
        # SDK not installed
        extras = {
            "deepgram": "deepgram",
            "anthropic": "anthropic",
            "openai": "openai",
            "elevenlabs": "elevenlabs",
            "silero": "silero",
        }
        extra = extras.get(name, name)
        raise ProviderNotAvailable(
            f"Provider '{name}' requires: pip install jvaf[{extra}]"
        ) from e


def available_providers(category: str) -> list[str]:
    """List providers that are importable and have API keys configured.

    Mock/local providers are always available. API-based providers
    require both the SDK installed and the env var set.
    """
    always_available = {"mock", "energy", "voicevox"}
    all_names = list(PROVIDER_DEFAULTS.get(category, {}).keys())
    result = []

    for name in all_names:
        if name in always_available:
            result.append(name)
            continue
        # Check env var
        env_var = ENV_VARS.get(name)
        if env_var and not os.environ.get(env_var):
            continue
        # Check SDK importable
        try:
            _get_builtin(category, name)
            result.append(name)
        except (ImportError, ProviderNotAvailable):
            continue

    # Include custom registrations
    if category in _CUSTOM:
        for name in _CUSTOM[category]:
            if name not in result:
                result.append(name)

    return result


def detect_all_available() -> dict[str, list[str]]:
    """Detect all available providers across all categories."""
    return {
        cat: available_providers(cat)
        for cat in ("stt", "llm", "tts", "vad")
    }
