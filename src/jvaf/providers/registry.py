"""Provider registry — data-driven, extensible provider resolution.

Adding a new provider requires:
1. Write the provider file (e.g., stt_google.py)
2. Add one entry to _BUILTIN_PROVIDERS below

That's it. The registry handles lazy imports, availability detection,
and default config values automatically.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any


@dataclass
class ProviderEntry:
    """Everything the registry needs to know about a provider."""

    module: str  # e.g. "jvaf.providers.stt_google"
    cls_name: str  # e.g. "GoogleSTT"
    env_var: str = ""  # required env var (empty = local/no key needed)
    pip_extra: str = ""  # pip install jvaf[extra]
    defaults: dict[str, Any] = field(default_factory=dict)  # config defaults on swap
    local: bool = False  # True = no API key needed (local model/engine)


# ---------------------------------------------------------------------------
# Builtin provider catalog — add new providers here
# ---------------------------------------------------------------------------

_BUILTIN_PROVIDERS: dict[str, dict[str, ProviderEntry]] = {
    "stt": {
        "mock": ProviderEntry(
            module="jvaf.providers.stt", cls_name="MockSTT",
            local=True, defaults={"model": ""},
        ),
        "deepgram": ProviderEntry(
            module="jvaf.providers.stt_deepgram", cls_name="DeepgramSTT",
            env_var="DEEPGRAM_API_KEY", pip_extra="deepgram",
            defaults={"model": "nova-2"},
        ),
        "openai": ProviderEntry(
            module="jvaf.providers.stt_openai", cls_name="OpenAISTT",
            env_var="OPENAI_API_KEY", pip_extra="openai",
            defaults={"model": "whisper-1"},
        ),
        "google": ProviderEntry(
            module="jvaf.providers.stt_google", cls_name="GoogleSTT",
            env_var="GOOGLE_API_KEY", pip_extra="google",
            defaults={"model": "latest_long"},
        ),
        "hf_whisper": ProviderEntry(
            module="jvaf.providers.stt_hf_whisper", cls_name="HFWhisperSTT",
            local=True, pip_extra="huggingface",
            defaults={"model": "openai/whisper-large-v3"},
        ),
    },
    "llm": {
        "mock": ProviderEntry(
            module="jvaf.providers.llm", cls_name="MockLLM",
            local=True, defaults={"model": ""},
        ),
        "anthropic": ProviderEntry(
            module="jvaf.providers.llm_anthropic", cls_name="AnthropicLLM",
            env_var="ANTHROPIC_API_KEY", pip_extra="anthropic",
            defaults={"model": "claude-sonnet-4-20250514"},
        ),
        "openai": ProviderEntry(
            module="jvaf.providers.llm_openai", cls_name="OpenAILLM",
            env_var="OPENAI_API_KEY", pip_extra="openai",
            defaults={"model": "gpt-4o-mini"},
        ),
        "google": ProviderEntry(
            module="jvaf.providers.llm_google", cls_name="GoogleLLM",
            env_var="GOOGLE_API_KEY", pip_extra="google",
            defaults={"model": "gemini-2.0-flash"},
        ),
        "mistral": ProviderEntry(
            module="jvaf.providers.llm_mistral", cls_name="MistralLLM",
            env_var="MISTRAL_API_KEY", pip_extra="mistral",
            defaults={"model": "mistral-medium-latest"},
        ),
        "groq": ProviderEntry(
            module="jvaf.providers.llm_groq", cls_name="GroqLLM",
            env_var="GROQ_API_KEY", pip_extra="groq",
            defaults={"model": "llama-3.3-70b-versatile"},
        ),
        "hf_local": ProviderEntry(
            module="jvaf.providers.llm_hf_local", cls_name="HFLocalLLM",
            local=True, pip_extra="huggingface",
            defaults={"model": "meta-llama/Llama-3.1-8B-Instruct"},
        ),
    },
    "tts": {
        "mock": ProviderEntry(
            module="jvaf.providers.tts", cls_name="MockTTS",
            local=True, defaults={"model": "", "voice_id": ""},
        ),
        "elevenlabs": ProviderEntry(
            module="jvaf.providers.tts_elevenlabs", cls_name="ElevenLabsTTS",
            env_var="ELEVENLABS_API_KEY", pip_extra="elevenlabs",
            defaults={"model": "eleven_multilingual_v2", "voice_id": ""},
        ),
        "openai": ProviderEntry(
            module="jvaf.providers.tts_openai", cls_name="OpenAITTS",
            env_var="OPENAI_API_KEY", pip_extra="openai",
            defaults={"model": "tts-1", "voice_id": "alloy"},
        ),
        "voicevox": ProviderEntry(
            module="jvaf.providers.tts_voicevox", cls_name="VoicevoxTTS",
            local=True, defaults={"model": "", "voice_id": "", "speaker_id": 1},
        ),
        "google": ProviderEntry(
            module="jvaf.providers.tts_google", cls_name="GoogleTTS",
            env_var="GOOGLE_API_KEY", pip_extra="google",
            defaults={"model": "", "voice_id": "ja-JP-Neural2-B"},
        ),
    },
    "vad": {
        "energy": ProviderEntry(
            module="jvaf.providers.vad", cls_name="EnergyVAD",
            local=True, defaults={},
        ),
        "silero": ProviderEntry(
            module="jvaf.providers.vad_silero", cls_name="SileroVAD",
            local=True, pip_extra="silero", defaults={},
        ),
    },
}


# ---------------------------------------------------------------------------
# Custom registrations (user-defined or plugin providers)
# ---------------------------------------------------------------------------

_CUSTOM: dict[str, dict[str, ProviderEntry]] = {}


class ProviderNotAvailable(Exception):
    """Raised when a provider's SDK or API key is not available."""


def register(
    category: str,
    name: str,
    cls: type | None = None,
    *,
    module: str = "",
    cls_name: str = "",
    env_var: str = "",
    pip_extra: str = "",
    defaults: dict[str, Any] | None = None,
    local: bool = False,
) -> None:
    """Register a provider. Can pass a class directly or a module path for lazy import.

    Examples:
        # Direct class registration
        register("stt", "my_stt", MySTTProvider)

        # Lazy import registration
        register("stt", "my_stt", module="my_pkg.stt", cls_name="MySTT",
                 env_var="MY_API_KEY", defaults={"model": "v1"})
    """
    if cls is not None:
        # Wrap class in an entry for consistency
        entry = ProviderEntry(
            module="", cls_name=cls.__name__,
            env_var=env_var, pip_extra=pip_extra,
            defaults=defaults or {}, local=local,
        )
        entry._cls = cls  # type: ignore[attr-defined]
    else:
        entry = ProviderEntry(
            module=module, cls_name=cls_name,
            env_var=env_var, pip_extra=pip_extra,
            defaults=defaults or {}, local=local,
        )
    _CUSTOM.setdefault(category, {})[name] = entry


def _resolve_entry(entry: ProviderEntry) -> type:
    """Resolve a ProviderEntry to its class via lazy import."""
    # Check for directly registered class
    if hasattr(entry, "_cls"):
        return entry._cls  # type: ignore[attr-defined]
    mod = import_module(entry.module)
    return getattr(mod, entry.cls_name)


def _find_entry(category: str, name: str) -> ProviderEntry:
    """Look up a provider entry by category and name."""
    # Custom first
    if category in _CUSTOM and name in _CUSTOM[category]:
        return _CUSTOM[category][name]
    # Builtin
    cat = _BUILTIN_PROVIDERS.get(category, {})
    if name in cat:
        return cat[name]
    raise ProviderNotAvailable(f"Unknown provider: {category}/{name}")


def get_class(category: str, name: str) -> type:
    """Get provider class by category and name.

    Lazy-imports the module on first access. Raises ProviderNotAvailable
    if the SDK is not installed.
    """
    entry = _find_entry(category, name)
    try:
        return _resolve_entry(entry)
    except ImportError as e:
        hint = f"pip install jvaf[{entry.pip_extra}]" if entry.pip_extra else str(e)
        raise ProviderNotAvailable(
            f"Provider '{name}' requires: {hint}"
        ) from e


def get_defaults(category: str, name: str) -> dict[str, Any]:
    """Get default config values for a provider."""
    try:
        entry = _find_entry(category, name)
        return dict(entry.defaults)
    except ProviderNotAvailable:
        return {}


def get_env_var(name: str) -> str:
    """Get the env var name for a provider."""
    for cat in list(_BUILTIN_PROVIDERS.values()) + list(_CUSTOM.values()):
        if name in cat:
            return cat[name].env_var
    return ""


def available_providers(category: str) -> list[str]:
    """List providers that are importable and have API keys configured.

    Local providers are always available (if SDK installed).
    API providers require both SDK + env var.
    """
    entries: dict[str, ProviderEntry] = {}
    entries.update(_BUILTIN_PROVIDERS.get(category, {}))
    entries.update(_CUSTOM.get(category, {}))

    result = []
    for name, entry in entries.items():
        # Check env var for API providers
        if not entry.local and entry.env_var and not os.environ.get(entry.env_var):
            continue
        # Check SDK importable
        if entry.pip_extra:
            try:
                _resolve_entry(entry)
            except (ImportError, AttributeError):
                continue
        result.append(name)
    return result


def detect_all_available() -> dict[str, list[str]]:
    """Detect all available providers across all categories."""
    return {
        cat: available_providers(cat)
        for cat in ("stt", "llm", "tts", "vad")
    }


def list_all_providers() -> dict[str, list[str]]:
    """List ALL registered providers (available or not)."""
    result = {}
    for cat in set(list(_BUILTIN_PROVIDERS) + list(_CUSTOM)):
        entries: dict[str, ProviderEntry] = {}
        entries.update(_BUILTIN_PROVIDERS.get(cat, {}))
        entries.update(_CUSTOM.get(cat, {}))
        result[cat] = list(entries.keys())
    return result


# Backwards-compatible PROVIDER_DEFAULTS dict (computed from entries)
PROVIDER_DEFAULTS: dict[str, dict[str, dict[str, Any]]] = {
    cat: {name: dict(entry.defaults) for name, entry in entries.items()}
    for cat, entries in _BUILTIN_PROVIDERS.items()
}
