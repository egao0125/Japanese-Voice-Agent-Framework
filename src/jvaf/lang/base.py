"""Language pack ABC — encapsulates all language-specific configuration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class TurnTakingThresholds:
    """Language-specific turn-taking thresholds."""

    silence_with_prosody_sec: float = 0.5
    silence_without_prosody_sec: float = 0.7
    confidence_threshold: float = 0.65
    min_barge_in_chars: int = 4
    max_backchannel_duration_ms: float = 500.0
    weight_silence: float = 0.35
    weight_pitch: float = 0.30
    weight_energy: float = 0.15
    weight_duration: float = 0.10
    weight_lengthening: float = 0.10


@dataclass(frozen=True)
class BackchannelThresholds:
    """Language-specific backchannel timing."""

    min_interval_sec: float = 5.0
    neural_min_interval_sec: float = 1.5
    min_speech_before_bc_sec: float = 1.5
    min_pause_for_bc_sec: float = 0.05
    max_pause_for_bc_sec: float = 0.80
    proactive_threshold_sec: float = 3.5


class LanguagePack(ABC):
    """Encapsulates all language-specific configuration.

    Provides turn-taking thresholds, backchannel variants,
    and text processing rules per language.
    """

    @property
    @abstractmethod
    def code(self) -> str:
        """ISO 639-1 language code."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name."""
        ...

    @property
    @abstractmethod
    def turn_taking(self) -> TurnTakingThresholds:
        ...

    @property
    @abstractmethod
    def backchannel(self) -> BackchannelThresholds:
        ...

    @abstractmethod
    def get_backchannel_variants(self) -> dict[str, list[tuple[str, str]]]:
        """Return backchannel text variants per category.

        Returns dict mapping category name to list of (key, display_text) tuples.
        Categories: reactive, understanding, formal, empathy, surprise.
        """
        ...

    def normalize_for_tts(self, text: str) -> str:
        """Normalize text for TTS (override for language-specific rules)."""
        return text
