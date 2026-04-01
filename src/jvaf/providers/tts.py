"""Text-to-speech provider ABC and mock implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from jvaf.core.frames import (
    Frame,
    FrameDirection,
    LLMResponseFrame,
    TTSAudioFrame,
)
from jvaf.core.processor import FrameProcessor


class TTSProvider(FrameProcessor, ABC):
    """Abstract text-to-speech provider.

    Receives LLMResponseFrame, emits TTSAudioFrame.
    """

    def __init__(self, *, sample_rate: int = 16000, name: str | None = None):
        super().__init__(name=name or self.__class__.__name__)
        self.sample_rate = sample_rate

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if isinstance(frame, LLMResponseFrame) and direction == FrameDirection.DOWNSTREAM:
            await self.synthesize(frame.text)
        else:
            await self.push_frame(frame, direction)

    @abstractmethod
    async def synthesize(self, text: str) -> None:
        """Synthesize text and push TTSAudioFrame(s) downstream."""
        ...

    async def synthesize_to_bytes(self, text: str) -> bytes:
        """Synthesize text and return raw audio bytes (for caching)."""
        return self._generate_silence(len(text) * 50)

    def _generate_silence(self, duration_ms: float) -> bytes:
        """Generate silence of given duration."""
        n_samples = int(self.sample_rate * duration_ms / 1000)
        return np.zeros(n_samples, dtype=np.int16).tobytes()


class MockTTS(TTSProvider):
    """Mock TTS that generates silence with correct duration.

    Simulates ~100ms per character for Japanese text.
    """

    def __init__(self, *, sample_rate: int = 16000):
        super().__init__(sample_rate=sample_rate, name="MockTTS")

    async def synthesize(self, text: str) -> None:
        duration_ms = max(100, len(text) * 100)
        audio_bytes = self._generate_silence(duration_ms)
        await self.push_frame(
            TTSAudioFrame(
                audio=audio_bytes,
                sample_rate=self.sample_rate,
                source_text=text,
            )
        )

    async def synthesize_to_bytes(self, text: str) -> bytes:
        duration_ms = max(100, len(text) * 100)
        return self._generate_silence(duration_ms)
