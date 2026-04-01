"""Speech-to-text provider ABC and mock implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod

from jvaf.core.frames import (
    Frame,
    FrameDirection,
    InputAudioFrame,
    TranscriptionFrame,
)
from jvaf.core.processor import FrameProcessor


class STTProvider(FrameProcessor, ABC):
    """Abstract speech-to-text provider.

    Receives InputAudioFrame, emits TranscriptionFrame.
    """

    def __init__(self, *, language: str = "ja", name: str | None = None):
        super().__init__(name=name or self.__class__.__name__)
        self.language = language

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if isinstance(frame, InputAudioFrame) and direction == FrameDirection.DOWNSTREAM:
            await self.transcribe(frame)
        else:
            await self.push_frame(frame, direction)

    @abstractmethod
    async def transcribe(self, audio: InputAudioFrame) -> None:
        """Process audio and push TranscriptionFrame(s) downstream."""
        ...


class MockSTT(STTProvider):
    """Mock STT that returns configurable transcription.

    For autoresearch testing — simulates STT with configurable
    latency, accuracy, and responses.
    """

    def __init__(
        self,
        *,
        language: str = "ja",
        default_text: str = "もしもし、お世話になっております。",
        responses: list[str] | None = None,
    ):
        super().__init__(language=language, name="MockSTT")
        self._default_text = default_text
        self._responses = list(responses) if responses else []
        self._call_count = 0

    async def transcribe(self, audio: InputAudioFrame) -> None:
        if self._responses and self._call_count < len(self._responses):
            text = self._responses[self._call_count]
        else:
            text = self._default_text
        self._call_count += 1
        await self.push_frame(
            TranscriptionFrame(text=text, language=self.language, confidence=0.95)
        )
