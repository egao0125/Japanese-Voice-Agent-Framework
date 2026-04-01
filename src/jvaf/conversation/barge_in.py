"""Barge-in strategies — detect when user speech should interrupt the bot."""

from __future__ import annotations

from abc import ABC, abstractmethod

from jvaf.core.frames import (
    BotSpeakingFrame,
    Frame,
    FrameDirection,
    InterimTranscriptionFrame,
    InterruptionFrame,
    TranscriptionFrame,
)
from jvaf.core.processor import FrameProcessor


class BargeInStrategy(FrameProcessor, ABC):
    """Abstract barge-in strategy.

    Monitors transcription during bot speech. If the user is
    saying something substantial (not just aizuchi), emits
    InterruptionFrame to cancel bot output.
    """

    def __init__(self, name: str | None = None):
        super().__init__(name=name or self.__class__.__name__)
        self._bot_speaking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if isinstance(frame, BotSpeakingFrame):
            self._bot_speaking = frame.is_speaking
            await self.push_frame(frame, direction)
        elif isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            if self._bot_speaking and await self.should_interrupt(frame.text):
                await self.push_frame(InterruptionFrame(), FrameDirection.UPSTREAM)
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    @abstractmethod
    async def should_interrupt(self, text: str) -> bool:
        """Decide if user text should interrupt bot speech."""
        ...


class CharCountBargeIn(BargeInStrategy):
    """Character-count based barge-in for Japanese.

    Blocks aizuchi like "はい"(2 chars) while allowing real
    interruptions like "すみません"(5 chars).
    """

    def __init__(self, min_chars: int = 4):
        super().__init__(name="CharCountBargeIn")
        self._min_chars = min_chars

    async def should_interrupt(self, text: str) -> bool:
        return len(text.strip()) >= self._min_chars
