"""Turn-taking strategies — detect when the user has finished speaking."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from jvaf.core.frames import (
    Frame,
    FrameDirection,
    TranscriptionFrame,
    UserTurnEndFrame,
    VADEvent,
    VADState,
)
from jvaf.core.processor import FrameProcessor


@dataclass
class TurnEndDecision:
    should_end: bool = False
    confidence: float = 0.0
    reason: str = ""


class TurnTakingStrategy(FrameProcessor, ABC):
    """Abstract turn-taking strategy.

    Receives VADEvents and TranscriptionFrames.
    Emits UserTurnEndFrame when it decides the user's turn is over.
    Passes all other frames through.
    """

    def __init__(self, name: str | None = None):
        super().__init__(name=name or self.__class__.__name__)
        self._accumulated_text = ""
        self._is_user_speaking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if isinstance(frame, VADEvent):
            if frame.state == VADState.SPEAKING:
                self._is_user_speaking = True
            elif frame.state == VADState.SILENCE:
                self._is_user_speaking = False
                decision = await self.evaluate_turn_end()
                if decision.should_end and self._accumulated_text.strip():
                    await self.push_frame(
                        UserTurnEndFrame(
                            text=self._accumulated_text.strip(),
                            confidence=decision.confidence,
                        )
                    )
                    self._accumulated_text = ""
            await self.push_frame(frame, direction)
        elif isinstance(frame, TranscriptionFrame):
            self._accumulated_text += frame.text
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    @abstractmethod
    async def evaluate_turn_end(self) -> TurnEndDecision:
        """Decide if the user's turn has ended after silence detected."""
        ...


class SilenceTurnTaking(TurnTakingStrategy):
    """Simple silence-based turn-taking.

    Ends the turn after any VAD silence event.
    Good enough for mock/demo, but real conversations need
    the HeuristicTurnTaking strategy.
    """

    def __init__(self, silence_threshold_sec: float = 0.5):
        super().__init__(name="SilenceTurnTaking")
        self._threshold = silence_threshold_sec

    async def evaluate_turn_end(self) -> TurnEndDecision:
        return TurnEndDecision(should_end=True, confidence=0.8, reason="silence")
