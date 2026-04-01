"""Voice activity detection provider ABC and energy-based implementation."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np

from jvaf.core.frames import (
    Frame,
    FrameDirection,
    InputAudioFrame,
    VADEvent,
    VADState,
)
from jvaf.core.processor import FrameProcessor


class VADProvider(FrameProcessor, ABC):
    """Abstract VAD provider.

    Receives InputAudioFrame, emits VADEvent on state transitions.
    Always passes audio frames through unchanged.
    """

    def __init__(
        self,
        *,
        min_speech_ms: float = 250,
        min_silence_ms: float = 300,
        name: str | None = None,
    ):
        super().__init__(name=name or self.__class__.__name__)
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self._state = VADState.SILENCE
        self._state_start = time.monotonic()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if isinstance(frame, InputAudioFrame) and direction == FrameDirection.DOWNSTREAM:
            event = await self.analyze(frame)
            if event is not None:
                await self.push_frame(event, direction)
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    @abstractmethod
    async def analyze(self, audio: InputAudioFrame) -> VADEvent | None:
        """Analyze audio for voice activity. Return event on state change."""
        ...

    @property
    def state(self) -> VADState:
        return self._state


class EnergyVAD(VADProvider):
    """Simple energy-based VAD.

    Uses RMS energy threshold to detect speech/silence transitions.
    Suitable for mock/demo pipelines. For production, use Silero VAD.
    """

    def __init__(
        self,
        *,
        threshold_db: float = -35.0,
        min_speech_ms: float = 250,
        min_silence_ms: float = 300,
        **kwargs,
    ):
        super().__init__(min_speech_ms=min_speech_ms, min_silence_ms=min_silence_ms, name="EnergyVAD")
        self._threshold_db = threshold_db

    async def analyze(self, audio: InputAudioFrame) -> VADEvent | None:
        samples = audio.to_numpy()
        if len(samples) == 0:
            return None

        rms = np.sqrt(np.mean(samples**2))
        db = 20 * np.log10(max(rms, 1e-10))
        is_speech = db > self._threshold_db

        now = time.monotonic()
        elapsed_ms = (now - self._state_start) * 1000

        if is_speech and self._state == VADState.SILENCE:
            if elapsed_ms >= self.min_silence_ms or self._state_start == 0:
                self._state = VADState.SPEAKING
                self._state_start = now
                return VADEvent(state=VADState.SPEAKING, confidence=min(1.0, db / -20))
        elif not is_speech and self._state == VADState.SPEAKING:
            if elapsed_ms >= self.min_speech_ms:
                self._state = VADState.SILENCE
                self._state_start = now
                return VADEvent(state=VADState.SILENCE, confidence=1.0)
        return None
