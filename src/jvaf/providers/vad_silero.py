"""Silero VAD provider — neural voice activity detection."""

from __future__ import annotations

import time

import numpy as np

from jvaf.core.frames import Frame, FrameDirection, InputAudioFrame, VADEvent, VADState
from jvaf.providers.vad import VADProvider


class SileroVAD(VADProvider):
    """Silero VAD — high-accuracy neural VAD, runs locally on CPU.

    Requires: pip install jvaf[silero]
    """

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        min_speech_ms: float = 250,
        min_silence_ms: float = 300,
        **kwargs,
    ):
        super().__init__(
            threshold_db=0,  # Not used by Silero
            min_speech_ms=min_speech_ms,
            min_silence_ms=min_silence_ms,
            name="SileroVAD",
        )
        self._threshold = threshold
        self._model = None
        self._is_speaking = False
        self._state_start = 0.0

    async def setup(self) -> None:
        import torch
        self._model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if isinstance(frame, InputAudioFrame) and direction == FrameDirection.DOWNSTREAM:
            await self._analyze_silero(frame)
        await self.push_frame(frame, direction)

    async def _analyze_silero(self, frame: InputAudioFrame) -> None:
        import torch

        samples = frame.to_numpy()
        if len(samples) < 512:
            return

        tensor = torch.from_numpy(samples)
        prob = self._model(tensor, frame.sample_rate).item()

        now = time.monotonic()
        if prob >= self._threshold and not self._is_speaking:
            if now - self._state_start >= self._min_speech_ms / 1000:
                self._is_speaking = True
                self._state_start = now
                await self.push_frame(VADEvent(state=VADState.SPEAKING, confidence=prob))
        elif prob < self._threshold and self._is_speaking:
            if now - self._state_start >= self._min_silence_ms / 1000:
                self._is_speaking = False
                self._state_start = now
                await self.push_frame(VADEvent(state=VADState.SILENCE, confidence=1.0 - prob))

    async def analyze(self, audio: InputAudioFrame) -> None:
        # Handled in process_frame directly
        pass
