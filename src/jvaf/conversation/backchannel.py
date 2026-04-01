"""Backchannel system — signals, triggers, selector, injector.

First-class backchannel (aizuchi) support with 4 trigger paths,
signal taxonomy, shuffle-bag selection, and interrupt-resistant injection.
"""

from __future__ import annotations

import random
import time
from enum import Enum

import numpy as np

from jvaf.core.frames import (
    BackchannelTriggerFrame,
    BotSpeakingFrame,
    Frame,
    FrameDirection,
    InputAudioFrame,
    UninterruptibleAudioFrame,
    VADEvent,
    VADState,
)
from jvaf.core.processor import FrameProcessor
from jvaf.providers.tts import TTSProvider


# ---------------------------------------------------------------------------
# Signal taxonomy
# ---------------------------------------------------------------------------


class BackchannelSignal(str, Enum):
    CONTINUER = "continuer"
    ASSESSMENT = "assessment"
    FORMAL_ACK = "formal_ack"
    EMPATHETIC = "empathetic"
    SURPRISED = "surprised"


SIGNAL_TO_CATEGORIES: dict[str, list[str]] = {
    "continuer": ["reactive", "understanding"],
    "assessment": ["understanding", "reactive"],
    "formal_ack": ["formal", "reactive"],
    "empathetic": ["empathy", "understanding"],
    "surprised": ["surprise", "reactive"],
}


# ---------------------------------------------------------------------------
# Selector — shuffle-bag text selection
# ---------------------------------------------------------------------------


class BackchannelSelector:
    """Selects backchannel text using shuffle-bag for even distribution."""

    def __init__(self, variants: dict[str, list[tuple[str, str]]]):
        self._variants = variants
        self._bags: dict[str, list[tuple[str, str]]] = {}

    def select(self, signal: str) -> tuple[str, str] | None:
        """Select a (key, text) pair for the given signal."""
        categories = SIGNAL_TO_CATEGORIES.get(signal, ["reactive"])
        for cat in categories:
            if cat in self._variants and self._variants[cat]:
                return self._pick_from_bag(cat)
        return None

    def _pick_from_bag(self, category: str) -> tuple[str, str]:
        if category not in self._bags or not self._bags[category]:
            self._bags[category] = list(self._variants[category])
            random.shuffle(self._bags[category])
        return self._bags[category].pop()


# ---------------------------------------------------------------------------
# Trigger detector
# ---------------------------------------------------------------------------


class BackchannelTriggerDetector(FrameProcessor):
    """Detects backchannel opportunities via multiple paths.

    Trigger paths:
    1. Reactive (pause-resume): user pauses 50-800ms then resumes
    2. Energy dip: sub-VAD energy drop detection
    3. Proactive: user speaks continuously for N seconds

    Emits BackchannelTriggerFrame on detection.
    """

    def __init__(
        self,
        *,
        min_speech_before_bc: float = 1.5,
        min_pause_for_bc: float = 0.05,
        max_pause_for_bc: float = 0.80,
        min_bc_interval: float = 5.0,
        proactive_threshold: float = 3.5,
        energy_dip_threshold_db: float = 30.0,
    ):
        super().__init__(name="BCTriggerDetector")
        self._min_speech_before = min_speech_before_bc
        self._min_pause = min_pause_for_bc
        self._max_pause = max_pause_for_bc
        self._min_interval = min_bc_interval
        self._proactive_threshold = proactive_threshold
        self._energy_threshold = energy_dip_threshold_db

        self._last_bc_time = 0.0
        self._speech_start = 0.0
        self._silence_start = 0.0
        self._cumulative_speech = 0.0
        self._is_speaking = False
        self._bot_speaking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        now = time.monotonic()

        if isinstance(frame, BotSpeakingFrame):
            self._bot_speaking = frame.is_speaking
        elif isinstance(frame, VADEvent):
            if frame.state == VADState.SPEAKING:
                if not self._is_speaking:
                    # User started speaking — check reactive trigger
                    pause_dur = now - self._silence_start if self._silence_start else 0
                    if (
                        self._min_pause <= pause_dur <= self._max_pause
                        and self._cumulative_speech >= self._min_speech_before
                        and not self._bot_speaking
                        and (now - self._last_bc_time) >= self._min_interval
                    ):
                        self._last_bc_time = now
                        await self.push_frame(
                            BackchannelTriggerFrame(signal="continuer", source="reactive")
                        )
                    self._speech_start = now
                self._is_speaking = True
            elif frame.state == VADState.SILENCE:
                if self._is_speaking:
                    self._cumulative_speech += now - self._speech_start
                    self._silence_start = now
                self._is_speaking = False

                # Proactive: long continuous speech
                if (
                    self._cumulative_speech >= self._proactive_threshold
                    and not self._bot_speaking
                    and (now - self._last_bc_time) >= self._min_interval
                ):
                    self._last_bc_time = now
                    self._cumulative_speech = 0
                    await self.push_frame(
                        BackchannelTriggerFrame(signal="continuer", source="proactive")
                    )
        elif isinstance(frame, InputAudioFrame):
            # Energy dip detection on audio frames
            samples = frame.to_numpy()
            if len(samples) > 0 and self._is_speaking and not self._bot_speaking:
                rms = np.sqrt(np.mean(samples**2))
                db = 20 * np.log10(max(rms, 1e-10))
                if db < -self._energy_threshold and (now - self._last_bc_time) >= self._min_interval:
                    self._last_bc_time = now
                    await self.push_frame(
                        BackchannelTriggerFrame(signal="continuer", source="energy")
                    )

        await self.push_frame(frame, direction)


# ---------------------------------------------------------------------------
# Injector
# ---------------------------------------------------------------------------


class BackchannelInjector(FrameProcessor):
    """Injects backchannel audio into the output stream.

    On BackchannelTriggerFrame: selects text, looks up cached TTS audio,
    wraps as UninterruptibleAudioFrame, and pushes downstream.
    """

    def __init__(
        self,
        *,
        tts: TTSProvider,
        selector: BackchannelSelector,
        sample_rate: int = 16000,
    ):
        super().__init__(name="BCInjector")
        self._tts = tts
        self._selector = selector
        self._sample_rate = sample_rate
        self._cache: dict[str, bytes] = {}

    async def setup(self) -> None:
        """Pre-synthesize all backchannel variants."""
        for cat_variants in self._selector._variants.values():
            for key, text in cat_variants:
                if key not in self._cache:
                    audio_bytes = await self._tts.synthesize_to_bytes(text)
                    self._cache[key] = audio_bytes
        print(f"BC cache warmed: {len(self._cache)} entries")

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if isinstance(frame, BackchannelTriggerFrame):
            selection = self._selector.select(frame.signal)
            if selection:
                key, text = selection
                audio = self._cache.get(key, b"")
                if audio:
                    await self.push_frame(
                        UninterruptibleAudioFrame(
                            audio=audio,
                            sample_rate=self._sample_rate,
                            metadata={"backchannel": True, "text": text},
                        )
                    )
        else:
            await self.push_frame(frame, direction)
