"""OpenAI TTS provider."""

from __future__ import annotations

import os

from jvaf.core.frames import TTSAudioFrame
from jvaf.providers.tts import TTSProvider


class OpenAITTS(TTSProvider):
    """OpenAI TTS provider.

    Env: OPENAI_API_KEY
    Uses PCM output at 24kHz, resampled to target sample_rate.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        voice_id: str = "alloy",
        model: str = "tts-1",
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, name="OpenAITTS")
        self._voice = voice_id or "alloy"
        self._model = model
        self._client = None

    async def setup(self) -> None:
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI()

    async def synthesize(self, text: str) -> None:
        audio_bytes = await self.synthesize_to_bytes(text)
        await self.push_frame(
            TTSAudioFrame(
                audio=audio_bytes,
                sample_rate=self.sample_rate,
                source_text=text,
            )
        )

    async def synthesize_to_bytes(self, text: str) -> bytes:
        import numpy as np

        response = await self._client.audio.speech.create(
            model=self._model,
            voice=self._voice,
            input=text,
            response_format="pcm",  # Raw 24kHz 16-bit mono PCM
        )
        pcm_24k = np.frombuffer(response.content, dtype=np.int16)

        # Resample from 24kHz to target sample_rate if needed
        if self.sample_rate != 24000:
            indices = np.linspace(0, len(pcm_24k) - 1, int(len(pcm_24k) * self.sample_rate / 24000))
            pcm_resampled = np.interp(indices, np.arange(len(pcm_24k)), pcm_24k.astype(np.float64))
            return pcm_resampled.astype(np.int16).tobytes()
        return response.content
