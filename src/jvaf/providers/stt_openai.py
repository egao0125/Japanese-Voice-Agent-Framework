"""OpenAI Whisper STT provider."""

from __future__ import annotations

import io
import os
import struct

from jvaf.core.frames import InputAudioFrame, TranscriptionFrame
from jvaf.providers.stt import STTProvider


def _pcm_to_wav(pcm: bytes, sample_rate: int, channels: int = 1, sample_width: int = 2) -> bytes:
    """Wrap raw PCM bytes in a WAV header."""
    buf = io.BytesIO()
    data_size = len(pcm)
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, channels, sample_rate,
                          sample_rate * channels * sample_width, channels * sample_width,
                          sample_width * 8))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm)
    return buf.getvalue()


class OpenAISTT(STTProvider):
    """OpenAI Whisper STT provider.

    Env: OPENAI_API_KEY
    """

    def __init__(self, *, language: str = "ja", model: str = "whisper-1", **kwargs):
        super().__init__(language=language, name="OpenAISTT")
        self._model = model
        self._client = None

    async def setup(self) -> None:
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI()  # reads OPENAI_API_KEY from env

    async def transcribe(self, audio: InputAudioFrame) -> None:
        wav_bytes = _pcm_to_wav(audio.audio, audio.sample_rate)
        wav_file = io.BytesIO(wav_bytes)
        wav_file.name = "audio.wav"

        response = await self._client.audio.transcriptions.create(
            model=self._model,
            file=wav_file,
            language=self.language,
        )
        if response.text.strip():
            await self.push_frame(
                TranscriptionFrame(
                    text=response.text,
                    language=self.language,
                    confidence=0.9,  # Whisper API doesn't return confidence
                )
            )
