"""Deepgram STT provider — Nova-2 speech-to-text."""

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


class DeepgramSTT(STTProvider):
    """Deepgram Nova STT provider.

    Env: DEEPGRAM_API_KEY
    """

    def __init__(self, *, language: str = "ja", model: str = "nova-2", **kwargs):
        super().__init__(language=language, name="DeepgramSTT")
        self._model = model
        self._api_key = os.environ.get("DEEPGRAM_API_KEY", "")
        self._client = None

    async def setup(self) -> None:
        from deepgram import DeepgramClient
        if not self._api_key:
            raise ValueError("DEEPGRAM_API_KEY not set")
        self._client = DeepgramClient(self._api_key)

    async def transcribe(self, audio: InputAudioFrame) -> None:
        from deepgram import PrerecordedOptions

        wav_bytes = _pcm_to_wav(audio.audio, audio.sample_rate)
        source = {"buffer": wav_bytes, "mimetype": "audio/wav"}
        options = PrerecordedOptions(
            model=self._model,
            language=self.language,
            smart_format=True,
        )
        response = await self._client.listen.asyncrest.v("1").transcribe_file(source, options)
        alt = response.results.channels[0].alternatives[0]
        if alt.transcript.strip():
            await self.push_frame(
                TranscriptionFrame(
                    text=alt.transcript,
                    language=self.language,
                    confidence=alt.confidence,
                )
            )
