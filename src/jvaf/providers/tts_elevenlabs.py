"""ElevenLabs TTS provider."""

from __future__ import annotations

import os

import numpy as np

from jvaf.core.frames import TTSAudioFrame
from jvaf.providers.tts import TTSProvider


class ElevenLabsTTS(TTSProvider):
    """ElevenLabs TTS provider.

    Env: ELEVENLABS_API_KEY
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        voice_id: str = "",
        model: str = "eleven_multilingual_v2",
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, name="ElevenLabsTTS")
        self._voice_id = voice_id
        self._model = model
        self._client = None

    async def setup(self) -> None:
        from elevenlabs import AsyncElevenLabs
        self._client = AsyncElevenLabs()  # reads ELEVENLABS_API_KEY from env

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
        from elevenlabs import VoiceSettings

        voice = self._voice_id or "JBFqnCBsd6RMkjVDRZzb"  # default: George

        audio_iter = await self._client.text_to_speech.convert(
            voice_id=voice,
            text=text,
            model_id=self._model,
            voice_settings=VoiceSettings(stability=0.5, similarity_boost=0.75),
        )

        # Collect all audio chunks (MP3 bytes)
        mp3_chunks = []
        async for chunk in audio_iter:
            mp3_chunks.append(chunk)
        mp3_bytes = b"".join(mp3_chunks)

        # Convert MP3 to raw PCM int16
        return self._mp3_to_pcm(mp3_bytes)

    def _mp3_to_pcm(self, mp3_bytes: bytes) -> bytes:
        """Convert MP3 bytes to int16 PCM. Falls back to silence on error."""
        try:
            import io
            import array

            # Try pydub if available
            from pydub import AudioSegment
            seg = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
            seg = seg.set_frame_rate(self.sample_rate).set_channels(1).set_sample_width(2)
            return seg.raw_data
        except ImportError:
            # Fallback: return silence proportional to input size
            duration_ms = max(100, len(mp3_bytes) // 16)
            return self._generate_silence(duration_ms)
