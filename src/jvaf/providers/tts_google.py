"""Google Cloud TTS provider."""

from __future__ import annotations

import os

from jvaf.core.frames import TTSAudioFrame
from jvaf.providers.tts import TTSProvider


class GoogleTTS(TTSProvider):
    """Google Cloud Text-to-Speech provider.

    Env: GOOGLE_API_KEY or GOOGLE_APPLICATION_CREDENTIALS
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        voice_id: str = "ja-JP-Neural2-B",
        model: str = "",
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, name="GoogleTTS")
        self._voice_name = voice_id
        self._client = None

    async def setup(self) -> None:
        from google.cloud import texttospeech_v1 as tts
        self._client = tts.TextToSpeechAsyncClient()

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
        from google.cloud import texttospeech_v1 as tts

        synthesis_input = tts.SynthesisInput(text=text)
        voice = tts.VoiceSelectionParams(
            language_code="ja-JP",
            name=self._voice_name,
        )
        audio_config = tts.AudioConfig(
            audio_encoding=tts.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.sample_rate,
        )
        response = await self._client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )
        # LINEAR16 response is raw PCM (no WAV header)
        return response.audio_content
