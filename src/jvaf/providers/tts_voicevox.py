"""VOICEVOX TTS provider — local Japanese TTS engine."""

from __future__ import annotations

import os

from jvaf.core.frames import TTSAudioFrame
from jvaf.providers.tts import TTSProvider


class VoicevoxTTS(TTSProvider):
    """VOICEVOX local TTS provider.

    No API key needed — requires a running VOICEVOX engine.
    Env: VOICEVOX_URL (default: http://localhost:50021)
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        speaker_id: int = 1,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, name="VoicevoxTTS")
        self._speaker_id = speaker_id
        self._base_url = os.environ.get("VOICEVOX_URL", "http://localhost:50021")
        self._http = None

    async def setup(self) -> None:
        import httpx
        self._http = httpx.AsyncClient(base_url=self._base_url, timeout=30.0)
        # Verify VOICEVOX is running
        resp = await self._http.get("/version")
        resp.raise_for_status()

    async def cleanup(self) -> None:
        if self._http:
            await self._http.aclose()

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
        # Step 1: audio query
        resp = await self._http.post(
            "/audio_query",
            params={"text": text, "speaker": self._speaker_id},
        )
        resp.raise_for_status()
        query = resp.json()

        # Step 2: synthesis
        resp = await self._http.post(
            "/synthesis",
            params={"speaker": self._speaker_id},
            json=query,
        )
        resp.raise_for_status()

        # Response is WAV — strip 44-byte header to get raw PCM
        wav_bytes = resp.content
        return wav_bytes[44:]
