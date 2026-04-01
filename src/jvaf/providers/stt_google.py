"""Google Cloud STT provider."""

from __future__ import annotations

import os

from jvaf.core.frames import InputAudioFrame, TranscriptionFrame
from jvaf.providers.stt import STTProvider


class GoogleSTT(STTProvider):
    """Google Cloud Speech-to-Text provider.

    Env: GOOGLE_API_KEY or GOOGLE_APPLICATION_CREDENTIALS
    """

    def __init__(self, *, language: str = "ja", model: str = "latest_long", **kwargs):
        super().__init__(language=language, name="GoogleSTT")
        self._model = model
        self._client = None

    async def setup(self) -> None:
        from google.cloud.speech_v2 import SpeechAsyncClient
        self._client = SpeechAsyncClient()

    async def transcribe(self, audio: InputAudioFrame) -> None:
        from google.cloud.speech_v2.types import (
            AutoDetectDecodingConfig,
            RecognitionConfig,
            RecognizeRequest,
        )

        config = RecognitionConfig(
            auto_decoding_config=AutoDetectDecodingConfig(),
            language_codes=[self.language],
            model=self._model,
        )

        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        request = RecognizeRequest(
            recognizer=f"projects/{project_id}/locations/global/recognizers/_",
            config=config,
            content=audio.audio,
        )

        response = await self._client.recognize(request=request)
        for result in response.results:
            alt = result.alternatives[0]
            if alt.transcript.strip():
                await self.push_frame(
                    TranscriptionFrame(
                        text=alt.transcript,
                        language=self.language,
                        confidence=alt.confidence,
                    )
                )
