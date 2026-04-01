"""Hugging Face Whisper STT — runs locally, no API key needed."""

from __future__ import annotations

import numpy as np

from jvaf.core.frames import InputAudioFrame, TranscriptionFrame
from jvaf.providers.stt import STTProvider


class HFWhisperSTT(STTProvider):
    """Local Whisper via Hugging Face transformers.

    Runs on GPU if available, CPU otherwise. No API key needed.
    Requires: pip install jvaf[huggingface]
    """

    def __init__(self, *, language: str = "ja", model: str = "openai/whisper-large-v3", **kwargs):
        super().__init__(language=language, name="HFWhisperSTT")
        self._model_id = model
        self._pipe = None

    async def setup(self) -> None:
        import torch
        from transformers import pipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=self._model_id,
            device=device,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )

    async def transcribe(self, audio: InputAudioFrame) -> None:
        samples = audio.to_numpy().astype(np.float32)
        if len(samples) == 0:
            return

        result = self._pipe(
            {"raw": samples, "sampling_rate": audio.sample_rate},
            generate_kwargs={"language": self.language},
        )
        text = result.get("text", "").strip()
        if text:
            await self.push_frame(
                TranscriptionFrame(text=text, language=self.language, confidence=0.9)
            )
