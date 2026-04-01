"""Hugging Face local LLM — runs on GPU/CPU, no API key needed."""

from __future__ import annotations

from jvaf.core.frames import LLMResponseFrame
from jvaf.providers.llm import LLMProvider


class HFLocalLLM(LLMProvider):
    """Local LLM via Hugging Face transformers.

    Runs on GPU if available, CPU otherwise. No API key needed.
    Requires: pip install jvaf[huggingface]
    """

    def __init__(
        self,
        *,
        system_prompt: str = "",
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ):
        super().__init__(system_prompt=system_prompt, name="HFLocalLLM")
        self._model_id = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._pipe = None

    async def setup(self) -> None:
        import torch
        from transformers import pipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._pipe = pipeline(
            "text-generation",
            model=self._model_id,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )

    async def generate(self, user_text: str, history: list[dict[str, str]]) -> None:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(history)

        result = self._pipe(
            messages,
            max_new_tokens=self._max_tokens,
            temperature=self._temperature,
            do_sample=self._temperature > 0,
            return_full_text=False,
        )
        text = result[0]["generated_text"].strip()
        self._history.append({"role": "assistant", "content": text})
        await self.push_frame(LLMResponseFrame(text=text, is_final=True))
