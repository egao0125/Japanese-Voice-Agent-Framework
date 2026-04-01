"""Mistral LLM provider."""

from __future__ import annotations

import os

from jvaf.core.frames import LLMResponseFrame
from jvaf.providers.llm import LLMProvider


class MistralLLM(LLMProvider):
    """Mistral AI LLM provider.

    Env: MISTRAL_API_KEY
    """

    def __init__(
        self,
        *,
        system_prompt: str = "",
        model: str = "mistral-medium-latest",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ):
        super().__init__(system_prompt=system_prompt, name="MistralLLM")
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = None

    async def setup(self) -> None:
        from mistralai import Mistral
        self._client = Mistral()  # reads MISTRAL_API_KEY from env

    async def generate(self, user_text: str, history: list[dict[str, str]]) -> None:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend({"role": m["role"], "content": m["content"]} for m in history)

        full_text = ""
        async for chunk in self._client.chat.stream_async(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        ):
            delta = chunk.data.choices[0].delta.content or ""
            if delta:
                full_text += delta
                await self.push_frame(LLMResponseFrame(text=delta, is_final=False))

        self._history.append({"role": "assistant", "content": full_text})
        await self.push_frame(LLMResponseFrame(text=full_text, is_final=True))
