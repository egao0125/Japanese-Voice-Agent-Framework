"""Groq LLM provider — ultra-fast inference."""

from __future__ import annotations

import os

from jvaf.core.frames import LLMResponseFrame
from jvaf.providers.llm import LLMProvider


class GroqLLM(LLMProvider):
    """Groq LLM provider (OpenAI-compatible API, ultra-fast inference).

    Env: GROQ_API_KEY
    """

    def __init__(
        self,
        *,
        system_prompt: str = "",
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ):
        super().__init__(system_prompt=system_prompt, name="GroqLLM")
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = None

    async def setup(self) -> None:
        from groq import AsyncGroq
        self._client = AsyncGroq()  # reads GROQ_API_KEY from env

    async def generate(self, user_text: str, history: list[dict[str, str]]) -> None:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend({"role": m["role"], "content": m["content"]} for m in history)

        full_text = ""
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                full_text += delta
                await self.push_frame(LLMResponseFrame(text=delta, is_final=False))

        self._history.append({"role": "assistant", "content": full_text})
        await self.push_frame(LLMResponseFrame(text=full_text, is_final=True))
