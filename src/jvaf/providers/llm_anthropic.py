"""Anthropic Claude LLM provider."""

from __future__ import annotations

import os

from jvaf.core.frames import LLMResponseFrame
from jvaf.providers.llm import LLMProvider


class AnthropicLLM(LLMProvider):
    """Anthropic Claude LLM provider.

    Env: ANTHROPIC_API_KEY
    """

    def __init__(
        self,
        *,
        system_prompt: str = "",
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ):
        super().__init__(system_prompt=system_prompt, name="AnthropicLLM")
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = None

    async def setup(self) -> None:
        from anthropic import AsyncAnthropic
        self._client = AsyncAnthropic()  # reads ANTHROPIC_API_KEY from env

    async def generate(self, user_text: str, history: list[dict[str, str]]) -> None:
        messages = [{"role": m["role"], "content": m["content"]} for m in history]

        full_text = ""
        async with self._client.messages.stream(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=self.system_prompt or "You are a helpful assistant.",
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                full_text += text
                await self.push_frame(LLMResponseFrame(text=text, is_final=False))

        self._history.append({"role": "assistant", "content": full_text})
        await self.push_frame(LLMResponseFrame(text=full_text, is_final=True))
