"""Google Gemini LLM provider."""

from __future__ import annotations

import os

from jvaf.core.frames import LLMResponseFrame
from jvaf.providers.llm import LLMProvider


class GoogleLLM(LLMProvider):
    """Google Gemini LLM provider.

    Env: GOOGLE_API_KEY
    """

    def __init__(
        self,
        *,
        system_prompt: str = "",
        model: str = "gemini-2.0-flash",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ):
        super().__init__(system_prompt=system_prompt, name="GoogleLLM")
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = None

    async def setup(self) -> None:
        from google import genai
        self._client = genai.Client()  # reads GOOGLE_API_KEY from env

    async def generate(self, user_text: str, history: list[dict[str, str]]) -> None:
        from google.genai.types import Content, GenerateContentConfig, Part

        contents = []
        if self.system_prompt:
            contents.append(Content(role="user", parts=[Part(text=f"System: {self.system_prompt}")]))
            contents.append(Content(role="model", parts=[Part(text="Understood.")]))
        for msg in history:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append(Content(role=role, parts=[Part(text=msg["content"])]))

        config = GenerateContentConfig(
            temperature=self._temperature,
            max_output_tokens=self._max_tokens,
        )

        full_text = ""
        async for chunk in self._client.aio.models.generate_content_stream(
            model=self._model,
            contents=contents,
            config=config,
        ):
            text = chunk.text or ""
            if text:
                full_text += text
                await self.push_frame(LLMResponseFrame(text=text, is_final=False))

        self._history.append({"role": "assistant", "content": full_text})
        await self.push_frame(LLMResponseFrame(text=full_text, is_final=True))
