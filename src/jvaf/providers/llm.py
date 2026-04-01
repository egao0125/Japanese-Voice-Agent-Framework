"""LLM provider ABC and mock implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod

from jvaf.core.frames import (
    Frame,
    FrameDirection,
    LLMResponseFrame,
    TranscriptionFrame,
    UserTurnEndFrame,
)
from jvaf.core.processor import FrameProcessor


class LLMProvider(FrameProcessor, ABC):
    """Abstract LLM provider.

    Receives TranscriptionFrame or UserTurnEndFrame,
    emits LLMResponseFrame (streaming text chunks).
    """

    def __init__(self, *, system_prompt: str = "", name: str | None = None):
        super().__init__(name=name or self.__class__.__name__)
        self.system_prompt = system_prompt
        self._history: list[dict[str, str]] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if isinstance(frame, (TranscriptionFrame, UserTurnEndFrame)):
            text = frame.text if hasattr(frame, "text") else ""
            if text.strip():
                self._history.append({"role": "user", "content": text})
                await self.generate(text, self._history)
        else:
            await self.push_frame(frame, direction)

    @abstractmethod
    async def generate(self, user_text: str, history: list[dict[str, str]]) -> None:
        """Generate a response and push LLMResponseFrame(s) downstream."""
        ...


class MockLLM(LLMProvider):
    """Mock LLM that returns configurable responses.

    For autoresearch testing — simulates LLM with configurable
    responses and streaming behavior.
    """

    def __init__(
        self,
        *,
        response: str = "はい、承知いたしました。",
        responses: list[str] | None = None,
        system_prompt: str = "",
    ):
        super().__init__(system_prompt=system_prompt, name="MockLLM")
        self._default_response = response
        self._responses = list(responses) if responses else []
        self._call_count = 0

    async def generate(self, user_text: str, history: list[dict[str, str]]) -> None:
        if self._responses and self._call_count < len(self._responses):
            text = self._responses[self._call_count]
        else:
            text = self._default_response
        self._call_count += 1
        self._history.append({"role": "assistant", "content": text})
        await self.push_frame(LLMResponseFrame(text=text, is_final=True))
