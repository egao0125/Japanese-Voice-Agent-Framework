"""FrameProcessor — the single interface all pipeline components implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Awaitable, Callable

from .frames import Frame, FrameDirection


class FrameProcessor(ABC):
    """Base class for all pipeline processors.

    Processors receive frames, optionally transform them, and push
    them to the next processor. The Pipeline handles frame routing.
    """

    def __init__(self, name: str | None = None):
        self._name = name or self.__class__.__name__
        self._push_downstream: Callable[[Frame], Awaitable[None]] | None = None
        self._push_upstream: Callable[[Frame], Awaitable[None]] | None = None

    @property
    def name(self) -> str:
        return self._name

    def set_downstream(self, callback: Callable[[Frame], Awaitable[None]]) -> None:
        """Called by Pipeline to wire downstream routing."""
        self._push_downstream = callback

    def set_upstream(self, callback: Callable[[Frame], Awaitable[None]]) -> None:
        """Called by Pipeline to wire upstream routing."""
        self._push_upstream = callback

    async def push_frame(
        self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
    ) -> None:
        """Push a frame to the next processor."""
        if direction == FrameDirection.DOWNSTREAM and self._push_downstream:
            await self._push_downstream(frame)
        elif direction == FrameDirection.UPSTREAM and self._push_upstream:
            await self._push_upstream(frame)

    @abstractmethod
    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Process a single frame. Override in subclasses."""
        ...

    async def setup(self) -> None:
        """Called once when the pipeline starts."""
        pass

    async def cleanup(self) -> None:
        """Called once when the pipeline stops."""
        pass


class PassthroughProcessor(FrameProcessor):
    """Processor that passes all frames through unchanged."""

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await self.push_frame(frame, direction)
