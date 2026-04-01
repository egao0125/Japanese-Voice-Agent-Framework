"""Transport abstraction — audio I/O for different connection types."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from jvaf.core.frames import (
    Frame,
    FrameDirection,
    InputAudioFrame,
    OutputAudioFrame,
    UninterruptibleAudioFrame,
)
from jvaf.core.processor import FrameProcessor


class AudioInput(FrameProcessor):
    """Source of audio frames — first processor in the pipeline."""

    def __init__(self, name: str = "AudioInput"):
        super().__init__(name=name)
        self._queue: asyncio.Queue[InputAudioFrame] = asyncio.Queue()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await self.push_frame(frame, direction)

    async def feed(self, frame: InputAudioFrame) -> None:
        """External call to feed audio into the pipeline."""
        await self.push_frame(frame, FrameDirection.DOWNSTREAM)


class AudioOutput(FrameProcessor):
    """Sink for audio frames — last processor in the pipeline."""

    def __init__(self, name: str = "AudioOutput"):
        super().__init__(name=name)
        self.output_frames: list[OutputAudioFrame] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if isinstance(frame, OutputAudioFrame):
            self.output_frames.append(frame)
        else:
            await self.push_frame(frame, FrameDirection.UPSTREAM)


class Transport(ABC):
    """Combined audio I/O transport.

    Represents a connection to an audio source/sink pair (e.g.,
    WebSocket, local mic/speaker, SIP).
    """

    @abstractmethod
    def input(self) -> AudioInput:
        """Get the audio input processor."""
        ...

    @abstractmethod
    def output(self) -> AudioOutput:
        """Get the audio output processor."""
        ...

    @abstractmethod
    async def connect(self) -> None:
        """Establish the transport connection."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the transport connection."""
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        ...


class MockTransport(Transport):
    """Mock transport for testing — records output, plays scripted input."""

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        scripted_inputs: Sequence[InputAudioFrame] | None = None,
    ):
        self._sample_rate = sample_rate
        self._input = AudioInput(name="MockInput")
        self._output = AudioOutput(name="MockOutput")
        self._connected = False
        self._scripted = list(scripted_inputs) if scripted_inputs else []

    def input(self) -> AudioInput:
        return self._input

    def output(self) -> AudioOutput:
        return self._output

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def play_scripted(self) -> None:
        """Feed all scripted input frames into the pipeline."""
        for frame in self._scripted:
            await self._input.feed(frame)

    @property
    def recorded_output(self) -> list[OutputAudioFrame]:
        """Access all output frames that were sent."""
        return self._output.output_frames
