"""Tests for pipeline wiring and frame flow."""

import asyncio

import pytest

from jvaf.core.frames import (
    Frame,
    FrameDirection,
    InputAudioFrame,
    StartFrame,
    StopFrame,
    TranscriptionFrame,
)
from jvaf.core.pipeline import Pipeline
from jvaf.core.processor import FrameProcessor


class PassthroughProcessor(FrameProcessor):
    """Records frames that pass through."""

    def __init__(self, name: str = "passthrough"):
        super().__init__(name=name)
        self.seen: list[Frame] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        self.seen.append(frame)
        await self.push_frame(frame, direction)


class DropProcessor(FrameProcessor):
    """Drops all frames (sink)."""

    def __init__(self):
        super().__init__(name="drop")

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        pass  # intentionally drops


@pytest.mark.asyncio
async def test_pipeline_wiring():
    p1 = PassthroughProcessor("p1")
    p2 = PassthroughProcessor("p2")
    p3 = DropProcessor()

    pipeline = Pipeline([p1, p2, p3])
    await pipeline.run()

    # StartFrame should flow through p1 and p2
    assert any(isinstance(f, StartFrame) for f in p1.seen)
    assert any(isinstance(f, StartFrame) for f in p2.seen)


@pytest.mark.asyncio
async def test_pipeline_push_frame():
    p1 = PassthroughProcessor("p1")
    p2 = PassthroughProcessor("p2")
    p3 = DropProcessor()

    pipeline = Pipeline([p1, p2, p3])
    await pipeline.run()

    frame = TranscriptionFrame(text="test", language="ja", confidence=1.0)
    await pipeline.push_frame(frame)

    assert any(isinstance(f, TranscriptionFrame) for f in p1.seen)
    assert any(isinstance(f, TranscriptionFrame) for f in p2.seen)


@pytest.mark.asyncio
async def test_pipeline_stop():
    p1 = PassthroughProcessor("p1")
    p2 = DropProcessor()

    pipeline = Pipeline([p1, p2])
    await pipeline.run()
    assert pipeline.running

    await pipeline.stop()
    assert not pipeline.running


@pytest.mark.asyncio
async def test_processor_setup_cleanup():
    class TrackingProcessor(FrameProcessor):
        def __init__(self):
            super().__init__(name="tracking")
            self.setup_called = False
            self.cleanup_called = False

        async def setup(self):
            self.setup_called = True

        async def cleanup(self):
            self.cleanup_called = True

        async def process_frame(self, frame, direction):
            await self.push_frame(frame, direction)

    tp = TrackingProcessor()
    drop = DropProcessor()
    pipeline = Pipeline([tp, drop])

    await pipeline.run()
    assert tp.setup_called

    await pipeline.stop()
    assert tp.cleanup_called
