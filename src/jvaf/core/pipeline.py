"""Pipeline — connects FrameProcessors and manages frame flow."""

from __future__ import annotations

import asyncio

from .frames import Frame, FrameDirection, StartFrame, StopFrame
from .processor import FrameProcessor


class Pipeline:
    """Connects FrameProcessors in sequence and manages frame flow.

    Frames flow DOWNSTREAM through the chain by default.
    SystemFrames can flow UPSTREAM for interruption handling.
    """

    def __init__(self, processors: list[FrameProcessor]):
        self._processors = processors
        self._running = False
        self._wire()

    def _wire(self) -> None:
        """Wire push callbacks between adjacent processors."""
        for i, proc in enumerate(self._processors):
            # Downstream: this processor pushes to the next
            if i + 1 < len(self._processors):
                next_proc = self._processors[i + 1]

                async def push_down(frame: Frame, _next=next_proc) -> None:
                    await _next.process_frame(frame, FrameDirection.DOWNSTREAM)

                proc.set_downstream(push_down)

            # Upstream: this processor pushes to the previous
            if i > 0:
                prev_proc = self._processors[i - 1]

                async def push_up(frame: Frame, _prev=prev_proc) -> None:
                    await _prev.process_frame(frame, FrameDirection.UPSTREAM)

                proc.set_upstream(push_up)

    async def run(self) -> None:
        """Start the pipeline. Calls setup() on all processors, then sends StartFrame."""
        self._running = True
        for proc in self._processors:
            await proc.setup()
        await self._processors[0].process_frame(StartFrame(), FrameDirection.DOWNSTREAM)

    async def stop(self) -> None:
        """Stop the pipeline. Sends StopFrame, then calls cleanup()."""
        await self._processors[0].process_frame(StopFrame(), FrameDirection.DOWNSTREAM)
        self._running = False
        for proc in reversed(self._processors):
            await proc.cleanup()

    async def push_frame(self, frame: Frame) -> None:
        """Push a frame into the pipeline (at the first processor)."""
        if self._processors:
            await self._processors[0].process_frame(frame, FrameDirection.DOWNSTREAM)

    @property
    def processors(self) -> list[FrameProcessor]:
        return self._processors

    @property
    def running(self) -> bool:
        return self._running
