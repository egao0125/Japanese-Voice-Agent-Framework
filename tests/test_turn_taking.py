"""Tests for turn-taking strategies."""

import pytest

from jvaf.conversation.turn_taking import (
    SilenceTurnTaking,
    TurnEndDecision,
    TurnTakingStrategy,
)
from jvaf.core.frames import (
    FrameDirection,
    TranscriptionFrame,
    UserTurnEndFrame,
    VADEvent,
    VADState,
)
from jvaf.core.processor import FrameProcessor


class FrameCollector(FrameProcessor):
    def __init__(self):
        super().__init__(name="collector")
        self.frames = []

    async def process_frame(self, frame, direction):
        self.frames.append(frame)


@pytest.mark.asyncio
async def test_silence_turn_taking():
    tt = SilenceTurnTaking()
    collector = FrameCollector()
    tt.set_downstream(lambda f: collector.process_frame(f, FrameDirection.DOWNSTREAM))

    # User speaks
    await tt.process_frame(VADEvent(state=VADState.SPEAKING), FrameDirection.DOWNSTREAM)
    await tt.process_frame(
        TranscriptionFrame(text="テスト", language="ja", confidence=1.0),
        FrameDirection.DOWNSTREAM,
    )
    # User stops
    await tt.process_frame(VADEvent(state=VADState.SILENCE), FrameDirection.DOWNSTREAM)

    # Should emit UserTurnEndFrame
    turn_ends = [f for f in collector.frames if isinstance(f, UserTurnEndFrame)]
    assert len(turn_ends) == 1
    assert turn_ends[0].text == "テスト"


@pytest.mark.asyncio
async def test_silence_turn_no_text():
    """Silence without accumulated text should NOT emit UserTurnEndFrame."""
    tt = SilenceTurnTaking()
    collector = FrameCollector()
    tt.set_downstream(lambda f: collector.process_frame(f, FrameDirection.DOWNSTREAM))

    await tt.process_frame(VADEvent(state=VADState.SPEAKING), FrameDirection.DOWNSTREAM)
    await tt.process_frame(VADEvent(state=VADState.SILENCE), FrameDirection.DOWNSTREAM)

    turn_ends = [f for f in collector.frames if isinstance(f, UserTurnEndFrame)]
    assert len(turn_ends) == 0


@pytest.mark.asyncio
async def test_text_accumulation():
    tt = SilenceTurnTaking()
    collector = FrameCollector()
    tt.set_downstream(lambda f: collector.process_frame(f, FrameDirection.DOWNSTREAM))

    await tt.process_frame(VADEvent(state=VADState.SPEAKING), FrameDirection.DOWNSTREAM)
    await tt.process_frame(
        TranscriptionFrame(text="予約", language="ja", confidence=1.0),
        FrameDirection.DOWNSTREAM,
    )
    await tt.process_frame(
        TranscriptionFrame(text="したい", language="ja", confidence=1.0),
        FrameDirection.DOWNSTREAM,
    )
    await tt.process_frame(VADEvent(state=VADState.SILENCE), FrameDirection.DOWNSTREAM)

    turn_ends = [f for f in collector.frames if isinstance(f, UserTurnEndFrame)]
    assert len(turn_ends) == 1
    assert turn_ends[0].text == "予約したい"
