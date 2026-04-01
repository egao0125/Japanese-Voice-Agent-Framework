"""Tests for frame types."""

import numpy as np
import pytest

from jvaf.core.frames import (
    AudioFrame,
    BackchannelTriggerFrame,
    BotSpeakingFrame,
    Frame,
    InputAudioFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMResponseFrame,
    OutputAudioFrame,
    StartFrame,
    StopFrame,
    SystemFrame,
    TranscriptionFrame,
    TTSAudioFrame,
    UninterruptibleAudioFrame,
    UserTurnEndFrame,
    VADEvent,
    VADState,
)


class TestFrame:
    def test_base_frame(self):
        f = Frame()
        assert f.timestamp > 0
        assert f.metadata == {}

    def test_system_frame(self):
        f = StartFrame()
        assert isinstance(f, SystemFrame)
        assert isinstance(f, Frame)

    def test_start_stop(self):
        assert StartFrame()
        assert StopFrame()


class TestAudioFrame:
    def test_audio_frame_roundtrip(self):
        samples = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        f = AudioFrame.from_numpy(samples, sample_rate=16000)
        recovered = f.to_numpy()
        np.testing.assert_array_almost_equal(recovered, samples, decimal=3)

    def test_from_numpy(self):
        samples = np.array([0.5, -0.5], dtype=np.float32)
        f = AudioFrame.from_numpy(samples, sample_rate=16000)
        assert f.sample_rate == 16000
        np.testing.assert_array_almost_equal(f.to_numpy(), samples)

    def test_input_output_types(self):
        audio = b"\x00" * 64
        assert isinstance(InputAudioFrame(audio=audio, sample_rate=16000), AudioFrame)
        assert isinstance(OutputAudioFrame(audio=audio, sample_rate=16000), AudioFrame)

    def test_uninterruptible(self):
        f = UninterruptibleAudioFrame(audio=b"\x00" * 4, sample_rate=16000)
        assert isinstance(f, OutputAudioFrame)

    def test_tts_audio(self):
        f = TTSAudioFrame(audio=b"\x00" * 4, sample_rate=16000, source_text="hello")
        assert f.source_text == "hello"
        assert isinstance(f, OutputAudioFrame)


class TestTranscriptionFrame:
    def test_transcription(self):
        f = TranscriptionFrame(text="hello", language="ja", confidence=0.95)
        assert f.text == "hello"
        assert f.confidence == 0.95

    def test_interim(self):
        f = InterimTranscriptionFrame(text="hel", language="ja")
        assert f.text == "hel"


class TestVAD:
    def test_vad_event(self):
        e = VADEvent(state=VADState.SPEAKING, confidence=0.9)
        assert e.state == VADState.SPEAKING


class TestControlFrames:
    def test_interruption(self):
        assert isinstance(InterruptionFrame(), SystemFrame)

    def test_user_turn_end(self):
        f = UserTurnEndFrame(text="test", confidence=0.8)
        assert f.text == "test"
        assert f.confidence == 0.8

    def test_bot_speaking(self):
        f = BotSpeakingFrame(is_speaking=True)
        assert f.is_speaking is True

    def test_backchannel_trigger(self):
        f = BackchannelTriggerFrame(signal="continuer", source="reactive")
        assert f.signal == "continuer"

    def test_llm_response(self):
        f = LLMResponseFrame(text="response", is_final=True)
        assert f.is_final is True
