"""Tests for audio generator and WAV-based simulation."""

import wave

import numpy as np
import pytest

from jvaf.autoresearch.audio_gen import AudioGenerator, read_wav
from jvaf.autoresearch.config import TestScenario
from jvaf.autoresearch.simulator import ConversationSimulator
from jvaf.config import PipelineConfig


class TestAudioGenerator:
    @pytest.mark.asyncio
    async def test_generates_wav_files(self, tmp_path):
        """Should generate valid WAV files regardless of TTS provider."""
        gen = AudioGenerator(cache_dir=tmp_path / "audio")
        await gen.setup()

        scenario = TestScenario(
            name="test_scenario",
            description="Test",
            user_utterances=["こんにちは", "予約したいです"],
        )
        paths = await gen.generate_scenario(scenario)

        assert len(paths) == 2
        for p in paths:
            assert p.exists()
            assert p.suffix == ".wav"
            # Verify valid WAV
            with wave.open(str(p), "rb") as wf:
                assert wf.getnchannels() == 1
                assert wf.getsampwidth() == 2
                assert wf.getnframes() > 0

        await gen.cleanup()

    @pytest.mark.asyncio
    async def test_cache_reuse(self, tmp_path):
        gen = AudioGenerator(cache_dir=tmp_path / "audio")
        await gen.setup()

        scenario = TestScenario(
            name="cache_test",
            description="Test",
            user_utterances=["テスト"],
        )

        paths1 = await gen.generate_scenario(scenario)
        mtime1 = paths1[0].stat().st_mtime

        paths2 = await gen.generate_scenario(scenario)
        mtime2 = paths2[0].stat().st_mtime

        assert paths1[0] == paths2[0]
        assert mtime1 == mtime2
        await gen.cleanup()

    @pytest.mark.asyncio
    async def test_generate_all(self, tmp_path):
        gen = AudioGenerator(cache_dir=tmp_path / "audio")
        await gen.setup()

        scenarios = [
            TestScenario(name="s1", description="S1", user_utterances=["あ", "い"]),
            TestScenario(name="s2", description="S2", user_utterances=["う"]),
        ]
        result = await gen.generate_all(scenarios)

        assert "s1" in result
        assert "s2" in result
        assert len(result["s1"]) == 2
        assert len(result["s2"]) == 1
        await gen.cleanup()

    @pytest.mark.asyncio
    async def test_explicit_tone_provider(self, tmp_path):
        """Forcing 'tone' provider should always work without API keys."""
        gen = AudioGenerator(cache_dir=tmp_path / "audio", tts_provider="tone")
        provider = await gen.setup()
        assert provider == "tone"

        scenario = TestScenario(
            name="tone_test",
            description="Test",
            user_utterances=["テスト音声"],
        )
        paths = await gen.generate_scenario(scenario)
        assert paths[0].exists()

        with wave.open(str(paths[0]), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
            assert wf.getnframes() > 0

        await gen.cleanup()

    @pytest.mark.asyncio
    async def test_provider_detection(self, tmp_path):
        """Auto-detection should return a valid provider name."""
        gen = AudioGenerator(cache_dir=tmp_path / "audio")
        provider = await gen.setup()

        # Should be one of the known providers or 'tone'
        valid = {"tone", "voicevox", "google", "openai", "elevenlabs"}
        assert provider in valid
        await gen.cleanup()


class TestReadWav:
    def test_read_wav_basic(self, tmp_path):
        wav_path = tmp_path / "test.wav"
        samples = np.array([0, 1000, -1000, 5000, -5000], dtype=np.int16)

        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(samples.tobytes())

        audio = read_wav(wav_path)
        assert len(audio) == 5
        np.testing.assert_array_equal(audio, samples)

    def test_read_wav_resample(self, tmp_path):
        wav_path = tmp_path / "48k.wav"
        n_samples = 4800  # 100ms at 48kHz
        samples = np.zeros(n_samples, dtype=np.int16)
        samples[::2] = 1000

        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(48000)
            wf.writeframes(samples.tobytes())

        audio = read_wav(wav_path, target_sample_rate=16000)
        expected_len = int(n_samples * (16000 / 48000))
        assert len(audio) == expected_len


class TestSimulatorWithAudio:
    @pytest.mark.asyncio
    async def test_audio_cache_integration(self, tmp_path):
        """Simulator should accept audio cache from AudioGenerator."""
        gen = AudioGenerator(cache_dir=tmp_path / "audio", tts_provider="tone")
        await gen.setup()

        scenario = TestScenario(
            name="audio_test",
            description="Test with audio",
            user_utterances=["テスト音声です"],
        )
        audio_cache = await gen.generate_all([scenario])
        await gen.cleanup()

        sim = ConversationSimulator()
        sim.set_audio_cache(audio_cache)

        # With mock config, still uses parametric model
        config = PipelineConfig()
        result = await sim.run_scenario(config, scenario)
        assert result.turn_count == 1
        assert result.total_duration_ms > 0

    @pytest.mark.asyncio
    async def test_mock_backend_ignores_audio(self):
        """Mock backend should use parametric model regardless of audio cache."""
        sim = ConversationSimulator()
        sim.set_audio_cache({"test": []})

        config = PipelineConfig()  # transport.type = "mock"
        scenario = TestScenario(
            name="test",
            description="Test",
            user_utterances=["テスト"],
        )
        result = await sim.run_scenario(config, scenario)
        assert result.turns[0].quality_score > 0
