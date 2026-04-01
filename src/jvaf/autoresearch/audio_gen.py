"""Audio generator — synthesizes test utterances into WAV files.

Creates realistic audio input for pipeline testing.  Instead of feeding
numpy noise through STT, we synthesize the scripted user utterances
with TTS so the pipeline processes real speech.

The generated audio is cached to disk so it only needs to be synthesized
once regardless of how many autoresearch iterations run.

Usage:
    gen = AudioGenerator(cache_dir=Path("experiments/audio_cache"))
    await gen.setup()  # detect available TTS
    paths = await gen.generate_scenario(scenario)
    # paths = [Path("...scenario_1_00_a1b2c3.wav"), ...]
"""

from __future__ import annotations

import hashlib
import struct
import wave
from pathlib import Path

import numpy as np

from jvaf.autoresearch.config import TestScenario


# Provider preference for generating caller audio (separate from pipeline TTS)
_TTS_PREFERENCE = ["voicevox", "google", "openai", "elevenlabs"]


class AudioGenerator:
    """Synthesizes test utterances to WAV files using available TTS.

    Priority: user-provided WAV > voicevox (free local) > cloud TTS > sine tone fallback.
    """

    def __init__(
        self,
        cache_dir: Path,
        *,
        tts_provider: str = "auto",
        sample_rate: int = 16000,
    ):
        self._cache_dir = cache_dir
        self._tts_provider = tts_provider
        self._sample_rate = sample_rate
        self._tts = None
        self._provider_name = "tone"  # fallback

    async def setup(self) -> str:
        """Detect and initialize the best available TTS for audio generation.

        Returns the provider name that will be used.
        """
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        if self._tts_provider != "auto":
            self._provider_name = await self._try_provider(self._tts_provider)
            return self._provider_name

        # Auto-detect: try providers in preference order
        for provider in _TTS_PREFERENCE:
            name = await self._try_provider(provider)
            if name != "tone":
                self._provider_name = name
                return name

        self._provider_name = "tone"
        return "tone"

    async def _try_provider(self, provider: str) -> str:
        """Try to initialize a TTS provider and verify it can synthesize.

        Returns provider name on success, 'tone' on failure.
        """
        try:
            from jvaf.providers.registry import get_class

            cls = get_class("tts", provider)
            tts = cls(sample_rate=self._sample_rate)
            await tts.setup()

            # Verify synthesis actually works with a tiny test
            test_audio = await tts.synthesize_to_bytes("テスト")
            if not test_audio or len(test_audio) < 100:
                await tts.cleanup()
                return "tone"

            self._tts = tts
            return provider
        except Exception:
            return "tone"

    async def generate_scenario(self, scenario: TestScenario) -> list[Path]:
        """Generate WAV files for all utterances in a scenario.

        Returns list of WAV file paths (one per utterance).
        Skips generation if cache hit.
        """
        paths = []
        for i, utterance in enumerate(scenario.user_utterances):
            wav_path = self._cache_path(scenario.name, i, utterance)
            if not wav_path.exists():
                await self._generate_one(utterance, wav_path)
            paths.append(wav_path)
        return paths

    async def generate_all(
        self, scenarios: list[TestScenario]
    ) -> dict[str, list[Path]]:
        """Generate audio for all scenarios. Returns {scenario_name: [paths]}."""
        result = {}
        for scenario in scenarios:
            paths = await self.generate_scenario(scenario)
            result[scenario.name] = paths
        return result

    async def cleanup(self) -> None:
        """Clean up TTS provider."""
        if self._tts is not None:
            await self._tts.cleanup()
            self._tts = None

    @property
    def provider_name(self) -> str:
        return self._provider_name

    def _cache_path(self, scenario_name: str, index: int, text: str) -> Path:
        """Deterministic cache path based on text content."""
        h = hashlib.sha256(text.encode()).hexdigest()[:12]
        return self._cache_dir / f"{scenario_name}_{index:02d}_{h}.wav"

    async def _generate_one(self, text: str, path: Path) -> None:
        """Generate a single WAV file from text."""
        if self._tts is not None:
            audio_bytes = await self._synthesize_with_tts(text)
        else:
            audio_bytes = self._generate_tone_audio(text)

        _write_wav(path, audio_bytes, self._sample_rate)

    async def _synthesize_with_tts(self, text: str) -> bytes:
        """Use real TTS to synthesize speech."""
        return await self._tts.synthesize_to_bytes(text)

    def _generate_tone_audio(self, text: str) -> bytes:
        """Generate recognizable tone pattern as fallback.

        Not real speech, but activates VAD and gives STT something
        to process. Each character gets a distinct frequency burst
        so longer utterances produce longer audio.
        """
        duration_sec = max(0.5, len(text) * 0.12)
        n_samples = int(self._sample_rate * duration_sec)

        t = np.linspace(0, duration_sec, n_samples, dtype=np.float32)

        # Base frequency modulated by text content
        base_freq = 200.0
        audio = np.zeros(n_samples, dtype=np.float32)

        # Create frequency bursts for each "syllable"
        chars = max(1, len(text))
        samples_per_char = n_samples // chars

        for i in range(chars):
            start = i * samples_per_char
            end = min(start + samples_per_char, n_samples)
            freq = base_freq + (hash(text[i % len(text)]) % 300)
            segment_t = t[start:end] - t[start]
            audio[start:end] = 0.3 * np.sin(2 * np.pi * freq * segment_t)

        # Apply speech-like envelope
        envelope = np.ones(n_samples, dtype=np.float32)
        ramp = min(n_samples // 10, self._sample_rate // 10)
        if ramp > 0:
            envelope[:ramp] = np.linspace(0, 1, ramp)
            envelope[-ramp:] = np.linspace(1, 0, ramp)
        audio *= envelope

        # Convert float32 [-1, 1] to int16 PCM bytes
        audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        return audio_int16.tobytes()


def _write_wav(path: Path, pcm_bytes: bytes, sample_rate: int) -> None:
    """Write raw int16 PCM bytes to a WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def read_wav(path: Path, target_sample_rate: int = 16000) -> np.ndarray:
    """Read a WAV file and return int16 numpy array at target sample rate."""
    with wave.open(str(path), "rb") as wf:
        assert wf.getnchannels() <= 2, f"Expected mono/stereo, got {wf.getnchannels()} channels"
        assert wf.getsampwidth() == 2, f"Expected int16, got {wf.getsampwidth()} bytes"

        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
        source_rate = wf.getframerate()
        n_channels = wf.getnchannels()

    audio = np.frombuffer(raw, dtype=np.int16)

    # Stereo → mono
    if n_channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)

    # Resample if needed
    if source_rate != target_sample_rate:
        ratio = target_sample_rate / source_rate
        new_len = int(len(audio) * ratio)
        audio_f = audio.astype(np.float32)
        indices = np.linspace(0, len(audio_f) - 1, new_len)
        audio_f = np.interp(indices, np.arange(len(audio_f)), audio_f)
        audio = audio_f.astype(np.int16)

    return audio
