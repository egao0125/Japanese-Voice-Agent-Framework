"""Conversation simulator — runs scripted scenarios through the pipeline.

For mock backends, uses a parametric simulation model that makes config
changes produce meaningfully different scores.  For real backends, feeds
actual audio (synthesized or pre-recorded WAV) through the pipeline so
STT transcribes real speech and LLM responds to real questions.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from jvaf.agent import VoiceAgent
from jvaf.config import PipelineConfig
from jvaf.core.frames import InputAudioFrame
from jvaf.autoresearch.config import TestScenario


# ---------------------------------------------------------------------------
# Provider profiles: simulated quality/latency characteristics.
# Quality values represent how well the provider handles Japanese voice.
# Latency values are base per-call milliseconds.
# ---------------------------------------------------------------------------

_PROVIDER_PROFILES: dict[str, dict[str, dict[str, float]]] = {
    "stt": {
        "mock": {"latency_ms": 5, "quality": 0.70},
        "deepgram": {"latency_ms": 80, "quality": 0.93},
        "openai": {"latency_ms": 200, "quality": 0.88},
        "google": {"latency_ms": 120, "quality": 0.91},
        "hf_whisper": {"latency_ms": 500, "quality": 0.86},
    },
    "llm": {
        "mock": {"latency_ms": 5, "quality": 0.60},
        "anthropic": {"latency_ms": 200, "quality": 0.95},
        "openai": {"latency_ms": 180, "quality": 0.93},
        "google": {"latency_ms": 160, "quality": 0.88},
        "mistral": {"latency_ms": 120, "quality": 0.84},
        "groq": {"latency_ms": 50, "quality": 0.80},
        "hf_local": {"latency_ms": 400, "quality": 0.72},
    },
    "tts": {
        "mock": {"latency_ms": 5, "quality": 0.55},
        "elevenlabs": {"latency_ms": 200, "quality": 0.95},
        "openai": {"latency_ms": 150, "quality": 0.88},
        "google": {"latency_ms": 100, "quality": 0.83},
        "voicevox": {"latency_ms": 80, "quality": 0.78},
    },
}

# Sweet spots for tunable parameters — slightly offset from defaults
# so the search has something real to discover.
_TUNING_CURVES: dict[str, dict[str, float]] = {
    "vad.threshold_db": {"optimal": -38.0, "penalty_per_unit": 0.004},
    "vad.min_speech_ms": {"optimal": 220.0, "penalty_per_unit": 0.0003},
    "vad.min_silence_ms": {"optimal": 350.0, "penalty_per_unit": 0.0003},
    "turn_taking.silence_threshold_sec": {"optimal": 0.45, "penalty_per_unit": 0.08},
    "backchannel.min_interval_sec": {"optimal": 4.0, "penalty_per_unit": 0.015},
    "backchannel.min_speech_before_bc_sec": {"optimal": 1.8, "penalty_per_unit": 0.03},
    "barge_in.min_chars": {"optimal": 4.0, "penalty_per_unit": 0.025},
    "llm.temperature": {"optimal": 0.6, "penalty_per_unit": 0.12},
}


def _lookup(category: str, provider: str, key: str) -> float:
    return _PROVIDER_PROFILES.get(category, {}).get(provider, {}).get(key, 50.0)


@dataclass
class TurnRecord:
    """Record of a single conversational turn."""

    user_text: str = ""
    agent_text: str = ""
    latency_ms: float = 0.0
    backchannel_count: int = 0
    barge_in_triggered: bool = False
    quality_score: float = 0.0


@dataclass
class SimulationResult:
    """Result of running one scenario through the pipeline."""

    scenario_name: str = ""
    turns: list[TurnRecord] = field(default_factory=list)
    total_duration_ms: float = 0.0
    backchannel_total: int = 0
    output_frame_count: int = 0
    pipeline_errors: list[str] = field(default_factory=list)

    @property
    def avg_latency_ms(self) -> float:
        latencies = [t.latency_ms for t in self.turns if t.latency_ms > 0]
        return sum(latencies) / len(latencies) if latencies else 0.0

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def avg_quality(self) -> float:
        scores = [t.quality_score for t in self.turns if t.quality_score > 0]
        return sum(scores) / len(scores) if scores else 0.0


class ConversationSimulator:
    """Simulates multi-turn conversations for pipeline evaluation.

    Three modes:
    - Mock backend: parametric simulation model (fast, config-aware)
    - Real backend + audio: feeds WAV/generated audio through real pipeline
    - Real backend + no audio: feeds synthetic noise (fallback)
    """

    def __init__(self, sample_rate: int = 16000):
        self._sample_rate = sample_rate
        self._audio_cache: dict[str, list[Path]] = {}

    def set_audio_cache(self, cache: dict[str, list[Path]]) -> None:
        """Set pre-generated audio paths from AudioGenerator.

        Keys are scenario names, values are lists of WAV paths
        (one per utterance in order).
        """
        self._audio_cache = cache

    async def run_scenario(
        self, config: PipelineConfig, scenario: TestScenario
    ) -> SimulationResult:
        """Run a single scenario through a fresh pipeline."""
        if config.transport.type == "mock":
            return self._simulate_modeled(config, scenario)
        return await self._simulate_real(config, scenario)

    async def run_all(
        self, config: PipelineConfig, scenarios: list[TestScenario]
    ) -> list[SimulationResult]:
        """Run all scenarios sequentially."""
        results = []
        for scenario in scenarios:
            if not scenario.user_utterances:
                scenario.user_utterances = self._generate_default_utterances(scenario)
            result = await self.run_scenario(config, scenario)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Mock backend: parametric simulation model
    # ------------------------------------------------------------------

    def _simulate_modeled(
        self, config: PipelineConfig, scenario: TestScenario
    ) -> SimulationResult:
        """Config-aware simulation for mock backend.

        Provider choice affects base quality/latency.
        Threshold tuning affects score via sweet-spot curves.
        Small random noise prevents deterministic runs.
        """
        result = SimulationResult(scenario_name=scenario.name)

        # Compute config-dependent factors once
        provider_quality = self._model_provider_quality(config)
        tuning_penalty = self._model_tuning_penalty(config)
        base_latency = self._model_base_latency(config)

        for utterance in scenario.user_utterances:
            turn = TurnRecord(user_text=utterance)

            # Latency: provider base + silence threshold wait + noise
            silence_ms = config.turn_taking.silence_threshold_sec * 1000
            noise = random.gauss(0, base_latency * 0.08)
            turn.latency_ms = max(10.0, base_latency + silence_ms + noise)

            # Backchannel count: based on speech duration and BC config
            speech_sec = max(0.5, len(utterance) * 0.15)
            turn.backchannel_count = self._model_bc_count(config, speech_sec)

            # Quality: provider quality - tuning penalty + noise
            q_noise = random.gauss(0, 0.02)
            turn.quality_score = max(0.0, min(1.0,
                provider_quality - tuning_penalty + q_noise
            ))

            turn.agent_text = f"[simulated response to: {utterance[:30]}]"
            result.turns.append(turn)

        # Model stability: extreme VAD settings cause errors
        result.pipeline_errors = self._model_errors(config)

        result.total_duration_ms = sum(t.latency_ms for t in result.turns)
        result.output_frame_count = max(1, len(result.turns) * 3)
        result.backchannel_total = sum(t.backchannel_count for t in result.turns)

        return result

    def _model_provider_quality(self, config: PipelineConfig) -> float:
        """Aggregate quality from provider profiles."""
        stt_q = _lookup("stt", config.stt.provider, "quality")
        llm_q = _lookup("llm", config.llm.provider, "quality")
        tts_q = _lookup("tts", config.tts.provider, "quality")
        return stt_q * 0.30 + llm_q * 0.45 + tts_q * 0.25

    def _model_tuning_penalty(self, config: PipelineConfig) -> float:
        """Penalty for parameters being away from their sweet spots."""
        penalty = 0.0
        param_values = {
            "vad.threshold_db": config.vad.threshold_db,
            "vad.min_speech_ms": config.vad.min_speech_ms,
            "vad.min_silence_ms": config.vad.min_silence_ms,
            "turn_taking.silence_threshold_sec": config.turn_taking.silence_threshold_sec,
            "backchannel.min_interval_sec": config.backchannel.min_interval_sec,
            "backchannel.min_speech_before_bc_sec": config.backchannel.min_speech_before_bc_sec,
            "barge_in.min_chars": float(config.barge_in.min_chars),
            "llm.temperature": config.llm.temperature,
        }
        for path, current in param_values.items():
            curve = _TUNING_CURVES.get(path)
            if curve:
                distance = abs(current - curve["optimal"])
                penalty += distance * curve["penalty_per_unit"]
        return penalty

    def _model_base_latency(self, config: PipelineConfig) -> float:
        """Base latency from provider selection (STT + LLM + TTS)."""
        stt_lat = _lookup("stt", config.stt.provider, "latency_ms")
        llm_lat = _lookup("llm", config.llm.provider, "latency_ms")
        tts_lat = _lookup("tts", config.tts.provider, "latency_ms")
        return stt_lat + llm_lat + tts_lat

    def _model_bc_count(self, config: PipelineConfig, speech_sec: float) -> int:
        """Model backchannel count from timing config."""
        if not config.backchannel.enabled:
            return 0
        if speech_sec < config.backchannel.min_speech_before_bc_sec:
            return 0
        remaining = speech_sec - config.backchannel.min_speech_before_bc_sec
        if config.backchannel.min_interval_sec <= 0:
            return 0
        slots = int(remaining / config.backchannel.min_interval_sec)
        trigger_prob = min(0.9, len(config.backchannel.triggers) * 0.20)
        return sum(1 for _ in range(max(0, slots)) if random.random() < trigger_prob)

    def _model_errors(self, config: PipelineConfig) -> list[str]:
        """Model pipeline errors from extreme config values."""
        errors = []
        if config.vad.threshold_db < -50:
            if random.random() < 0.3:
                errors.append("VAD false positive cascade")
        if config.vad.threshold_db > -20:
            if random.random() < 0.4:
                errors.append("VAD missed user speech")
        if config.vad.min_speech_ms < 100:
            if random.random() < 0.2:
                errors.append("Audio fragmentation from low min_speech_ms")
        return errors

    # ------------------------------------------------------------------
    # Real backend: actual pipeline execution with audio
    # ------------------------------------------------------------------

    async def _simulate_real(
        self, config: PipelineConfig, scenario: TestScenario
    ) -> SimulationResult:
        """Run actual pipeline with real providers and real audio.

        If audio cache is available, feeds WAV files through the pipeline
        so STT transcribes real speech. Otherwise falls back to synthetic audio.
        """
        result = SimulationResult(scenario_name=scenario.name)
        start = time.monotonic()

        agent = VoiceAgent.from_config(config)
        audio_paths = self._audio_cache.get(scenario.name, [])

        try:
            await agent.start()

            # Track LLM history length before each turn to extract responses
            llm = agent._llm  # noqa: SLF001

            for i, utterance in enumerate(scenario.user_utterances):
                history_before = len(llm._history)  # noqa: SLF001

                # Use real audio if available, otherwise synthetic
                if i < len(audio_paths) and audio_paths[i].exists():
                    turn = await self._turn_with_audio(
                        agent, utterance, audio_paths[i]
                    )
                else:
                    turn = await self._turn_with_synthetic(agent, utterance)

                # Extract agent's response text from LLM history
                history_after = llm._history  # noqa: SLF001
                if len(history_after) > history_before:
                    # Last entry should be the assistant response
                    for entry in reversed(history_after[history_before:]):
                        if entry.get("role") == "assistant":
                            turn.agent_text = entry["content"]
                            break

                result.turns.append(turn)

            await agent.stop()
        except Exception as e:
            result.pipeline_errors.append(str(e))

        result.total_duration_ms = (time.monotonic() - start) * 1000
        result.output_frame_count = len(
            agent.transport.recorded_output  # type: ignore[attr-defined]
        )
        result.backchannel_total = sum(t.backchannel_count for t in result.turns)

        return result

    async def _turn_with_audio(
        self, agent: VoiceAgent, utterance: str, audio_path: Path
    ) -> TurnRecord:
        """Feed real WAV audio through the pipeline for one turn."""
        from jvaf.autoresearch.audio_gen import read_wav

        turn = TurnRecord(user_text=utterance)
        t0 = time.monotonic()

        audio = read_wav(audio_path, target_sample_rate=self._sample_rate)

        pipeline = agent.pipeline
        if not pipeline:
            return turn

        # Feed audio in 100ms chunks with real-time pacing
        chunk_samples = self._sample_rate // 10
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i : i + chunk_samples]
            frame = InputAudioFrame(
                audio=chunk.tobytes(),
                sample_rate=self._sample_rate,
            )
            await pipeline.push_frame(frame)
            # Pace at ~real-time to let pipeline process naturally
            await asyncio.sleep(0.05)

        # Wait for pipeline to finish processing
        await asyncio.sleep(0.2)

        turn.latency_ms = (time.monotonic() - t0) * 1000
        return turn

    async def _turn_with_synthetic(
        self, agent: VoiceAgent, utterance: str
    ) -> TurnRecord:
        """Fall back to synthetic noise audio for one turn."""
        turn = TurnRecord(user_text=utterance)
        t0 = time.monotonic()

        duration_sec = max(0.5, len(utterance) * 0.15)
        audio = self._generate_speech_audio(duration_sec)

        pipeline = agent.pipeline
        if not pipeline:
            return turn

        chunk_samples = self._sample_rate // 10
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i : i + chunk_samples]
            frame = InputAudioFrame(
                audio=chunk.tobytes(),
                sample_rate=self._sample_rate,
            )
            await pipeline.push_frame(frame)
            await asyncio.sleep(0)

        await asyncio.sleep(0.01)

        turn.latency_ms = (time.monotonic() - t0) * 1000
        output = agent.transport.recorded_output  # type: ignore[attr-defined]
        turn.agent_text = f"[{len(output)} output frames]"

        return turn

    def _generate_speech_audio(self, duration_sec: float) -> np.ndarray:
        """Generate synthetic speech-like audio (noise with envelope)."""
        n_samples = int(self._sample_rate * duration_sec)
        noise = np.random.randn(n_samples).astype(np.float32) * 0.3
        envelope = np.ones(n_samples, dtype=np.float32)
        ramp = min(n_samples // 10, self._sample_rate // 10)
        envelope[:ramp] = np.linspace(0, 1, ramp)
        envelope[-ramp:] = np.linspace(1, 0, ramp)
        return noise * envelope

    def _generate_default_utterances(self, scenario: TestScenario) -> list[str]:
        """Generate realistic Japanese utterances for simulation.

        Utterances need to be long enough (~6-10s of speech) to trigger
        backchannel evaluation. Japanese phone conversations have long
        turns with aizuchi opportunities.
        """
        _UTTERANCE_POOLS = [
            [
                "もしもし、お世話になっております。予約の件でお電話させていただきました。来週の火曜日に予約を取りたいのですが",
                "午後の三時頃でお願いできますでしょうか。もし空きがなければ別の日でも構いませんので",
                "はい、ありがとうございます。では来週の火曜日の午後三時でお願いいたします。失礼いたします",
            ],
            [
                "すみません、ちょっとお聞きしたいことがあるんですけど、先日の検査結果についてなんですが",
                "あの、結果が出るまでにどのくらいかかりますか。それと、もし異常があった場合は連絡いただけるのでしょうか",
                "わかりました。ご丁寧にありがとうございます。また何かあればご連絡いたします",
            ],
            [
                "こんにちは、初めてお電話します。ホームページを見てお電話したのですが、今度の土曜日に空きはありますか",
                "料金のことなんですけど、保険は使えますでしょうか。あと、初診の場合はどのくらい時間がかかりますか",
                "なるほど、了解しました。それでは土曜日の午前中でお願いしたいのですが、予約を入れていただけますか",
                "はい、名前は田中太郎です。電話番号は〇九〇の一二三四の五六七八です。よろしくお願いいたします",
            ],
        ]
        idx = hash(scenario.name) % len(_UTTERANCE_POOLS)
        return _UTTERANCE_POOLS[idx]
