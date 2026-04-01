"""Conversation simulator — runs scripted scenarios through the pipeline."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import numpy as np

from jvaf.agent import VoiceAgent
from jvaf.config import PipelineConfig
from jvaf.core.frames import (
    InputAudioFrame,
    OutputAudioFrame,
    TranscriptionFrame,
    UserTurnEndFrame,
    VADEvent,
    VADState,
)
from jvaf.autoresearch.config import TestScenario


@dataclass
class TurnRecord:
    """Record of a single conversational turn."""

    user_text: str = ""
    agent_text: str = ""
    latency_ms: float = 0.0
    backchannel_count: int = 0
    barge_in_triggered: bool = False


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


class ConversationSimulator:
    """Simulates multi-turn conversations for pipeline evaluation.

    Feeds scripted user utterances through a pipeline built from config,
    captures agent responses, and measures timing/quality metrics.
    """

    def __init__(self, sample_rate: int = 16000):
        self._sample_rate = sample_rate

    async def run_scenario(
        self, config: PipelineConfig, scenario: TestScenario
    ) -> SimulationResult:
        """Run a single scenario through a fresh pipeline."""
        result = SimulationResult(scenario_name=scenario.name)
        start = time.monotonic()

        agent = VoiceAgent.from_config(config)

        try:
            await agent.start()

            for utterance in scenario.user_utterances:
                turn = await self._simulate_turn(agent, utterance)
                result.turns.append(turn)

            await agent.stop()
        except Exception as e:
            result.pipeline_errors.append(str(e))

        result.total_duration_ms = (time.monotonic() - start) * 1000
        result.output_frame_count = len(agent.transport.recorded_output)  # type: ignore[attr-defined]
        result.backchannel_total = sum(t.backchannel_count for t in result.turns)

        return result

    async def run_all(
        self, config: PipelineConfig, scenarios: list[TestScenario]
    ) -> list[SimulationResult]:
        """Run all scenarios sequentially."""
        results = []
        for scenario in scenarios:
            # Add default utterances if none specified
            if not scenario.user_utterances:
                scenario.user_utterances = self._generate_default_utterances(scenario)
            result = await self.run_scenario(config, scenario)
            results.append(result)
        return results

    async def _simulate_turn(self, agent: VoiceAgent, utterance: str) -> TurnRecord:
        """Simulate a single user turn: audio → VAD → STT → LLM → TTS."""
        turn = TurnRecord(user_text=utterance)
        t0 = time.monotonic()

        # Generate synthetic audio for the utterance
        # Duration heuristic: ~150ms per character for Japanese
        duration_sec = max(0.5, len(utterance) * 0.15)
        audio = self._generate_speech_audio(duration_sec)

        pipeline = agent.pipeline
        if not pipeline:
            return turn

        # Feed audio in chunks (simulate real-time)
        chunk_samples = self._sample_rate // 10  # 100ms chunks
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i : i + chunk_samples]
            frame = InputAudioFrame(
                audio=chunk.tobytes(),
                sample_rate=self._sample_rate,
            )
            await pipeline.push_frame(frame)
            # Tiny yield to let pipeline process
            await asyncio.sleep(0)

        # Wait for pipeline to settle
        await asyncio.sleep(0.01)

        turn.latency_ms = (time.monotonic() - t0) * 1000

        # Count output frames for this turn
        output = agent.transport.recorded_output  # type: ignore[attr-defined]
        turn.agent_text = f"[{len(output)} output frames]"

        return turn

    def _generate_speech_audio(self, duration_sec: float) -> np.ndarray:
        """Generate synthetic speech-like audio (noise with envelope)."""
        n_samples = int(self._sample_rate * duration_sec)
        # White noise with amplitude modulation to simulate speech energy
        noise = np.random.randn(n_samples).astype(np.float32) * 0.3
        # Simple envelope: ramp up, sustain, ramp down
        envelope = np.ones(n_samples, dtype=np.float32)
        ramp = min(n_samples // 10, self._sample_rate // 10)
        envelope[:ramp] = np.linspace(0, 1, ramp)
        envelope[-ramp:] = np.linspace(1, 0, ramp)
        return noise * envelope

    def _generate_default_utterances(self, scenario: TestScenario) -> list[str]:
        """Generate placeholder utterances from scenario description."""
        return [
            f"[simulated: {scenario.description}]",
            "[simulated: follow-up question]",
            "[simulated: closing]",
        ]
