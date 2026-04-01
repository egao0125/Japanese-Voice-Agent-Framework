"""Pipeline evaluator — scores simulation results against program goals.

Two evaluation layers:
  1. Infrastructure metrics: latency, stability, throughput, backchannel
  2. Content quality: task completion, language, accuracy, naturalness
     (via LLM-as-judge for real backends, parametric model for mock)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jvaf.autoresearch.config import AutoresearchConfig
from jvaf.autoresearch.judge import ContentJudge, ContentScore
from jvaf.autoresearch.simulator import SimulationResult


@dataclass
class EvalScore:
    """Evaluation result with overall score and per-metric breakdown."""

    overall: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)
    details: dict[str, str] = field(default_factory=dict)
    content_scores: list[ContentScore] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"overall={self.overall:.3f}"]
        for k, v in self.metrics.items():
            parts.append(f"{k}={v:.3f}")
        return " | ".join(parts)


class PipelineEvaluator:
    """Evaluates pipeline quality against program goals.

    Infrastructure metrics (from simulator):
    - quality: provider + tuning score (simulation model) or content judge
    - latency: average turn latency (lower is better)
    - stability: error rate
    - backchannel: BC rate per turn (target: 0.3-0.7)
    - throughput: conversation completeness

    Content metrics (from LLM judge, real backends only):
    - task_completion: scenario goal progress
    - language_quality: keigo, formality, grammar
    - factual_accuracy: no hallucination
    - naturalness: conversation flow
    """

    # Default weights
    _DEFAULT_WEIGHTS = {
        "quality": 0.35,
        "latency": 0.25,
        "stability": 0.20,
        "backchannel": 0.10,
        "throughput": 0.10,
    }

    def __init__(
        self,
        *,
        backend: str = "mock",
        use_case: str = "",
        focus_params: set[str] | None = None,
    ):
        self._judge = ContentJudge(backend=backend, use_case=use_case)
        self._backend = backend
        self._weights = self._compute_weights(focus_params or set())

    def _compute_weights(self, focus_params: set[str]) -> dict[str, float]:
        """Adjust metric weights based on improvement focus.

        If user says 'improve latency', latency weight increases.
        If 'improve backchannel', backchannel weight increases.
        Weights are normalized to sum to 1.0.
        """
        if not focus_params:
            return dict(self._DEFAULT_WEIGHTS)

        weights = dict(self._DEFAULT_WEIGHTS)

        # Boost weights for focused areas
        focus_text = " ".join(focus_params)
        if "latency" in focus_text or "silence_threshold" in focus_text:
            weights["latency"] += 0.15
        if "backchannel" in focus_text or "triggers" in focus_text:
            weights["backchannel"] += 0.15
        if "provider_stt" in focus_text or "provider_llm" in focus_text or "provider_tts" in focus_text:
            weights["quality"] += 0.10
        if "threshold_db" in focus_text or "min_speech" in focus_text or "min_silence" in focus_text:
            weights["stability"] += 0.05

        # Normalize to 1.0
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def evaluate(
        self, results: list[SimulationResult], config: AutoresearchConfig
    ) -> EvalScore:
        """Score a set of simulation results against program goals."""
        if not results:
            return EvalScore(overall=0.0)

        metrics: dict[str, float] = {}
        details: dict[str, str] = {}

        # Quality score (from simulation model — reflects provider choice + tuning)
        avg_qualities = [r.avg_quality for r in results if r.avg_quality > 0]
        if avg_qualities:
            quality = sum(avg_qualities) / len(avg_qualities)
            metrics["quality"] = quality
            details["quality"] = f"avg={quality:.3f}"
        else:
            # Fallback: binary output check for real backends
            total_output = sum(r.output_frame_count for r in results)
            metrics["quality"] = 1.0 if total_output > 0 else 0.0
            details["quality"] = f"{total_output} frames"

        # Latency score (lower is better)
        # Target: <300ms = 1.0, 300-800ms = good, >2000ms = 0.0
        avg_latencies = [r.avg_latency_ms for r in results if r.avg_latency_ms > 0]
        if avg_latencies:
            avg_lat = sum(avg_latencies) / len(avg_latencies)
            lat_score = max(0.0, 1.0 - (avg_lat / 1500.0) ** 0.8)
            metrics["latency"] = lat_score
            details["latency"] = f"avg={avg_lat:.0f}ms"
        else:
            metrics["latency"] = 0.5
            details["latency"] = "no latency data"

        # Stability score (no errors = 1.0)
        total_scenarios = len(results)
        error_scenarios = sum(1 for r in results if r.pipeline_errors)
        stability = 1.0 - (error_scenarios / total_scenarios) if total_scenarios else 0.0
        metrics["stability"] = stability
        details["stability"] = f"{error_scenarios}/{total_scenarios} errors"

        # Backchannel rate (target: 0.3-0.7 BCs per turn)
        total_turns = sum(r.turn_count for r in results)
        bc_total = sum(r.backchannel_total for r in results)
        if total_turns > 0:
            bc_rate = bc_total / total_turns
            if 0.3 <= bc_rate <= 0.7:
                bc_score = 1.0
            elif bc_rate < 0.3:
                bc_score = bc_rate / 0.3
            else:
                bc_score = max(0.0, 1.0 - (bc_rate - 0.7) / 1.0)
            metrics["backchannel"] = bc_score
            details["backchannel"] = f"{bc_total} total, {bc_rate:.2f}/turn"

        # Throughput: turns processed without error
        expected_turns = sum(len(s.user_utterances) for s in config.test_scenarios) or total_turns
        throughput = min(1.0, total_turns / expected_turns) if expected_turns else 1.0
        metrics["throughput"] = throughput
        details["throughput"] = f"{total_turns} turns"

        # Overall: weighted average (focus-adjusted)
        overall = sum(metrics.get(k, 0.5) * w for k, w in self._weights.items())

        return EvalScore(overall=overall, metrics=metrics, details=details)

    async def evaluate_with_content(
        self,
        results: list[SimulationResult],
        config: AutoresearchConfig,
    ) -> EvalScore:
        """Full evaluation including content quality from LLM judge.

        Call this instead of evaluate() when running with real backends
        and you want content-level scoring.
        """
        # Get infrastructure score first
        score = self.evaluate(results, config)

        # Run content judge per scenario
        content_scores: list[ContentScore] = []
        for result, scenario in zip(results, config.test_scenarios):
            cs = await self._judge.judge(scenario, result.turns)
            content_scores.append(cs)

        score.content_scores = content_scores

        if content_scores:
            # Aggregate content metrics
            avg_task = sum(c.task_completion for c in content_scores) / len(content_scores)
            avg_lang = sum(c.language_quality for c in content_scores) / len(content_scores)
            avg_fact = sum(c.factual_accuracy for c in content_scores) / len(content_scores)
            avg_nat = sum(c.naturalness for c in content_scores) / len(content_scores)
            avg_behavior = sum(c.behavior_rate for c in content_scores) / len(content_scores)

            score.metrics["task_completion"] = avg_task
            score.metrics["language_quality"] = avg_lang
            score.metrics["factual_accuracy"] = avg_fact
            score.metrics["content_naturalness"] = avg_nat
            score.metrics["behavior_coverage"] = avg_behavior

            score.details["task_completion"] = f"{avg_task:.3f}"
            score.details["language_quality"] = f"{avg_lang:.3f}"
            score.details["factual_accuracy"] = f"{avg_fact:.3f}"
            score.details["content_naturalness"] = f"{avg_nat:.3f}"
            score.details["behavior_coverage"] = f"{avg_behavior:.1%}"

            # Blend content score into overall (replaces simulation model quality)
            content_overall = sum(c.overall for c in content_scores) / len(content_scores)
            # Use content quality instead of parametric quality when available
            score.metrics["quality"] = content_overall
            score.details["quality"] = f"content={content_overall:.3f}"

            # Recalculate overall with content-aware quality
            weights = {
                "quality": 0.35,
                "latency": 0.25,
                "stability": 0.20,
                "backchannel": 0.10,
                "throughput": 0.10,
            }
            score.overall = sum(score.metrics.get(k, 0.5) * w for k, w in weights.items())

        return score
