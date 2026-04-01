"""Pipeline evaluator — scores simulation results against program goals."""

from __future__ import annotations

from dataclasses import dataclass, field

from jvaf.autoresearch.config import AutoresearchConfig
from jvaf.autoresearch.simulator import SimulationResult


@dataclass
class EvalScore:
    """Evaluation result with overall score and per-metric breakdown."""

    overall: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)
    details: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        parts = [f"overall={self.overall:.3f}"]
        for k, v in self.metrics.items():
            parts.append(f"{k}={v:.3f}")
        return " | ".join(parts)


class PipelineEvaluator:
    """Evaluates pipeline quality against program goals.

    Metrics:
    - latency: average turn latency
    - throughput: turns processed without error
    - stability: error rate
    - backchannel_rate: BCs per turn (if enabled)
    - turn_count: conversation completeness
    """

    def evaluate(
        self, results: list[SimulationResult], config: AutoresearchConfig
    ) -> EvalScore:
        """Score a set of simulation results against program goals."""
        if not results:
            return EvalScore(overall=0.0)

        metrics: dict[str, float] = {}
        details: dict[str, str] = {}

        # Latency score (lower is better, target: <500ms avg)
        avg_latencies = [r.avg_latency_ms for r in results if r.avg_latency_ms > 0]
        if avg_latencies:
            avg_lat = sum(avg_latencies) / len(avg_latencies)
            # Score: 1.0 at 0ms, 0.5 at 500ms, 0.0 at 2000ms
            lat_score = max(0.0, 1.0 - avg_lat / 2000.0)
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

        # Throughput: total turns processed
        total_turns = sum(r.turn_count for r in results)
        expected_turns = sum(len(s.user_utterances) for s in config.test_scenarios) or total_turns
        throughput = min(1.0, total_turns / expected_turns) if expected_turns else 1.0
        metrics["throughput"] = throughput
        details["throughput"] = f"{total_turns} turns"

        # Output quality: did the pipeline produce output frames?
        total_output = sum(r.output_frame_count for r in results)
        has_output = 1.0 if total_output > 0 else 0.0
        metrics["output"] = has_output
        details["output"] = f"{total_output} frames"

        # Backchannel rate (if enabled in goals)
        bc_total = sum(r.backchannel_total for r in results)
        if total_turns > 0:
            bc_rate = bc_total / total_turns
            # Score: 0.3-0.7 BCs per turn is ideal
            if 0.3 <= bc_rate <= 0.7:
                bc_score = 1.0
            elif bc_rate < 0.3:
                bc_score = bc_rate / 0.3
            else:
                bc_score = max(0.0, 1.0 - (bc_rate - 0.7) / 1.0)
            metrics["backchannel"] = bc_score
            details["backchannel"] = f"{bc_total} total, {bc_rate:.2f}/turn"

        # Overall: weighted average
        weights = {
            "latency": 0.25,
            "stability": 0.30,
            "throughput": 0.20,
            "output": 0.15,
            "backchannel": 0.10,
        }
        overall = sum(metrics.get(k, 0.5) * w for k, w in weights.items())

        return EvalScore(overall=overall, metrics=metrics, details=details)
