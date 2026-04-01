"""Pipeline proposer — phased search strategy with backtracking.

Search phases:
  1. TOURNAMENT  — test each provider per category, rank them
  2. COMBINATION — try top-N provider combos end-to-end
  3. TUNING      — optimize thresholds within the chosen stack
  4. REVALIDATE  — re-test runner-ups with tuned params, backtrack if better

Phases are not linear — revalidation can drop back to combination,
and tuning stagnation triggers revalidation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from jvaf.autoresearch.config import AutoresearchConfig
from jvaf.autoresearch.log import ExperimentLog
from jvaf.config import PipelineConfig
from jvaf.providers.registry import PROVIDER_DEFAULTS, get_defaults


class SearchPhase(str, Enum):
    TOURNAMENT = "tournament"
    COMBINATION = "combination"
    TUNING = "tuning"
    REVALIDATE = "revalidate"


@dataclass
class Proposal:
    """A proposed pipeline configuration change."""

    config: PipelineConfig
    hypothesis: str
    diff: dict
    phase: str = ""


# Threshold mutations for Phase 3
_THRESHOLD_MUTATIONS: list[tuple[str, str, dict]] = [
    (
        "Lower VAD threshold",
        "vad.threshold_db",
        {"field": "vad", "sub": "threshold_db", "values": [-30.0, -35.0, -40.0, -45.0]},
    ),
    (
        "Adjust silence threshold",
        "turn_taking.silence_threshold_sec",
        {"field": "turn_taking", "sub": "silence_threshold_sec", "values": [0.3, 0.4, 0.5, 0.6, 0.7]},
    ),
    (
        "Tune backchannel interval",
        "backchannel.min_interval_sec",
        {"field": "backchannel", "sub": "min_interval_sec", "values": [3.0, 4.0, 5.0, 6.0, 7.0]},
    ),
    (
        "Adjust barge-in chars",
        "barge_in.min_chars",
        {"field": "barge_in", "sub": "min_chars", "values": [3, 4, 5, 6]},
    ),
    (
        "Tune min speech before BC",
        "backchannel.min_speech_before_bc_sec",
        {"field": "backchannel", "sub": "min_speech_before_bc_sec", "values": [1.0, 1.5, 2.0, 2.5]},
    ),
    (
        "Adjust VAD min speech duration",
        "vad.min_speech_ms",
        {"field": "vad", "sub": "min_speech_ms", "values": [150, 200, 250, 300, 400]},
    ),
    (
        "Tune VAD min silence duration",
        "vad.min_silence_ms",
        {"field": "vad", "sub": "min_silence_ms", "values": [200, 300, 400, 500]},
    ),
    (
        "Toggle BC triggers",
        "backchannel.triggers",
        {
            "field": "backchannel",
            "sub": "triggers",
            "values": [
                ["reactive", "proactive", "energy"],
                ["reactive", "proactive"],
                ["reactive", "energy"],
                ["reactive"],
            ],
        },
    ),
    (
        "Adjust LLM temperature",
        "llm.temperature",
        {"field": "llm", "sub": "temperature", "values": [0.3, 0.5, 0.7, 0.9, 1.0]},
    ),
]

CATEGORIES = ["stt", "llm", "tts", "vad"]
CATEGORY_TO_FIELD = {"stt": "stt", "llm": "llm", "tts": "tts", "vad": "vad"}


class PipelineProposer:
    """Phased search strategy with backtracking.

    Phase 1 (Tournament): Test each available provider per category.
      Holds other categories at current best. Builds a ranking.

    Phase 2 (Combination): Take top-2 per category, test promising
      combos. Picks the best full stack.

    Phase 3 (Tuning): Hill-climb on thresholds within the chosen stack.
      After stagnation_limit misses, moves to Phase 4.

    Phase 4 (Revalidate): Re-tests runner-up providers with tuned
      params. If a runner-up wins → back to Phase 2. Otherwise done
      (restarts Phase 3 for another tuning pass).
    """

    def __init__(
        self,
        *,
        backend: str = "mock",
        available_providers: dict[str, list[str]] | None = None,
        stagnation_limit: int = 5,
    ):
        self._backend = backend
        self._available = available_providers or {
            "stt": ["mock"],
            "llm": ["mock"],
            "tts": ["mock"],
            "vad": ["energy"],
        }
        self._stagnation_limit = stagnation_limit

        # Phase state
        self._phase = SearchPhase.TOURNAMENT
        self._tournament_queue: list[tuple[str, str]] = []
        self._provider_scores: dict[str, dict[str, float]] = {c: {} for c in CATEGORIES}
        self._combo_queue: list[dict[str, str]] = []
        self._tuning_idx = 0
        self._stagnation = 0
        self._revalidation_queue: list[tuple[str, str]] = []
        self._revalidation_baseline = 0.0
        self._revalidation_found_better = False

        # Build tournament queue
        self._init_tournament()

    def _init_tournament(self) -> None:
        """Queue up all provider tests for Phase 1."""
        self._tournament_queue = []
        for cat in CATEGORIES:
            providers = self._available.get(cat, [])
            for provider in providers:
                self._tournament_queue.append((cat, provider))
        # Skip tournament if only one provider per category
        if all(len(self._available.get(c, [])) <= 1 for c in CATEGORIES):
            self._phase = SearchPhase.TUNING

    @property
    def phase(self) -> SearchPhase:
        return self._phase

    def propose(
        self,
        base_config: PipelineConfig,
        program: AutoresearchConfig,
        log: ExperimentLog,
    ) -> Proposal:
        """Propose based on current search phase."""
        # Update state from log (scores from last iteration)
        self._ingest_last_result(log)

        if self._phase == SearchPhase.TOURNAMENT:
            return self._propose_tournament(base_config)
        elif self._phase == SearchPhase.COMBINATION:
            return self._propose_combination(base_config)
        elif self._phase == SearchPhase.TUNING:
            return self._propose_tuning(base_config)
        elif self._phase == SearchPhase.REVALIDATE:
            return self._propose_revalidation(base_config)
        # Fallback
        return self._propose_tuning(base_config)

    def _ingest_last_result(self, log: ExperimentLog) -> None:
        """Read the last experiment result to update phase state."""
        if not log.entries:
            return
        last = log.entries[-1]

        if self._phase == SearchPhase.TOURNAMENT:
            # Record score for the provider that was just tested
            for path, change in last.config_diff.items():
                if path.endswith(".provider"):
                    cat = path.split(".")[0]
                    provider = change.get("to", "")
                    if provider:
                        self._provider_scores[cat][provider] = last.score

        elif self._phase == SearchPhase.TUNING:
            if last.kept:
                self._stagnation = 0
            else:
                self._stagnation += 1
                if self._stagnation >= self._stagnation_limit:
                    self._transition_to_revalidation(log)

        elif self._phase == SearchPhase.REVALIDATE:
            if last.kept:
                self._revalidation_found_better = True

    # ------------------------------------------------------------------
    # Phase 1: Tournament
    # ------------------------------------------------------------------

    def _propose_tournament(self, base_config: PipelineConfig) -> Proposal:
        """Test next provider in the tournament queue."""
        if not self._tournament_queue:
            self._transition_to_combination(base_config)
            return self._propose_combination(base_config)

        cat, provider = self._tournament_queue.pop(0)
        new_config = base_config.model_copy(deep=True)
        section = getattr(new_config, cat)
        old_provider = section.provider
        section.provider = provider

        # Apply provider defaults
        defaults = get_defaults(cat, provider)
        for key, val in defaults.items():
            if hasattr(section, key):
                setattr(section, key, val)

        diff = {f"{cat}.provider": {"from": old_provider, "to": provider}}

        remaining = len(self._tournament_queue)
        return Proposal(
            config=new_config,
            hypothesis=f"[P1:tournament] Test {cat}={provider} ({remaining} remaining)",
            diff=diff,
            phase=SearchPhase.TOURNAMENT,
        )

    # ------------------------------------------------------------------
    # Phase 2: Combination
    # ------------------------------------------------------------------

    def _transition_to_combination(self, base_config: PipelineConfig) -> None:
        """Build combo queue from tournament rankings."""
        self._phase = SearchPhase.COMBINATION

        # Get top-2 per category
        top: dict[str, list[str]] = {}
        for cat in CATEGORIES:
            scores = self._provider_scores.get(cat, {})
            if not scores:
                # No tournament data — use whatever's available
                top[cat] = self._available.get(cat, ["mock"])[:2]
            else:
                ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                top[cat] = [name for name, _ in ranked[:2]]

        # Generate combos: best-of-each + swap one runner-up at a time
        best_combo = {cat: providers[0] for cat, providers in top.items()}
        self._combo_queue = [best_combo]

        for cat in CATEGORIES:
            if len(top[cat]) > 1:
                variant = dict(best_combo)
                variant[cat] = top[cat][1]
                self._combo_queue.append(variant)

        # Remove combo that matches current config (already tested)
        current = {cat: getattr(base_config, cat).provider for cat in CATEGORIES}
        self._combo_queue = [c for c in self._combo_queue if c != current]

    def _propose_combination(self, base_config: PipelineConfig) -> Proposal:
        """Test next provider combination."""
        if not self._combo_queue:
            self._phase = SearchPhase.TUNING
            self._stagnation = 0
            self._tuning_idx = 0
            return self._propose_tuning(base_config)

        combo = self._combo_queue.pop(0)
        new_config = base_config.model_copy(deep=True)
        diff: dict[str, Any] = {}

        for cat, provider in combo.items():
            section = getattr(new_config, cat)
            old = section.provider
            if old != provider:
                section.provider = provider
                defaults = get_defaults(cat, provider)
                for key, val in defaults.items():
                    if hasattr(section, key):
                        setattr(section, key, val)
                diff[f"{cat}.provider"] = {"from": old, "to": provider}

        combo_str = " ".join(f"{c}={p}" for c, p in combo.items() if diff.get(f"{c}.provider"))
        remaining = len(self._combo_queue)
        return Proposal(
            config=new_config,
            hypothesis=f"[P2:combo] {combo_str or 'baseline'} ({remaining} remaining)",
            diff=diff,
            phase=SearchPhase.COMBINATION,
        )

    # ------------------------------------------------------------------
    # Phase 3: Tuning
    # ------------------------------------------------------------------

    def _propose_tuning(self, base_config: PipelineConfig) -> Proposal:
        """Cycle through threshold mutations."""
        mutation = _THRESHOLD_MUTATIONS[self._tuning_idx % len(_THRESHOLD_MUTATIONS)]
        self._tuning_idx += 1

        hypothesis, path, spec = mutation
        new_config = base_config.model_copy(deep=True)

        section = getattr(new_config, spec["field"])
        current = getattr(section, spec["sub"])
        candidates = [v for v in spec["values"] if v != current]
        new_value = random.choice(candidates) if candidates else spec["values"][0]
        setattr(section, spec["sub"], new_value)

        diff = {path: {"from": current, "to": new_value}}

        return Proposal(
            config=new_config,
            hypothesis=f"[P3:tune] {hypothesis} ({current} → {new_value})",
            diff=diff,
            phase=SearchPhase.TUNING,
        )

    # ------------------------------------------------------------------
    # Phase 4: Revalidation (can backtrack)
    # ------------------------------------------------------------------

    def _transition_to_revalidation(self, log: ExperimentLog) -> None:
        """Build revalidation queue from runner-up providers."""
        self._phase = SearchPhase.REVALIDATE
        self._revalidation_found_better = False
        self._revalidation_baseline = log.best_score()
        self._revalidation_queue = []

        # For each category, find runner-ups we haven't tried recently
        for cat in CATEGORIES:
            scores = self._provider_scores.get(cat, {})
            if len(scores) < 2:
                continue
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            # Add runner-ups (skip the current best)
            for name, _ in ranked[1:]:
                self._revalidation_queue.append((cat, name))

    def _propose_revalidation(self, base_config: PipelineConfig) -> Proposal:
        """Re-test runner-up providers with current tuned parameters."""
        if not self._revalidation_queue:
            if self._revalidation_found_better:
                # A runner-up won — go back to Phase 2 with updated rankings
                self._transition_to_combination(base_config)
                return self._propose_combination(base_config)
            else:
                # No improvement — back to tuning with reset stagnation
                self._phase = SearchPhase.TUNING
                self._stagnation = 0
                return self._propose_tuning(base_config)

        cat, provider = self._revalidation_queue.pop(0)
        new_config = base_config.model_copy(deep=True)
        section = getattr(new_config, cat)
        old_provider = section.provider
        section.provider = provider

        defaults = get_defaults(cat, provider)
        for key, val in defaults.items():
            if hasattr(section, key):
                setattr(section, key, val)

        diff = {f"{cat}.provider": {"from": old_provider, "to": provider}}
        remaining = len(self._revalidation_queue)

        return Proposal(
            config=new_config,
            hypothesis=f"[P4:revalidate] Re-test {cat}={provider} with tuned params ({remaining} remaining)",
            diff=diff,
            phase=SearchPhase.REVALIDATE,
        )
