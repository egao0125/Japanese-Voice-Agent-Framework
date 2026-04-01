"""Pipeline proposer — suggests config changes based on experiment history."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from jvaf.autoresearch.config import AutoresearchConfig
from jvaf.autoresearch.log import ExperimentLog
from jvaf.config import PipelineConfig
from jvaf.providers.registry import PROVIDER_DEFAULTS


@dataclass
class Proposal:
    """A proposed pipeline configuration change."""

    config: PipelineConfig
    hypothesis: str
    diff: dict


# Predefined threshold mutations
_THRESHOLD_MUTATIONS: list[tuple[str, str, dict]] = [
    (
        "Lower VAD threshold for better speech detection",
        "vad.threshold_db",
        {"field": "vad", "sub": "threshold_db", "values": [-30.0, -35.0, -40.0, -45.0]},
    ),
    (
        "Adjust silence threshold for turn-taking",
        "turn_taking.silence_threshold_sec",
        {"field": "turn_taking", "sub": "silence_threshold_sec", "values": [0.3, 0.4, 0.5, 0.6, 0.7]},
    ),
    (
        "Tune backchannel interval",
        "backchannel.min_interval_sec",
        {"field": "backchannel", "sub": "min_interval_sec", "values": [3.0, 4.0, 5.0, 6.0, 7.0]},
    ),
    (
        "Adjust barge-in character threshold",
        "barge_in.min_chars",
        {"field": "barge_in", "sub": "min_chars", "values": [3, 4, 5, 6]},
    ),
    (
        "Tune minimum speech before backchannel",
        "backchannel.min_speech_before_bc_sec",
        {"field": "backchannel", "sub": "min_speech_before_bc_sec", "values": [1.0, 1.5, 2.0, 2.5]},
    ),
    (
        "Adjust VAD minimum speech duration",
        "vad.min_speech_ms",
        {"field": "vad", "sub": "min_speech_ms", "values": [150, 200, 250, 300, 400]},
    ),
    (
        "Tune VAD minimum silence duration",
        "vad.min_silence_ms",
        {"field": "vad", "sub": "min_silence_ms", "values": [200, 300, 400, 500]},
    ),
    (
        "Toggle backchannel triggers",
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

# Provider-swap mutations (values come from available_providers at runtime)
_PROVIDER_SWAPS: list[tuple[str, str]] = [
    ("Try different STT provider", "stt"),
    ("Try different LLM provider", "llm"),
    ("Try different TTS provider", "tts"),
    ("Try different VAD provider", "vad"),
]


class PipelineProposer:
    """Proposes pipeline config changes for autoresearch.

    Mutations include both threshold tuning AND provider swaps.
    Provider swaps are constrained to what's actually available
    (SDKs installed + API keys set).

    Mock mode: cycles through predefined mutations.
    LLM mode: uses an LLM to propose changes based on history + goals.
    """

    def __init__(
        self,
        *,
        backend: str = "mock",
        available_providers: dict[str, list[str]] | None = None,
    ):
        self._backend = backend
        self._mutation_idx = 0
        self._available = available_providers or {
            "stt": ["mock"],
            "llm": ["mock"],
            "tts": ["mock"],
            "vad": ["energy"],
        }
        # Build combined mutation list
        self._mutations = self._build_mutations()

    def _build_mutations(self) -> list:
        """Combine threshold mutations + provider swaps into one list."""
        mutations = list(_THRESHOLD_MUTATIONS)

        for hypothesis, category in _PROVIDER_SWAPS:
            providers = self._available.get(category, [])
            if len(providers) > 1:
                mutations.append((
                    hypothesis,
                    f"{category}.provider",
                    {"field": category, "sub": "provider", "values": providers, "is_swap": True},
                ))

        return mutations

    def propose(
        self,
        base_config: PipelineConfig,
        program: AutoresearchConfig,
        log: ExperimentLog,
    ) -> Proposal:
        """Propose a single config change."""
        if self._backend == "mock":
            return self._propose_mock(base_config, log)
        return self._propose_llm(base_config, program, log)

    def _propose_mock(self, base_config: PipelineConfig, log: ExperimentLog) -> Proposal:
        """Deterministic mock proposer — cycles through mutations."""
        mutations = self._mutations
        mutation = mutations[self._mutation_idx % len(mutations)]
        self._mutation_idx += 1

        hypothesis, path, spec = mutation
        new_config = base_config.model_copy(deep=True)

        # Apply mutation
        section = getattr(new_config, spec["field"])
        current = getattr(section, spec["sub"])
        candidates = [v for v in spec["values"] if v != current]
        new_value = random.choice(candidates) if candidates else spec["values"][0]
        setattr(section, spec["sub"], new_value)

        diff: dict[str, Any] = {path: {"from": current, "to": new_value}}

        # On provider swap, also apply provider defaults
        if spec.get("is_swap"):
            defaults = PROVIDER_DEFAULTS.get(spec["field"], {}).get(new_value, {})
            for key, val in defaults.items():
                setattr(section, key, val)
                diff[f"{spec['field']}.{key}"] = {"from": "auto", "to": val}

        return Proposal(
            config=new_config,
            hypothesis=f"{hypothesis} ({current} → {new_value})",
            diff=diff,
        )

    def _propose_llm(
        self,
        base_config: PipelineConfig,
        program: AutoresearchConfig,
        log: ExperimentLog,
    ) -> Proposal:
        """LLM-driven proposer — placeholder for real LLM integration."""
        # TODO: Integrate with LLM provider for intelligent proposals
        # For now, fall back to mock
        return self._propose_mock(base_config, log)
