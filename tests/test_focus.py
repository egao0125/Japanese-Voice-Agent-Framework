"""Tests for improvement focus feature — focused optimization of specific pipeline areas."""

from pathlib import Path

import pytest

from jvaf.autoresearch.config import (
    AutoresearchConfig,
    TestScenario,
    detect_focus_params,
)
from jvaf.autoresearch.evaluator import PipelineEvaluator
from jvaf.autoresearch.log import ExperimentLog
from jvaf.autoresearch.proposer import PipelineProposer, SearchPhase
from jvaf.config import PipelineConfig


# -- Focus parsing ----------------------------------------------------------


PROGRAM_WITH_FOCUS = """\
# Improve My Agent

## Use Case
Dental clinic receptionist.

## Goals
- Latency: <400ms
- Quality: polite keigo

## Improve
- Reduce turn-taking latency
- Better backchannel timing

## Test Scenarios
1. Patient calls to book
   - Should greet politely
"""


class TestFocusParsing:
    def test_detect_focus_from_markdown(self, tmp_path):
        p = tmp_path / "program.md"
        p.write_text(PROGRAM_WITH_FOCUS)
        cfg = AutoresearchConfig.from_markdown(p)

        assert len(cfg.improvement_focus) == 2
        assert "Reduce turn-taking latency" in cfg.improvement_focus
        assert "Better backchannel timing" in cfg.improvement_focus

    def test_detect_focus_params_latency(self):
        params = detect_focus_params(["Reduce turn-taking latency"])
        assert "turn_taking.silence_threshold_sec" in params
        assert "vad.threshold_db" in params

    def test_detect_focus_params_backchannel(self):
        params = detect_focus_params(["Better backchannel timing"])
        assert "backchannel.min_interval_sec" in params
        assert "backchannel.min_speech_before_bc_sec" in params
        assert "backchannel.triggers" in params

    def test_detect_focus_params_stt(self):
        params = detect_focus_params(["Improve STT accuracy"])
        assert "provider_stt" in params

    def test_detect_focus_params_multiple(self):
        params = detect_focus_params(["Improve latency", "Better voice naturalness"])
        assert "turn_taking.silence_threshold_sec" in params
        assert "provider_tts" in params

    def test_detect_focus_params_empty(self):
        params = detect_focus_params([])
        assert len(params) == 0

    def test_focus_section_aliases(self, tmp_path):
        """Should parse ## Focus as well as ## Improve."""
        md = """\
# Agent

## Use Case
Test.

## Focus
- Latency
"""
        p = tmp_path / "program.md"
        p.write_text(md)
        cfg = AutoresearchConfig.from_markdown(p)
        assert len(cfg.improvement_focus) == 1
        assert "Latency" in cfg.improvement_focus


# -- Focused proposer -------------------------------------------------------


def _make_log() -> ExperimentLog:
    log = ExperimentLog.__new__(ExperimentLog)
    log._entries = []
    log._path = Path("/dev/null")
    return log


class TestFocusedProposer:
    def test_focus_prioritizes_mutations(self):
        """Focused mutations should appear first in tuning phase."""
        focus = {"backchannel.min_interval_sec", "backchannel.triggers"}
        proposer = PipelineProposer(backend="mock", focus_params=focus)

        # Force into tuning phase (mock has single provider per category)
        assert proposer.phase == SearchPhase.TUNING

        config = PipelineConfig()
        program = AutoresearchConfig()
        log = _make_log()

        # First proposals should be backchannel-related
        p1 = proposer.propose(config, program, log)
        assert "backchannel" in p1.hypothesis.lower() or any(
            "backchannel" in k for k in p1.diff
        )

    def test_no_focus_uses_default_order(self):
        """Without focus, mutations follow default order."""
        proposer = PipelineProposer(backend="mock")
        config = PipelineConfig()
        program = AutoresearchConfig()
        log = _make_log()

        # Should start with first default mutation (VAD threshold)
        p = proposer.propose(config, program, log)
        assert p.hypothesis  # Just verify it works

    def test_focus_provider_tournament_priority(self):
        """Focus on STT should prioritize STT category in tournament."""
        focus = {"provider_stt"}
        proposer = PipelineProposer(
            backend="mock",
            available_providers={
                "stt": ["mock", "deepgram"],
                "llm": ["mock", "openai"],
                "tts": ["mock"],
                "vad": ["energy"],
            },
            focus_params=focus,
        )

        assert proposer.phase == SearchPhase.TOURNAMENT

        config = PipelineConfig()
        program = AutoresearchConfig()
        log = _make_log()

        # First tournament proposal should be STT-related
        p = proposer.propose(config, program, log)
        assert "stt" in p.hypothesis.lower()


# -- Focused evaluator weights ----------------------------------------------


class TestFocusedEvaluator:
    def test_default_weights_sum_to_one(self):
        evaluator = PipelineEvaluator()
        weights = evaluator._weights
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_focus_boosts_latency_weight(self):
        focus = {"turn_taking.silence_threshold_sec"}
        evaluator = PipelineEvaluator(focus_params=focus)
        default = PipelineEvaluator()

        assert evaluator._weights["latency"] > default._weights["latency"]
        assert abs(sum(evaluator._weights.values()) - 1.0) < 1e-6

    def test_focus_boosts_backchannel_weight(self):
        focus = {"backchannel.triggers"}
        evaluator = PipelineEvaluator(focus_params=focus)
        default = PipelineEvaluator()

        assert evaluator._weights["backchannel"] > default._weights["backchannel"]
        assert abs(sum(evaluator._weights.values()) - 1.0) < 1e-6

    def test_focus_boosts_quality_weight(self):
        focus = {"provider_stt"}
        evaluator = PipelineEvaluator(focus_params=focus)
        default = PipelineEvaluator()

        assert evaluator._weights["quality"] > default._weights["quality"]

    def test_no_focus_uses_default_weights(self):
        evaluator = PipelineEvaluator()
        expected = {"quality": 0.35, "latency": 0.25, "stability": 0.20, "backchannel": 0.10, "throughput": 0.10}
        for k, v in expected.items():
            assert abs(evaluator._weights[k] - v) < 1e-6


# -- End-to-end focused autoresearch ----------------------------------------


@pytest.mark.asyncio
async def test_focused_autoresearch_e2e(tmp_path):
    """Full loop with improvement focus should prioritize focus areas."""
    from jvaf.autoresearch import AutoresearchLoop

    p = tmp_path / "program.md"
    p.write_text(PROGRAM_WITH_FOCUS)
    program = AutoresearchConfig.from_markdown(p)
    output_dir = tmp_path / "experiments"

    loop = AutoresearchLoop(
        program,
        output_dir=str(output_dir),
        backend="mock",
    )

    # Verify focus was wired through
    assert loop._focus_params
    assert "turn_taking.silence_threshold_sec" in loop._focus_params

    summary = await loop.run(iterations=5)
    assert summary["total_iterations"] == 5
    assert summary["best_score"] > 0

    # Check output files exist
    assert (output_dir / "log.tsv").exists()
    assert (output_dir / "best" / "config.yaml").exists()
