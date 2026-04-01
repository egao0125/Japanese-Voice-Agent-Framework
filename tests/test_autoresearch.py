"""End-to-end tests for the autoresearch loop."""

import tempfile
from pathlib import Path

import pytest

from jvaf.autoresearch import (
    AutoresearchConfig,
    AutoresearchLoop,
    ExperimentLog,
    PipelineEvaluator,
    PipelineProposer,
)
from jvaf.config import PipelineConfig


SAMPLE_PROGRAM = """\
# Test Agent

## Use Case
Test voice agent for unit testing.

## Goals
- Latency: <500ms
- Stability: no errors

## Constraints
- Language: Japanese
- Budget: mock providers only

## Test Scenarios
1. User says hello
2. User asks a question
"""


@pytest.fixture
def program_path(tmp_path):
    p = tmp_path / "program.md"
    p.write_text(SAMPLE_PROGRAM)
    return p


class TestAutoresearchConfig:
    def test_from_markdown(self, program_path):
        cfg = AutoresearchConfig.from_markdown(program_path)
        assert "Test voice agent" in cfg.use_case
        assert "Latency" in cfg.goals
        assert "Language" in cfg.constraints
        assert len(cfg.test_scenarios) == 2

    def test_summary(self, program_path):
        cfg = AutoresearchConfig.from_markdown(program_path)
        s = cfg.summary()
        assert "2 goals" in s
        assert "2 scenarios" in s


class TestExperimentLog:
    def test_append_and_read(self, tmp_path):
        from jvaf.autoresearch.log import ExperimentEntry

        log_path = tmp_path / "log.tsv"
        log = ExperimentLog(log_path)

        entry = ExperimentEntry(
            iteration=1,
            hypothesis="test hypothesis",
            config_diff={"vad.threshold_db": {"from": -35, "to": -40}},
            score=0.85,
            kept=True,
        )
        log.append(entry)

        # Re-read
        log2 = ExperimentLog(log_path)
        assert len(log2.entries) == 1
        assert log2.entries[0].hypothesis == "test hypothesis"
        assert log2.entries[0].kept is True
        assert log2.best_score() == 0.85


class TestPipelineProposer:
    def test_mock_proposer(self):
        proposer = PipelineProposer(backend="mock")
        config = PipelineConfig()
        from jvaf.autoresearch.config import AutoresearchConfig

        program = AutoresearchConfig()
        log = ExperimentLog.__new__(ExperimentLog)
        log._entries = []
        log._path = Path("/dev/null")

        proposal = proposer.propose(config, program, log)
        assert proposal.hypothesis
        assert proposal.diff
        assert isinstance(proposal.config, PipelineConfig)

    def test_proposer_cycles(self):
        """Mock proposer should cycle through different mutations."""
        proposer = PipelineProposer(backend="mock")
        config = PipelineConfig()
        program = AutoresearchConfig()
        log = ExperimentLog.__new__(ExperimentLog)
        log._entries = []
        log._path = Path("/dev/null")

        hypotheses = set()
        for _ in range(8):
            p = proposer.propose(config, program, log)
            hypotheses.add(p.hypothesis.split("(")[0].strip())

        assert len(hypotheses) >= 5  # At least 5 different mutation types


class TestPipelineEvaluator:
    def test_evaluate_empty(self):
        evaluator = PipelineEvaluator()
        config = AutoresearchConfig()
        score = evaluator.evaluate([], config)
        assert score.overall == 0.0


@pytest.mark.asyncio
async def test_autoresearch_end_to_end(program_path, tmp_path):
    """Full loop: parse program → run 5 iterations → check outputs."""
    program = AutoresearchConfig.from_markdown(program_path)
    output_dir = tmp_path / "experiments"

    loop = AutoresearchLoop(
        program,
        output_dir=str(output_dir),
        backend="mock",
    )
    summary = await loop.run(iterations=5)

    assert summary["total_iterations"] == 5
    assert summary["kept"] + summary["discarded"] == 5
    assert summary["best_score"] > 0

    # Check output files
    assert (output_dir / "log.tsv").exists()
    assert (output_dir / "best" / "config.yaml").exists()

    # Verify log has 5 entries
    log = ExperimentLog(output_dir / "log.tsv")
    assert len(log.entries) == 5
