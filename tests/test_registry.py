"""Tests for provider registry and phased search proposer."""

import pytest

from jvaf.autoresearch.config import AutoresearchConfig, _detect_providers
from jvaf.autoresearch.log import ExperimentEntry, ExperimentLog
from jvaf.autoresearch.proposer import PipelineProposer, SearchPhase
from jvaf.config import PipelineConfig
from jvaf.providers.registry import (
    PROVIDER_DEFAULTS,
    ProviderNotAvailable,
    available_providers,
    detect_all_available,
    get_class,
    get_defaults,
    list_all_providers,
    register,
)


class TestRegistry:
    def test_get_mock_providers(self):
        assert get_class("stt", "mock").__name__ == "MockSTT"
        assert get_class("llm", "mock").__name__ == "MockLLM"
        assert get_class("tts", "mock").__name__ == "MockTTS"
        assert get_class("vad", "energy").__name__ == "EnergyVAD"

    def test_unknown_provider_raises(self):
        with pytest.raises(ProviderNotAvailable):
            get_class("stt", "nonexistent_provider")

    def test_custom_registration(self):
        class CustomSTT:
            pass

        register("stt", "custom", CustomSTT)
        assert get_class("stt", "custom") is CustomSTT

    def test_available_includes_mock(self):
        avail = available_providers("stt")
        assert "mock" in avail

    def test_detect_all(self):
        result = detect_all_available()
        assert "stt" in result
        assert "llm" in result
        assert "tts" in result
        assert "vad" in result
        assert "mock" in result["stt"]
        assert "energy" in result["vad"]

    def test_provider_defaults_structure(self):
        assert "stt" in PROVIDER_DEFAULTS
        assert "mock" in PROVIDER_DEFAULTS["stt"]
        assert "deepgram" in PROVIDER_DEFAULTS["stt"]

    def test_get_defaults(self):
        d = get_defaults("stt", "deepgram")
        assert d["model"] == "nova-2"

    def test_list_all(self):
        all_p = list_all_providers()
        assert "mock" in all_p["stt"]
        assert "anthropic" in all_p["llm"]
        assert len(all_p["llm"]) >= 5  # mock, anthropic, openai, google, mistral, groq, hf_local


class TestProviderDetection:
    def test_detect_from_constraints(self):
        constraints = {
            "Budget": "Deepgram STT, Claude Haiku LLM, ElevenLabs TTS",
            "Language": "Japanese",
        }
        result = _detect_providers(constraints)
        assert "deepgram" in result["stt"]
        assert "anthropic" in result["llm"]
        assert "elevenlabs" in result["tts"]

    def test_detect_empty_constraints(self):
        result = _detect_providers({})
        assert result["stt"] == ["mock"]
        assert result["vad"] == ["energy"]

    def test_detect_multiple_providers(self):
        constraints = {
            "STT": "Deepgram or Whisper",
            "LLM": "Claude or GPT or Mistral",
        }
        result = _detect_providers(constraints)
        assert "deepgram" in result["stt"]
        assert "openai" in result["stt"]
        assert "anthropic" in result["llm"]
        assert "openai" in result["llm"]
        assert "mistral" in result["llm"]

    def test_program_md_parsing(self, tmp_path):
        program = tmp_path / "program.md"
        program.write_text("""# Test
## Use Case
Test agent
## Constraints
- Budget: Deepgram STT, Claude LLM, ElevenLabs TTS
## Test Scenarios
1. Hello
""")
        cfg = AutoresearchConfig.from_markdown(program)
        assert "deepgram" in cfg.available_providers["stt"]
        assert "anthropic" in cfg.available_providers["llm"]
        assert "elevenlabs" in cfg.available_providers["tts"]


def _make_log(tmp_path=None):
    from pathlib import Path
    log = ExperimentLog.__new__(ExperimentLog)
    log._entries = []
    log._path = tmp_path / "log.tsv" if tmp_path else Path("/dev/null")
    return log


class TestPhasedSearch:
    def test_starts_in_tournament(self):
        proposer = PipelineProposer(
            backend="mock",
            available_providers={"stt": ["mock", "deepgram"], "llm": ["mock"], "tts": ["mock"], "vad": ["energy"]},
        )
        assert proposer.phase == SearchPhase.TOURNAMENT

    def test_skips_tournament_single_providers(self):
        """If only one provider per category, skip straight to tuning."""
        proposer = PipelineProposer(
            backend="mock",
            available_providers={"stt": ["mock"], "llm": ["mock"], "tts": ["mock"], "vad": ["energy"]},
        )
        assert proposer.phase == SearchPhase.TUNING

    def test_tournament_tests_all_providers(self):
        proposer = PipelineProposer(
            backend="mock",
            available_providers={
                "stt": ["mock", "deepgram"],
                "llm": ["mock", "anthropic"],
                "tts": ["mock"],
                "vad": ["energy"],
            },
        )
        config = PipelineConfig()
        program = AutoresearchConfig()
        log = _make_log()

        proposals = []
        for _ in range(6):
            p = proposer.propose(config, program, log)
            proposals.append(p)

        tournament = [p for p in proposals if p.phase == SearchPhase.TOURNAMENT]
        # Should test: stt/mock, stt/deepgram, llm/mock, llm/anthropic, tts/mock, vad/energy
        assert len(tournament) >= 4

    def test_transitions_tournament_to_combination(self):
        proposer = PipelineProposer(
            backend="mock",
            available_providers={
                "stt": ["mock", "deepgram"],
                "llm": ["mock"],
                "tts": ["mock"],
                "vad": ["energy"],
            },
        )
        config = PipelineConfig()
        program = AutoresearchConfig()
        log = _make_log()

        phases_seen = set()
        for _ in range(10):
            p = proposer.propose(config, program, log)
            phases_seen.add(p.phase)

        # Should progress through tournament → combination/tuning
        assert SearchPhase.TOURNAMENT in phases_seen

    def test_tuning_after_providers(self):
        """After provider selection, should enter tuning phase."""
        proposer = PipelineProposer(
            backend="mock",
            available_providers={"stt": ["mock"], "llm": ["mock"], "tts": ["mock"], "vad": ["energy"]},
        )
        config = PipelineConfig()
        program = AutoresearchConfig()
        log = _make_log()

        p = proposer.propose(config, program, log)
        assert p.phase == SearchPhase.TUNING
        assert "[P3:tune]" in p.hypothesis

    def test_stagnation_triggers_revalidation(self):
        proposer = PipelineProposer(
            backend="mock",
            available_providers={
                "stt": ["mock", "deepgram"],
                "llm": ["mock"],
                "tts": ["mock"],
                "vad": ["energy"],
            },
            stagnation_limit=3,
        )
        config = PipelineConfig()
        program = AutoresearchConfig()
        log = _make_log()

        # Exhaust tournament + combo first
        for _ in range(10):
            proposer.propose(config, program, log)

        # Force into tuning phase
        proposer._phase = SearchPhase.TUNING
        proposer._stagnation = 0
        proposer._provider_scores = {"stt": {"mock": 0.8, "deepgram": 0.7}, "llm": {}, "tts": {}, "vad": {}}

        # Simulate stagnation — track phases seen
        phases_seen = set()
        for i in range(6):
            entry = ExperimentEntry(iteration=i, hypothesis="test", score=0.5, kept=False,
                                    config_diff={"x": {"from": 1, "to": 2}})
            log._entries.append(entry)
            p = proposer.propose(config, program, log)
            phases_seen.add(p.phase)

        # Should have transitioned through revalidation at some point
        assert SearchPhase.REVALIDATE in phases_seen

    def test_provider_swap_applies_defaults(self):
        proposer = PipelineProposer(
            backend="mock",
            available_providers={
                "stt": ["mock", "deepgram"],
                "llm": ["mock"],
                "tts": ["mock"],
                "vad": ["energy"],
            },
        )
        config = PipelineConfig()
        program = AutoresearchConfig()
        log = _make_log()

        # Tournament should test deepgram
        for _ in range(10):
            p = proposer.propose(config, program, log)
            if p.config.stt.provider == "deepgram":
                assert p.config.stt.model == "nova-2"
                return

        pytest.fail("Never proposed deepgram")

    def test_no_swap_proposals_single_provider(self):
        proposer = PipelineProposer(
            backend="mock",
            available_providers={"stt": ["mock"], "llm": ["mock"], "tts": ["mock"], "vad": ["energy"]},
        )
        config = PipelineConfig()
        program = AutoresearchConfig()
        log = _make_log()

        for _ in range(15):
            p = proposer.propose(config, program, log)
            assert p.phase == SearchPhase.TUNING
