"""Tests for provider registry and provider-swap proposer."""

import pytest

from jvaf.autoresearch.config import AutoresearchConfig, _detect_providers
from jvaf.autoresearch.log import ExperimentLog
from jvaf.autoresearch.proposer import PipelineProposer
from jvaf.config import PipelineConfig
from jvaf.providers.registry import (
    PROVIDER_DEFAULTS,
    ProviderNotAvailable,
    available_providers,
    detect_all_available,
    get_class,
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
        # Mock/energy always available
        assert "mock" in result["stt"]
        assert "energy" in result["vad"]

    def test_provider_defaults_structure(self):
        assert "stt" in PROVIDER_DEFAULTS
        assert "mock" in PROVIDER_DEFAULTS["stt"]
        assert "deepgram" in PROVIDER_DEFAULTS["stt"]


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
        # Should still have mock/energy defaults
        assert result["stt"] == ["mock"]
        assert result["vad"] == ["energy"]

    def test_detect_multiple_providers(self):
        constraints = {
            "STT": "Deepgram or Whisper",
            "LLM": "Claude or GPT",
        }
        result = _detect_providers(constraints)
        assert "deepgram" in result["stt"]
        assert "openai" in result["stt"]  # whisper → openai
        assert "anthropic" in result["llm"]
        assert "openai" in result["llm"]  # gpt → openai

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


class TestProviderSwapProposer:
    def _make_log(self):
        from pathlib import Path
        log = ExperimentLog.__new__(ExperimentLog)
        log._entries = []
        log._path = Path("/dev/null")
        return log

    def test_proposer_with_multiple_providers(self):
        """Proposer should include provider swap mutations when multiple available."""
        proposer = PipelineProposer(
            backend="mock",
            available_providers={
                "stt": ["mock", "deepgram"],
                "llm": ["mock", "anthropic"],
                "tts": ["mock", "elevenlabs"],
                "vad": ["energy"],
            },
        )
        config = PipelineConfig()
        program = AutoresearchConfig()
        log = self._make_log()

        # Cycle through all mutations — should include provider swaps
        hypotheses = []
        for _ in range(20):
            p = proposer.propose(config, program, log)
            hypotheses.append(p.hypothesis)

        swap_proposals = [h for h in hypotheses if "provider" in h.lower()]
        assert len(swap_proposals) >= 2, f"Expected provider swap proposals, got: {hypotheses}"

    def test_proposer_no_swap_single_provider(self):
        """Proposer should NOT include swaps when only one provider per category."""
        proposer = PipelineProposer(
            backend="mock",
            available_providers={
                "stt": ["mock"],
                "llm": ["mock"],
                "tts": ["mock"],
                "vad": ["energy"],
            },
        )
        config = PipelineConfig()
        program = AutoresearchConfig()
        log = self._make_log()

        hypotheses = []
        for _ in range(15):
            p = proposer.propose(config, program, log)
            hypotheses.append(p.hypothesis)

        swap_proposals = [h for h in hypotheses if "provider" in h.lower()]
        assert len(swap_proposals) == 0

    def test_provider_swap_applies_defaults(self):
        """When swapping providers, defaults should be applied."""
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
        log = self._make_log()

        # Find the STT swap proposal
        for _ in range(20):
            p = proposer.propose(config, program, log)
            if "STT" in p.hypothesis and "deepgram" in p.hypothesis:
                assert p.config.stt.provider == "deepgram"
                assert p.config.stt.model == "nova-2"
                return

        pytest.fail("Never got STT provider swap proposal")
