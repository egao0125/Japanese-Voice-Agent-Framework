"""Tests for conversation simulator."""

import pytest

from jvaf.autoresearch.config import AutoresearchConfig, TestScenario
from jvaf.autoresearch.simulator import ConversationSimulator
from jvaf.config import PipelineConfig


@pytest.mark.asyncio
async def test_simulator_basic():
    config = PipelineConfig()
    scenario = TestScenario(
        name="test",
        description="Basic test",
        user_utterances=["こんにちは", "予約したいです"],
    )

    sim = ConversationSimulator()
    result = await sim.run_scenario(config, scenario)

    assert result.scenario_name == "test"
    assert result.turn_count == 2
    assert result.total_duration_ms > 0
    assert not result.pipeline_errors


@pytest.mark.asyncio
async def test_simulator_default_utterances():
    """run_all auto-generates utterances for scenarios without them."""
    config = PipelineConfig()
    scenario = TestScenario(name="auto", description="Auto-generated utterances")

    sim = ConversationSimulator()
    results = await sim.run_all(config, [scenario])

    # run_all adds default realistic Japanese utterances (3-4 per scenario)
    assert results[0].turn_count >= 3


@pytest.mark.asyncio
async def test_simulator_run_all():
    config = PipelineConfig()
    scenarios = [
        TestScenario(name="s1", description="Scenario 1", user_utterances=["テスト"]),
        TestScenario(name="s2", description="Scenario 2", user_utterances=["テスト"]),
    ]

    sim = ConversationSimulator()
    results = await sim.run_all(config, scenarios)

    assert len(results) == 2
    assert results[0].scenario_name == "s1"
    assert results[1].scenario_name == "s2"
