"""Tests for content judge and expected behaviors parsing."""

import pytest

from jvaf.autoresearch.config import AutoresearchConfig, TestScenario
from jvaf.autoresearch.judge import (
    ContentJudge,
    ContentScore,
    _build_judge_prompt,
    parse_judge_response,
)
from jvaf.autoresearch.simulator import TurnRecord


class TestExpectedBehaviors:
    def test_parse_behaviors_from_markdown(self, tmp_path):
        program = tmp_path / "program.md"
        program.write_text("""# Test Agent
## Use Case
Test agent
## Test Scenarios
1. Patient books appointment
   - Should greet politely in keigo
   - Should ask for preferred date
   - Should confirm booking
2. Patient cancels
   - Should verify identity
""")
        cfg = AutoresearchConfig.from_markdown(program)
        assert len(cfg.test_scenarios) == 2
        assert len(cfg.test_scenarios[0].expected_behaviors) == 3
        assert "greet politely" in cfg.test_scenarios[0].expected_behaviors[0]
        assert len(cfg.test_scenarios[1].expected_behaviors) == 1
        assert "verify identity" in cfg.test_scenarios[1].expected_behaviors[0]

    def test_parse_no_behaviors(self, tmp_path):
        program = tmp_path / "program.md"
        program.write_text("""# Test
## Test Scenarios
1. Basic test
2. Another test
""")
        cfg = AutoresearchConfig.from_markdown(program)
        assert len(cfg.test_scenarios) == 2
        assert cfg.test_scenarios[0].expected_behaviors == []
        assert cfg.test_scenarios[1].expected_behaviors == []


class TestContentScore:
    def test_overall_weighted(self):
        score = ContentScore(
            task_completion=1.0,
            language_quality=1.0,
            factual_accuracy=1.0,
            naturalness=1.0,
        )
        assert score.overall == 1.0

    def test_overall_partial(self):
        score = ContentScore(
            task_completion=0.5,
            language_quality=0.5,
            factual_accuracy=0.5,
            naturalness=0.5,
        )
        assert score.overall == 0.5

    def test_behavior_rate_all_pass(self):
        score = ContentScore(
            behavior_scores={"greet": True, "ask date": True, "confirm": True}
        )
        assert score.behavior_rate == 1.0

    def test_behavior_rate_partial(self):
        score = ContentScore(
            behavior_scores={"greet": True, "ask date": False, "confirm": False}
        )
        assert abs(score.behavior_rate - 1 / 3) < 0.01

    def test_behavior_rate_empty(self):
        score = ContentScore()
        assert score.behavior_rate == 0.5  # Neutral when no behaviors


class TestParseJudgeResponse:
    def test_valid_json(self):
        response = '{"task_completion": 0.8, "language_quality": 0.9, "factual_accuracy": 0.7, "naturalness": 0.85, "behaviors": {"greet": true}, "reasoning": "Good"}'
        score = parse_judge_response(response)
        assert score.task_completion == 0.8
        assert score.language_quality == 0.9
        assert score.factual_accuracy == 0.7
        assert score.naturalness == 0.85
        assert score.behavior_scores["greet"] is True
        assert score.reasoning == "Good"

    def test_json_in_markdown(self):
        response = """Here is my evaluation:
```json
{"task_completion": 0.6, "language_quality": 0.7, "factual_accuracy": 0.8, "naturalness": 0.5, "behaviors": {}, "reasoning": "OK"}
```"""
        score = parse_judge_response(response)
        assert score.task_completion == 0.6

    def test_invalid_json(self):
        score = parse_judge_response("not json at all")
        assert score.task_completion == 0.0
        assert "Failed to parse" in score.reasoning

    def test_clamps_values(self):
        response = '{"task_completion": 1.5, "language_quality": -0.3, "factual_accuracy": 0.5, "naturalness": 0.5}'
        score = parse_judge_response(response)
        assert score.task_completion == 1.0
        assert score.language_quality == 0.0


class TestBuildJudgePrompt:
    def test_includes_scenario(self):
        scenario = TestScenario(name="test", description="Booking appointment")
        turns = [TurnRecord(user_text="予約したい", agent_text="承知しました")]
        prompt = _build_judge_prompt(scenario, turns)
        assert "Booking appointment" in prompt
        assert "予約したい" in prompt
        assert "承知しました" in prompt

    def test_includes_behaviors(self):
        scenario = TestScenario(
            name="test",
            description="Test",
            expected_behaviors=["Should greet", "Should confirm"],
        )
        prompt = _build_judge_prompt(scenario, [])
        assert "Should greet" in prompt
        assert "Should confirm" in prompt

    def test_includes_use_case(self):
        scenario = TestScenario(name="test", description="Test")
        prompt = _build_judge_prompt(scenario, [], use_case="Dental clinic")
        assert "Dental clinic" in prompt


class TestMockJudge:
    @pytest.mark.asyncio
    async def test_mock_returns_structural_score(self):
        judge = ContentJudge(backend="mock")
        scenario = TestScenario(
            name="test",
            description="Test",
            expected_behaviors=["Should greet", "Should confirm"],
        )
        turns = [
            TurnRecord(user_text="こんにちは", agent_text="[simulated response]"),
        ]
        score = await judge.judge(scenario, turns)
        # Mock with bracket-prefixed response → base score ~0.3
        assert score.task_completion > 0
        assert score.reasoning == "Mock backend — structural scoring only"
        # Behaviors should all be False for mock
        assert score.behavior_scores["Should greet"] is False

    @pytest.mark.asyncio
    async def test_mock_with_real_looking_response(self):
        judge = ContentJudge(backend="mock")
        scenario = TestScenario(name="test", description="Test")
        turns = [
            TurnRecord(user_text="こんにちは", agent_text="はい、お電話ありがとうございます"),
        ]
        score = await judge.judge(scenario, turns)
        # Real-looking response → higher base score
        assert score.task_completion >= 0.6

    @pytest.mark.asyncio
    async def test_mock_empty_turns(self):
        judge = ContentJudge(backend="mock")
        scenario = TestScenario(name="test", description="Test")
        score = await judge.judge(scenario, [])
        assert score.task_completion == 0.0
