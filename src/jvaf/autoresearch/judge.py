"""Content judge — LLM-as-judge for conversation quality evaluation.

Scores agent responses against expected behaviors defined in TestScenario.
Used with real backends; skipped for mock backend (parametric model used instead).

Evaluation dimensions:
  - task_completion: Did the agent make progress toward the scenario goal?
  - language_quality: Appropriate keigo, natural Japanese, correct formality?
  - factual_accuracy: No hallucinated information, no fabricated details?
  - naturalness: Does the conversation flow naturally? Appropriate turn-taking?
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from jvaf.autoresearch.config import TestScenario
from jvaf.autoresearch.simulator import TurnRecord


@dataclass
class ContentScore:
    """Content quality score from LLM judge."""

    task_completion: float = 0.0      # 0-1: did agent achieve scenario goal?
    language_quality: float = 0.0     # 0-1: keigo, formality, grammar
    factual_accuracy: float = 0.0     # 0-1: no hallucination, correct info
    naturalness: float = 0.0          # 0-1: conversation flow, turn-taking
    behavior_scores: dict[str, bool] = field(default_factory=dict)  # per-behavior pass/fail
    reasoning: str = ""               # judge's explanation

    @property
    def overall(self) -> float:
        """Weighted overall content quality score."""
        return (
            self.task_completion * 0.30
            + self.language_quality * 0.25
            + self.factual_accuracy * 0.25
            + self.naturalness * 0.20
        )

    @property
    def behavior_rate(self) -> float:
        """Fraction of expected behaviors observed."""
        if not self.behavior_scores:
            return 0.5  # No behaviors defined → neutral
        passed = sum(1 for v in self.behavior_scores.values() if v)
        return passed / len(self.behavior_scores)


_JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator for Japanese voice agent conversations.
You will be given a test scenario, expected behaviors, and the actual conversation.
Score the agent's performance on four dimensions (0.0 to 1.0 each).

IMPORTANT: Be strict but fair. A mock/placeholder response should score near 0.
A good but imperfect response should score 0.6-0.8. Only truly excellent responses get 0.9+.

Respond ONLY with valid JSON in this exact format:
{
  "task_completion": 0.0,
  "language_quality": 0.0,
  "factual_accuracy": 0.0,
  "naturalness": 0.0,
  "behaviors": {"behavior text": true/false, ...},
  "reasoning": "Brief explanation"
}"""


def _build_judge_prompt(
    scenario: TestScenario,
    turns: list[TurnRecord],
    use_case: str = "",
) -> str:
    """Build the evaluation prompt for the LLM judge."""
    lines = []
    lines.append(f"## Scenario: {scenario.description}")
    if use_case:
        lines.append(f"## Context: {use_case}")

    if scenario.expected_behaviors:
        lines.append("\n## Expected Behaviors:")
        for b in scenario.expected_behaviors:
            lines.append(f"- {b}")

    lines.append("\n## Conversation:")
    for i, turn in enumerate(turns, 1):
        lines.append(f"Turn {i}:")
        lines.append(f"  User: {turn.user_text}")
        lines.append(f"  Agent: {turn.agent_text}")
        if turn.backchannel_count > 0:
            lines.append(f"  (backchannels during user speech: {turn.backchannel_count})")

    lines.append("\n## Instructions:")
    lines.append("Score each dimension from 0.0 to 1.0:")
    lines.append("- task_completion: Did the agent make progress toward the scenario goal?")
    lines.append("- language_quality: Appropriate keigo, natural Japanese, correct formality?")
    lines.append("- factual_accuracy: No hallucinated information, no fabricated details?")
    lines.append("- naturalness: Does the conversation flow naturally?")
    if scenario.expected_behaviors:
        lines.append("- behaviors: For each expected behavior, was it observed? (true/false)")

    return "\n".join(lines)


def parse_judge_response(response: str) -> ContentScore:
    """Parse JSON response from the LLM judge into ContentScore."""
    # Try to extract JSON from the response (handle markdown code blocks)
    json_match = re.search(r"\{[\s\S]*\}", response)
    if not json_match:
        return ContentScore(reasoning="Failed to parse judge response")

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return ContentScore(reasoning="Invalid JSON from judge")

    return ContentScore(
        task_completion=_clamp(data.get("task_completion", 0.0)),
        language_quality=_clamp(data.get("language_quality", 0.0)),
        factual_accuracy=_clamp(data.get("factual_accuracy", 0.0)),
        naturalness=_clamp(data.get("naturalness", 0.0)),
        behavior_scores={
            str(k): bool(v) for k, v in data.get("behaviors", {}).items()
        },
        reasoning=data.get("reasoning", ""),
    )


class ContentJudge:
    """LLM-as-judge for conversation quality.

    Uses whatever LLM backend is available to evaluate agent responses
    against expected behaviors. For mock backend, returns a neutral score.
    """

    def __init__(self, *, backend: str = "mock", use_case: str = ""):
        self._backend = backend
        self._use_case = use_case

    async def judge(
        self,
        scenario: TestScenario,
        turns: list[TurnRecord],
    ) -> ContentScore:
        """Score a conversation against scenario expectations."""
        if self._backend == "mock":
            return self._mock_judge(scenario, turns)

        return await self._llm_judge(scenario, turns)

    def _mock_judge(
        self, scenario: TestScenario, turns: list[TurnRecord]
    ) -> ContentScore:
        """Quick heuristic scoring for mock backend.

        Can't do real content evaluation without a real LLM,
        but can score basic structural properties.
        """
        # Check if agent produced non-empty responses
        has_responses = any(
            t.agent_text and not t.agent_text.startswith("[") for t in turns
        )
        has_turns = len(turns) > 0

        # Basic structural score
        base = 0.3 if has_turns else 0.0
        if has_responses:
            base = 0.6

        # Behavior coverage: assume mock can't meet behaviors
        behaviors = {b: False for b in scenario.expected_behaviors}

        return ContentScore(
            task_completion=base,
            language_quality=base,
            factual_accuracy=0.5,  # No hallucination risk with mock
            naturalness=base * 0.8,
            behavior_scores=behaviors,
            reasoning="Mock backend — structural scoring only",
        )

    async def _llm_judge(
        self, scenario: TestScenario, turns: list[TurnRecord]
    ) -> ContentScore:
        """Use real LLM to evaluate conversation quality."""
        from jvaf.providers.registry import get_class

        prompt = _build_judge_prompt(scenario, turns, self._use_case)

        try:
            # Get the LLM class for judging
            # Prefer anthropic > openai > google for judging quality
            for provider in ["anthropic", "openai", "google"]:
                try:
                    llm_cls = get_class("llm", provider)
                    break
                except Exception:
                    continue
            else:
                return ContentScore(reasoning="No LLM provider available for judging")

            llm = llm_cls(system_prompt=_JUDGE_SYSTEM_PROMPT)
            await llm.setup()

            # Collect response
            response_parts: list[str] = []
            original_push = llm.push_frame

            async def capture_push(frame, direction=None):
                if hasattr(frame, "text"):
                    response_parts.append(frame.text)

            llm.set_downstream(capture_push)
            await llm.generate(prompt, [{"role": "user", "content": prompt}])
            await llm.cleanup()

            response = "".join(response_parts)
            return parse_judge_response(response)

        except Exception as e:
            return ContentScore(reasoning=f"Judge error: {e}")


def _clamp(value: float) -> float:
    """Clamp value between 0.0 and 1.0."""
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0
