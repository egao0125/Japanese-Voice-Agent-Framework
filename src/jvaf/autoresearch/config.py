"""Autoresearch configuration — parsed from the user's program.md."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TestScenario:
    """A single test scenario for evaluation."""

    name: str
    description: str
    user_utterances: list[str] = field(default_factory=list)


@dataclass
class AutoresearchConfig:
    """Configuration for the autoresearch loop, parsed from program.md.

    The user defines their voice agent's use case, goals, constraints,
    and test scenarios in a markdown file. This class parses it into
    a structured format the loop can act on.
    """

    use_case: str = ""
    goals: dict[str, str] = field(default_factory=dict)
    constraints: dict[str, str] = field(default_factory=dict)
    test_scenarios: list[TestScenario] = field(default_factory=list)
    iterations: int = 50
    backend: str = "mock"
    output_dir: str = "experiments"

    @classmethod
    def from_markdown(cls, path: str | Path) -> AutoresearchConfig:
        """Parse a program.md file into AutoresearchConfig."""
        text = Path(path).read_text(encoding="utf-8")
        config = cls()

        # Parse ## sections
        sections: dict[str, str] = {}
        current_section = ""
        current_lines: list[str] = []

        for line in text.splitlines():
            if line.startswith("## "):
                if current_section:
                    sections[current_section] = "\n".join(current_lines).strip()
                current_section = line[3:].strip().lower()
                current_lines = []
            else:
                current_lines.append(line)
        if current_section:
            sections[current_section] = "\n".join(current_lines).strip()

        # Use case
        for key in ("use case", "usecase", "description"):
            if key in sections:
                config.use_case = sections[key]
                break

        # Goals — parse "- key: value" lines
        if "goals" in sections:
            config.goals = _parse_kv_list(sections["goals"])

        # Constraints
        if "constraints" in sections:
            config.constraints = _parse_kv_list(sections["constraints"])

        # Test scenarios — numbered list items
        for key in ("test scenarios", "scenarios", "tests"):
            if key in sections:
                config.test_scenarios = _parse_scenarios(sections[key])
                break

        return config

    def summary(self) -> str:
        return (
            f"AutoresearchConfig: {len(self.goals)} goals, "
            f"{len(self.test_scenarios)} scenarios, "
            f"{self.iterations} iterations, backend={self.backend}"
        )


def _parse_kv_list(text: str) -> dict[str, str]:
    """Parse lines like '- Key: value' or '- value' into a dict."""
    result: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("-"):
            continue
        line = line[1:].strip()
        if ":" in line:
            k, v = line.split(":", 1)
            result[k.strip()] = v.strip()
        else:
            result[line] = ""
    return result


def _parse_scenarios(text: str) -> list[TestScenario]:
    """Parse numbered scenario list into TestScenario objects."""
    scenarios: list[TestScenario] = []
    for line in text.splitlines():
        line = line.strip()
        match = re.match(r"^\d+\.\s*(.+)$", line)
        if match:
            desc = match.group(1).strip()
            scenarios.append(TestScenario(name=f"scenario_{len(scenarios) + 1}", description=desc))
    return scenarios
