"""Experiment log — append-only TSV tracking all autoresearch iterations."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ExperimentEntry:
    """Single experiment result."""

    iteration: int
    hypothesis: str
    config_diff: dict = field(default_factory=dict)
    score: float = 0.0
    metrics: dict = field(default_factory=dict)
    kept: bool = False
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")


class ExperimentLog:
    """Append-only TSV log for experiment history.

    Columns: iteration, timestamp, hypothesis, config_diff, score, metrics, kept
    """

    COLUMNS = ["iteration", "timestamp", "hypothesis", "config_diff", "score", "metrics", "kept"]

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._entries: list[ExperimentEntry] = []
        if self._path.exists():
            self._load()

    def _load(self) -> None:
        with self._path.open() as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                self._entries.append(
                    ExperimentEntry(
                        iteration=int(row["iteration"]),
                        hypothesis=row["hypothesis"],
                        config_diff=json.loads(row.get("config_diff", "{}")),
                        score=float(row["score"]),
                        metrics=json.loads(row.get("metrics", "{}")),
                        kept=row["kept"] == "True",
                        timestamp=row["timestamp"],
                    )
                )

    def append(self, entry: ExperimentEntry) -> None:
        self._entries.append(entry)
        write_header = not self._path.exists() or self._path.stat().st_size == 0
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS, delimiter="\t")
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "iteration": entry.iteration,
                    "timestamp": entry.timestamp,
                    "hypothesis": entry.hypothesis,
                    "config_diff": json.dumps(entry.config_diff, ensure_ascii=False),
                    "score": f"{entry.score:.4f}",
                    "metrics": json.dumps(entry.metrics, ensure_ascii=False),
                    "kept": entry.kept,
                }
            )

    @property
    def entries(self) -> list[ExperimentEntry]:
        return self._entries

    def kept_entries(self) -> list[ExperimentEntry]:
        return [e for e in self._entries if e.kept]

    def best_score(self) -> float:
        kept = self.kept_entries()
        return max(e.score for e in kept) if kept else 0.0

    def history_summary(self, last_n: int = 10) -> str:
        """Format recent history for the proposer prompt."""
        recent = self._entries[-last_n:]
        lines = []
        for e in recent:
            status = "KEPT" if e.kept else "discarded"
            lines.append(f"  iter {e.iteration}: {e.hypothesis} → {e.score:.3f} ({status})")
        return "\n".join(lines) if lines else "  (no history)"
