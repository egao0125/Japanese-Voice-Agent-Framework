"""Autoresearch loop — the main product. Propose → build → test → eval → keep/discard."""

from __future__ import annotations

from pathlib import Path

from jvaf.config import PipelineConfig

from .audio_gen import AudioGenerator
from .config import AutoresearchConfig, detect_focus_params
from .evaluator import EvalScore, PipelineEvaluator
from .log import ExperimentEntry, ExperimentLog
from .proposer import PipelineProposer
from .simulator import ConversationSimulator


class AutoresearchLoop:
    """Autonomous pipeline optimization loop.

    1. READ program goals + experiment history
    2. GENERATE test audio (once, cached for all iterations)
    3. PROPOSE a pipeline config change
    4. BUILD pipeline with proposed config
    5. TEST via simulated conversations (real audio → real STT → real LLM)
    6. EVALUATE against goals (LLM-as-judge for real backends)
    7. KEEP or DISCARD based on score improvement
    8. LOG to experiment history
    9. GOTO 3
    """

    def __init__(
        self,
        program: AutoresearchConfig,
        *,
        base_config: PipelineConfig | None = None,
        output_dir: str | Path = "experiments",
        backend: str = "mock",
    ):
        self._program = program
        self._base_config = base_config or PipelineConfig()
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._backend = backend
        self._log = ExperimentLog(self._output_dir / "log.tsv")
        self._focus_params = detect_focus_params(program.improvement_focus)
        self._proposer = PipelineProposer(
            backend=backend,
            available_providers=program.available_providers or None,
            focus_params=self._focus_params,
        )
        self._simulator = ConversationSimulator()
        self._evaluator = PipelineEvaluator(
            backend=backend,
            use_case=program.use_case,
            focus_params=self._focus_params,
        )
        self._audio_gen = AudioGenerator(
            cache_dir=self._output_dir / "audio_cache",
        )

        self._best_config = self._base_config.model_copy(deep=True)
        self._best_score = self._log.best_score() if self._log.entries else 0.0
        self._kept_count = 0

    async def run(self, iterations: int | None = None) -> dict:
        """Run the autoresearch loop for N iterations.

        Returns summary dict with kept/discarded counts and best score.
        """
        n = iterations or self._program.iterations
        start_iter = len(self._log.entries)

        avail = self._program.available_providers
        provider_summary = ", ".join(
            f"{cat}:[{','.join(ps)}]" for cat, ps in avail.items() if len(ps) > 1
        ) if avail else "mock only"

        print(f"Starting autoresearch: {n} iterations")
        print(f"  Goals: {len(self._program.goals)}")
        print(f"  Scenarios: {len(self._program.test_scenarios)}")
        print(f"  Providers: {provider_summary}")
        if self._focus_params:
            print(f"  Focus: {', '.join(sorted(self._focus_params))}")
        print(f"  Output: {self._output_dir}")

        # Generate test audio (one-time, cached)
        if self._backend != "mock":
            await self._prepare_audio()

        print()

        for i in range(n):
            iter_num = start_iter + i + 1
            await self._run_iteration(iter_num)

        # Save best config
        best_dir = self._output_dir / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        self._best_config.to_yaml(best_dir / "config.yaml")

        summary = {
            "total_iterations": n,
            "kept": self._kept_count,
            "discarded": n - self._kept_count,
            "best_score": self._best_score,
            "best_config": str(best_dir / "config.yaml"),
        }

        print(f"\nAutoresearch complete:")
        print(f"  Final phase: {self._proposer.phase.value}")
        print(f"  Kept: {self._kept_count}/{n}")
        print(f"  Best score: {self._best_score:.4f}")
        print(f"  Best config: {best_dir / 'config.yaml'}")

        # Generate lessons
        self._write_lessons()

        # Cleanup
        await self._audio_gen.cleanup()

        return summary

    async def _prepare_audio(self) -> None:
        """Generate test audio for all scenarios (one-time setup).

        Uses available TTS to synthesize scripted user utterances into
        WAV files. These are fed through the real pipeline so STT
        transcribes real speech instead of noise.
        """
        # Ensure scenarios have utterances
        for scenario in self._program.test_scenarios:
            if not scenario.user_utterances:
                scenario.user_utterances = self._simulator._generate_default_utterances(  # noqa: SLF001
                    scenario
                )

        provider = await self._audio_gen.setup()
        print(f"  Audio: generating with {provider}")

        audio_cache = await self._audio_gen.generate_all(
            self._program.test_scenarios
        )
        self._simulator.set_audio_cache(audio_cache)

        total_files = sum(len(paths) for paths in audio_cache.values())
        print(f"  Audio: {total_files} files cached")

    async def _run_iteration(self, iter_num: int) -> None:
        """Run a single propose → test → eval → keep/discard iteration."""
        # 1. PROPOSE
        proposal = self._proposer.propose(self._best_config, self._program, self._log)
        print(f"[iter {iter_num}] Hypothesis: {proposal.hypothesis}")

        # 2. TEST — run scenarios with proposed config
        results = await self._simulator.run_all(
            proposal.config, self._program.test_scenarios
        )

        # 3. EVALUATE (content judge for real backends, parametric for mock)
        if self._backend != "mock":
            score = await self._evaluator.evaluate_with_content(results, self._program)
        else:
            score = self._evaluator.evaluate(results, self._program)

        # 4. KEEP or DISCARD
        kept = score.overall > self._best_score
        if kept:
            self._best_config = proposal.config.model_copy(deep=True)
            self._best_score = score.overall
            self._kept_count += 1

            # Save kept config
            iter_dir = self._output_dir / f"iter-{iter_num:04d}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            proposal.config.to_yaml(iter_dir / "config.yaml")

        status = "KEPT" if kept else "discarded"
        print(f"  Score: {score.overall:.4f} ({status}) | {score.summary()}")

        # 5. LOG
        entry = ExperimentEntry(
            iteration=iter_num,
            hypothesis=proposal.hypothesis,
            config_diff=proposal.diff,
            score=score.overall,
            metrics=score.metrics,
            kept=kept,
        )
        self._log.append(entry)

    def _write_lessons(self) -> None:
        """Auto-generate lessons.md from experiment history."""
        kept = self._log.kept_entries()
        if not kept:
            return

        lines = ["# Autoresearch Lessons\n"]
        lines.append(f"Total experiments: {len(self._log.entries)}")
        lines.append(f"Kept: {len(kept)}")
        lines.append(f"Best score: {self._best_score:.4f}\n")

        lines.append("## Improvements that worked\n")
        for e in kept:
            lines.append(f"- **iter {e.iteration}**: {e.hypothesis} (score: {e.score:.4f})")
            if e.config_diff:
                for path, change in e.config_diff.items():
                    lines.append(f"  - `{path}`: {change.get('from', '?')} → {change.get('to', '?')}")

        lessons_path = self._output_dir / "lessons.md"
        lessons_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
