"""CLI — jvaf command-line interface."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="jvaf",
        description="Japanese Voice Agent Framework — autoresearch for voice agents",
    )
    sub = parser.add_subparsers(dest="command")

    # autoresearch
    ar = sub.add_parser("autoresearch", help="Run autoresearch loop on a program.md")
    ar.add_argument("program", type=str, help="Path to program.md")
    ar.add_argument("--iterations", "-n", type=int, default=10, help="Number of iterations")
    ar.add_argument("--backend", type=str, default="mock", help="Backend: mock or llm")
    ar.add_argument("--output", "-o", type=str, default="experiments", help="Output directory")

    # run
    run_p = sub.add_parser("run", help="Run a voice agent from config")
    run_p.add_argument("config", type=str, help="Path to config.yaml")

    # test
    test_p = sub.add_parser("test", help="Run test scenarios against a config")
    test_p.add_argument("config", type=str, help="Path to config.yaml")
    test_p.add_argument("--scenarios", type=str, help="Path to scenarios file")

    args = parser.parse_args(argv)

    if args.command == "autoresearch":
        asyncio.run(_cmd_autoresearch(args))
    elif args.command == "run":
        asyncio.run(_cmd_run(args))
    elif args.command == "test":
        asyncio.run(_cmd_test(args))
    else:
        parser.print_help()
        sys.exit(1)


async def _cmd_autoresearch(args: argparse.Namespace) -> None:
    from jvaf.autoresearch import AutoresearchConfig, AutoresearchLoop

    program = AutoresearchConfig.from_markdown(args.program)
    program.iterations = args.iterations
    program.backend = args.backend
    program.output_dir = args.output

    loop = AutoresearchLoop(
        program,
        output_dir=args.output,
        backend=args.backend,
    )
    summary = await loop.run()
    print(f"\nDone. Best score: {summary['best_score']:.4f}")


async def _cmd_run(args: argparse.Namespace) -> None:
    from jvaf.agent import VoiceAgent
    from jvaf.config import PipelineConfig

    config = PipelineConfig.from_yaml(args.config)
    agent = VoiceAgent.from_config(config)
    print(f"Starting agent: {config.summary()}")
    try:
        await agent.start()
    except KeyboardInterrupt:
        await agent.stop()


async def _cmd_test(args: argparse.Namespace) -> None:
    from jvaf.autoresearch import ConversationSimulator, AutoresearchConfig
    from jvaf.config import PipelineConfig

    config = PipelineConfig.from_yaml(args.config)
    simulator = ConversationSimulator()

    if args.scenarios:
        program = AutoresearchConfig.from_markdown(args.scenarios)
        scenarios = program.test_scenarios
    else:
        from jvaf.autoresearch.config import TestScenario
        scenarios = [TestScenario(name="default", description="Basic conversation test")]

    results = await simulator.run_all(config, scenarios)
    for r in results:
        print(f"  {r.scenario_name}: {r.turn_count} turns, avg latency {r.avg_latency_ms:.0f}ms")
        if r.pipeline_errors:
            for err in r.pipeline_errors:
                print(f"    ERROR: {err}")


if __name__ == "__main__":
    main()
