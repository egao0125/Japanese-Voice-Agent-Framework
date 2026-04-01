"""Run autoresearch with mock providers — no API keys needed."""

import asyncio

from jvaf.autoresearch import AutoresearchConfig, AutoresearchLoop


async def main():
    # Parse the example program
    program = AutoresearchConfig.from_markdown("examples/dental_clinic.md")
    print(f"Program: {program.summary()}")
    print(f"Use case: {program.use_case[:80]}...")
    print()

    # Run 10 iterations with mock backend
    loop = AutoresearchLoop(
        program,
        output_dir="experiments",
        backend="mock",
    )
    summary = await loop.run(iterations=10)

    print(f"\nSummary: {summary}")


if __name__ == "__main__":
    asyncio.run(main())
