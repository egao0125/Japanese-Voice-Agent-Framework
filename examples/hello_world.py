"""Minimal JVAF agent — runs entirely with mock providers on CPU."""

import asyncio

from jvaf.agent import VoiceAgent
from jvaf.config import PipelineConfig


async def main():
    # Default config uses all mock providers
    config = PipelineConfig()
    print(f"Config: {config.summary()}")

    agent = VoiceAgent.from_config(config)

    print("Starting agent...")
    await agent.start()

    # Feed some synthetic audio through the pipeline
    import numpy as np
    from jvaf.core.frames import InputAudioFrame

    audio = np.random.randn(16000).astype(np.float32) * 0.3  # 1 second
    frame = InputAudioFrame(audio=audio.tobytes(), sample_rate=16000)
    await agent.pipeline.push_frame(frame)

    # Let pipeline settle
    await asyncio.sleep(0.1)

    output = agent.transport.recorded_output
    print(f"Output frames: {len(output)}")

    await agent.stop()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
