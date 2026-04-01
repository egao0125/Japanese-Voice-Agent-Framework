# JVAF — Japanese Voice Agent Framework

**Autoresearch framework for building real-time voice agents that can hear, understand, and respond naturally in Japanese.**

JVAF autonomously optimizes your voice agent pipeline — STT, turn-taking, backchannel (aizuchi), barge-in, LLM, and TTS — by iterating through experiments the way [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) optimizes neural networks.

## How It Works

1. **You describe your agent** in a `program.md` — use case, goals, constraints
2. **JVAF's autoresearch loop** proposes pipeline changes, tests them, keeps what works
3. **You get an optimized voice agent** with tuned turn-taking, natural backchannel, and low latency

```
program.md (your goals)
     │
     ▼
┌─────────────────────────────────────────────┐
│  1. READ program + lessons + history        │
│  2. PROPOSE pipeline change (LLM)           │
│  3. BUILD pipeline with proposed config     │
│  4. TEST via simulated conversation         │
│  5. EVALUATE against goals                  │
│  6. KEEP or DISCARD                         │
│  7. LOG + GOTO 1                            │
└─────────────────────────────────────────────┘
```

## Quick Start

```bash
pip install jvaf
```

### Run a mock agent (no API keys needed)

```python
import asyncio
from jvaf import VoiceAgent, PipelineConfig

async def main():
    agent = VoiceAgent.from_config(PipelineConfig())
    await agent.start()

asyncio.run(main())
```

### Run autoresearch

```bash
# Write your agent goals in program.md, then:
jvaf autoresearch program.md --base-config pipeline.yaml --iterations 50

# CPU demo with mock providers:
jvaf autoresearch examples/dental_clinic.md --backend mock --iterations 10
```

## What Makes JVAF Different

| Feature | LiveKit Agents | Pipecat | **JVAF** |
|---------|---------------|---------|----------|
| Backchannel (aizuchi) | None | None | **4 trigger paths** |
| Turn-taking | Silence + ML | Silence | **Prosodic + heuristic + silence** |
| Infrastructure | LiveKit required | Flexible | **Transport-agnostic** |
| Language support | English-first | English-first | **Japanese-first** |
| Auto-optimization | None | None | **Autoresearch loop** |
| Barge-in | Binary interrupt | Binary | **Character-aware** |

## Architecture

```
Transport → VAD → BC Trigger → STT → Turn-Taking → Barge-In → LLM → TTS → BC Injector → Output
```

Every component is swappable via ABCs. The autoresearch loop optimizes the pipeline configuration.

### Core Abstractions

- **`FrameProcessor`** — all pipeline components implement this
- **`PipelineConfig`** — every tunable parameter in one YAML
- **`VoiceAgent`** — assembles pipeline from config
- **`LanguagePack`** — thresholds, backchannel variants, normalization per language

### Providers (pluggable)

| Type | Built-in | Plugins |
|------|----------|---------|
| STT | MockSTT | Deepgram, OpenAI, Whisper |
| LLM | MockLLM | Anthropic Claude, OpenAI GPT |
| TTS | MockTTS | ElevenLabs, VOICEVOX |
| VAD | EnergyVAD | Silero |
| Transport | MockTransport | WebSocket, Local mic |

## Backchannel System

JVAF is the first voice agent framework with first-class backchannel support. Four trigger paths detect natural timing for aizuchi:

1. **Reactive** — user pauses 50-800ms then resumes (mid-turn micro-pause)
2. **Proactive** — user speaks continuously for 3+ seconds
3. **Energy dip** — sub-VAD energy drop detection (bypasses VAD lag)
4. **Neural** — ML model prediction (MaAI-style)

Signal taxonomy: `CONTINUER` (はい), `ASSESSMENT` (なるほど), `FORMAL_ACK` (承知いたしました), `EMPATHETIC` (それは大変ですね), `SURPRISED` (えっ)

## Example: Dental Clinic Receptionist

```markdown
# program.md

## Use Case
Japanese dental clinic phone receptionist.

## Goals
- Turn-taking latency: <400ms
- Backchannel: natural aizuchi every 3-5 seconds
- STT accuracy: >90% on dental terms
- Polite keigo throughout
```

```bash
jvaf autoresearch program.md --base-config clinic.yaml --iterations 50
```

Output: optimized `best/config.yaml` + experiment log + auto-generated lessons.

## Development

```bash
git clone https://github.com/egao0125/Japanese-Voice-Agent-Framework.git
cd Japanese-Voice-Agent-Framework
pip install -e ".[dev]"
pytest tests/
```

## License

MIT
