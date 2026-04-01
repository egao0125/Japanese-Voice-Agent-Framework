"""VoiceAgent — high-level orchestrator that assembles the pipeline from config."""

from __future__ import annotations

from jvaf.config import PipelineConfig
from jvaf.core.events import EventBus
from jvaf.core.pipeline import Pipeline
from jvaf.providers.llm import LLMProvider, MockLLM
from jvaf.providers.stt import MockSTT, STTProvider
from jvaf.providers.transport import MockTransport, Transport
from jvaf.providers.tts import MockTTS, TTSProvider
from jvaf.providers.vad import EnergyVAD, VADProvider


class VoiceAgent:
    """High-level agent that wires together all components.

    Can be constructed from explicit providers or from a PipelineConfig
    (which the autoresearch loop uses).
    """

    def __init__(
        self,
        *,
        transport: Transport | None = None,
        stt: STTProvider | None = None,
        llm: LLMProvider | None = None,
        tts: TTSProvider | None = None,
        vad: VADProvider | None = None,
        config: PipelineConfig | None = None,
    ):
        self._config = config or PipelineConfig()
        self._transport = transport or self._build_transport()
        self._stt = stt or self._build_stt()
        self._llm = llm or self._build_llm()
        self._tts = tts or self._build_tts()
        self._vad = vad or self._build_vad()
        self._events = EventBus()
        self._pipeline: Pipeline | None = None

    @classmethod
    def from_config(cls, config: PipelineConfig) -> VoiceAgent:
        """Build agent entirely from a PipelineConfig."""
        return cls(config=config)

    async def start(self) -> None:
        """Build and start the pipeline."""
        processors = self._build_pipeline()
        self._pipeline = Pipeline(processors)
        await self._transport.connect()
        await self._pipeline.run()

    async def stop(self) -> None:
        """Stop the pipeline and disconnect transport."""
        if self._pipeline:
            await self._pipeline.stop()
        await self._transport.disconnect()

    @property
    def pipeline(self) -> Pipeline | None:
        return self._pipeline

    @property
    def transport(self) -> Transport:
        return self._transport

    @property
    def events(self) -> EventBus:
        return self._events

    @property
    def config(self) -> PipelineConfig:
        return self._config

    def _build_pipeline(self) -> list:
        """Assemble the processor chain."""
        return [
            self._transport.input(),
            self._vad,
            self._stt,
            self._llm,
            self._tts,
            self._transport.output(),
        ]

    def _build_transport(self) -> Transport:
        return MockTransport(sample_rate=self._config.transport.sample_rate)

    def _build_stt(self) -> STTProvider:
        return MockSTT(language=self._config.stt.language)

    def _build_llm(self) -> LLMProvider:
        return MockLLM(system_prompt=self._config.llm.system_prompt)

    def _build_tts(self) -> TTSProvider:
        return MockTTS(sample_rate=self._config.tts.sample_rate)

    def _build_vad(self) -> VADProvider:
        return EnergyVAD(
            threshold_db=self._config.vad.threshold_db,
            min_speech_ms=self._config.vad.min_speech_ms,
            min_silence_ms=self._config.vad.min_silence_ms,
        )
