"""Pipeline configuration — the thing autoresearch optimizes."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class STTConfig(BaseModel):
    provider: str = "mock"
    language: str = "ja"


class LLMConfig(BaseModel):
    provider: str = "mock"
    model: str = ""
    system_prompt: str = ""


class TTSConfig(BaseModel):
    provider: str = "mock"
    voice_id: str = ""
    sample_rate: int = 16000


class VADConfig(BaseModel):
    provider: str = "energy"
    threshold_db: float = -35.0
    min_speech_ms: float = 250
    min_silence_ms: float = 300


class TurnTakingConfig(BaseModel):
    strategy: str = "silence"
    silence_threshold_sec: float = 0.5
    min_speech_ms: float = 300


class BargeInConfig(BaseModel):
    enabled: bool = True
    min_chars: int = 4


class BackchannelConfig(BaseModel):
    enabled: bool = True
    min_interval_sec: float = 5.0
    min_speech_before_bc_sec: float = 1.5
    triggers: list[str] = Field(default_factory=lambda: ["reactive", "proactive", "energy"])


class TransportConfig(BaseModel):
    type: str = "mock"
    sample_rate: int = 16000
    host: str = "0.0.0.0"
    port: int = 8765


class PipelineConfig(BaseModel):
    """Complete pipeline configuration — every tunable parameter.

    This is the object that the autoresearch loop proposes changes to.
    Serializable to/from YAML for experiment tracking.
    """

    stt: STTConfig = Field(default_factory=STTConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    turn_taking: TurnTakingConfig = Field(default_factory=TurnTakingConfig)
    barge_in: BargeInConfig = Field(default_factory=BargeInConfig)
    backchannel: BackchannelConfig = Field(default_factory=BackchannelConfig)
    transport: TransportConfig = Field(default_factory=TransportConfig)
    language: str = "ja"

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        with Path(path).open() as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw or {})

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.dump(self.model_dump(), default_flow_style=False, allow_unicode=True)
        )

    def summary(self) -> str:
        return (
            f"Pipeline: stt={self.stt.provider} llm={self.llm.provider} "
            f"tts={self.tts.provider} vad={self.vad.provider} "
            f"turn={self.turn_taking.strategy} bc={self.backchannel.enabled} "
            f"lang={self.language}"
        )
