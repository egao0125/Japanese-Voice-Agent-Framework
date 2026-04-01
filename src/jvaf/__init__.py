"""JVAF: Autoresearch framework for real-time Japanese voice agents."""

__version__ = "0.1.0"

from .agent import VoiceAgent
from .config import PipelineConfig

__all__ = ["VoiceAgent", "PipelineConfig"]
