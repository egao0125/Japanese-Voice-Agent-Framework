"""Autoresearch — autonomous pipeline optimization for voice agents."""

from .audio_gen import AudioGenerator, read_wav
from .config import AutoresearchConfig, TestScenario
from .evaluator import EvalScore, PipelineEvaluator
from .judge import ContentJudge, ContentScore
from .log import ExperimentEntry, ExperimentLog
from .loop import AutoresearchLoop
from .proposer import PipelineProposer, Proposal, SearchPhase
from .simulator import ConversationSimulator, SimulationResult, TurnRecord
