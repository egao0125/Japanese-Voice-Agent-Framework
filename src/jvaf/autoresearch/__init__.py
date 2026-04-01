"""Autoresearch — autonomous pipeline optimization for voice agents."""

from .config import AutoresearchConfig, TestScenario
from .evaluator import EvalScore, PipelineEvaluator
from .log import ExperimentEntry, ExperimentLog
from .loop import AutoresearchLoop
from .proposer import PipelineProposer, Proposal, SearchPhase
from .simulator import ConversationSimulator, SimulationResult, TurnRecord
