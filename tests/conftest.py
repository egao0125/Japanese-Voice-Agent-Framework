"""Shared fixtures for JVAF tests."""

import pytest

from jvaf.config import PipelineConfig


@pytest.fixture
def default_config():
    return PipelineConfig()
