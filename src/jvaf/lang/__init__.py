"""Language packs — language-specific configuration for turn-taking, backchannel, and text processing."""

from .base import LanguagePack
from .ja import JapaneseLanguagePack

__all__ = ["JapaneseLanguagePack", "LanguagePack"]
