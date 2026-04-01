"""Tests for backchannel system."""

import pytest

from jvaf.conversation.backchannel import (
    BackchannelSelector,
    BackchannelSignal,
    SIGNAL_TO_CATEGORIES,
)
from jvaf.lang.ja import JapaneseLanguagePack


class TestBackchannelSignal:
    def test_signal_values(self):
        assert BackchannelSignal.CONTINUER == "continuer"
        assert BackchannelSignal.ASSESSMENT == "assessment"
        assert BackchannelSignal.FORMAL_ACK == "formal_ack"
        assert BackchannelSignal.EMPATHETIC == "empathetic"
        assert BackchannelSignal.SURPRISED == "surprised"

    def test_signal_to_categories(self):
        assert "reactive" in SIGNAL_TO_CATEGORIES["continuer"]
        assert "formal" in SIGNAL_TO_CATEGORIES["formal_ack"]


class TestBackchannelSelector:
    def test_select_continuer(self):
        lang = JapaneseLanguagePack()
        selector = BackchannelSelector(lang.get_backchannel_variants())

        result = selector.select("continuer")
        assert result is not None
        key, text = result
        assert key  # non-empty key
        assert text  # non-empty text

    def test_select_all_signals(self):
        lang = JapaneseLanguagePack()
        selector = BackchannelSelector(lang.get_backchannel_variants())

        for signal in BackchannelSignal:
            result = selector.select(signal.value)
            assert result is not None, f"No selection for {signal}"

    def test_shuffle_bag_exhausts(self):
        """Shuffle bag should return all variants before repeating."""
        variants = {"reactive": [("a", "A"), ("b", "B"), ("c", "C")]}
        selector = BackchannelSelector(variants)

        seen = set()
        for _ in range(3):
            result = selector.select("continuer")
            assert result is not None
            seen.add(result[0])

        assert seen == {"a", "b", "c"}

    def test_unknown_signal(self):
        selector = BackchannelSelector({})
        result = selector.select("unknown")
        assert result is None
