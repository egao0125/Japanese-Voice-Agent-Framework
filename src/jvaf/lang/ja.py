"""Japanese language pack — turn-taking, backchannel variants, and text normalization."""

from __future__ import annotations

from .base import BackchannelThresholds, LanguagePack, TurnTakingThresholds


class JapaneseLanguagePack(LanguagePack):
    """Japanese language pack with keigo-aware defaults.

    Turn-taking tuned for Japanese conversation patterns:
    - Longer acceptable silences (Japanese tolerates more pause)
    - Pitch descent is primary turn-end signal
    - Backchannel frequency: every 3-5 seconds (business norm)
    - Character-based barge-in (no word boundaries in Japanese)
    """

    @property
    def code(self) -> str:
        return "ja"

    @property
    def name(self) -> str:
        return "Japanese"

    @property
    def turn_taking(self) -> TurnTakingThresholds:
        return TurnTakingThresholds(
            silence_with_prosody_sec=0.5,
            silence_without_prosody_sec=0.7,
            confidence_threshold=0.65,
            min_barge_in_chars=4,
            max_backchannel_duration_ms=500.0,
            weight_silence=0.35,
            weight_pitch=0.30,
            weight_energy=0.15,
            weight_duration=0.10,
            weight_lengthening=0.10,
        )

    @property
    def backchannel(self) -> BackchannelThresholds:
        return BackchannelThresholds(
            min_interval_sec=5.0,
            neural_min_interval_sec=1.5,
            min_speech_before_bc_sec=1.5,
            min_pause_for_bc_sec=0.05,
            max_pause_for_bc_sec=0.80,
            proactive_threshold_sec=3.5,
        )

    def get_backchannel_variants(self) -> dict[str, list[tuple[str, str]]]:
        return {
            "reactive": [
                ("hai", "はい"),
                ("ee", "ええ"),
                ("un", "うん"),
                ("hai_hai", "はいはい"),
            ],
            "understanding": [
                ("naruhodo", "なるほど"),
                ("sou_desu_ne", "そうですね"),
                ("sou_nan_desu_ne", "そうなんですね"),
            ],
            "formal": [
                ("sayou_de_gozaimasu_ka", "さようでございますか"),
                ("shouchi_itashimashita", "承知いたしました"),
                ("kashikomarimashita", "かしこまりました"),
            ],
            "empathy": [
                ("sore_wa_taihen", "それは大変ですね"),
                ("okimochi_wakarimasu", "お気持ちわかります"),
            ],
            "surprise": [
                ("e", "えっ"),
                ("sou_nan_desu_ka", "そうなんですか"),
                ("hontou_desu_ka", "本当ですか"),
            ],
        }

    def normalize_for_tts(self, text: str) -> str:
        """Normalize Japanese text for TTS (minimal — leave to TTS provider)."""
        return text
