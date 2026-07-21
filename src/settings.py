from evdev import ecodes

import json
import math
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
os.environ["HF_HOME"] = str(MODEL_DIR / ".cache")

ENV_FILE = Path("~/.config/glossy.env").expanduser()
CONFIG_FILE = PROJECT_ROOT / "config" / "config.json"


def load_settings(path=CONFIG_FILE):
    try:
        settings = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot load {path}: {error}") from error

    expected = {
        "model",
        "reasoning_effort",
        "transcription_model",
        "transcription_beam_size",
        "hold_seconds",
        "button",
        "visualizer_sensitivity",
        "speech_rms_threshold",
        "minimum_speech_seconds",
        "vad_aggressiveness",
        "speech_snr_ratio",
    }
    if not isinstance(settings, dict) or set(settings) != expected:
        raise RuntimeError(f"{path} must contain exactly: {', '.join(sorted(expected))}")
    if not isinstance(settings["model"], str) or not settings["model"].strip():
        raise RuntimeError("model must be a non-empty string")
    efforts = {None, "none", "low", "medium", "high", "xhigh"}
    if settings["reasoning_effort"] not in efforts:
        raise RuntimeError(
            "reasoning_effort must be null, none, low, medium, high, or xhigh"
        )
    if (
        not isinstance(settings["transcription_model"], str)
        or not settings["transcription_model"].strip()
    ):
        raise RuntimeError("transcription_model must be a non-empty string")
    beam_size = settings["transcription_beam_size"]
    if isinstance(beam_size, bool) or not isinstance(beam_size, int) or beam_size < 1:
        raise RuntimeError("transcription_beam_size must be a positive integer")
    hold_seconds = settings["hold_seconds"]
    if (
        isinstance(hold_seconds, bool)
        or not isinstance(hold_seconds, (int, float))
        or not math.isfinite(hold_seconds)
        or hold_seconds < 0
    ):
        raise RuntimeError("hold_seconds must be a non-negative number")
    if not isinstance(settings["button"], str) or not isinstance(
        getattr(ecodes, settings["button"], None), int
    ):
        raise RuntimeError("button must be a Linux evdev key name such as KEY_RIGHTALT")
    sensitivity = settings["visualizer_sensitivity"]
    if (
        isinstance(sensitivity, bool)
        or not isinstance(sensitivity, (int, float))
        or not math.isfinite(sensitivity)
        or sensitivity <= 0
    ):
        raise RuntimeError("visualizer_sensitivity must be a positive number")
    speech_threshold = settings["speech_rms_threshold"]
    if (
        isinstance(speech_threshold, bool)
        or not isinstance(speech_threshold, (int, float))
        or not math.isfinite(speech_threshold)
        or speech_threshold <= 0
    ):
        raise RuntimeError("speech_rms_threshold must be a positive number")
    minimum_speech = settings["minimum_speech_seconds"]
    if (
        isinstance(minimum_speech, bool)
        or not isinstance(minimum_speech, (int, float))
        or not math.isfinite(minimum_speech)
        or minimum_speech < 0
    ):
        raise RuntimeError("minimum_speech_seconds must be a non-negative number")
    aggressiveness = settings["vad_aggressiveness"]
    if isinstance(aggressiveness, bool) or aggressiveness not in range(4):
        raise RuntimeError("vad_aggressiveness must be an integer from 0 to 3")
    snr_ratio = settings["speech_snr_ratio"]
    if (
        isinstance(snr_ratio, bool)
        or not isinstance(snr_ratio, (int, float))
        or not math.isfinite(snr_ratio)
        or snr_ratio <= 1
    ):
        raise RuntimeError("speech_snr_ratio must be a number greater than 1")
    return settings


def load_environment(path=ENV_FILE):
    if path.exists():
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            key, separator, value = line.removeprefix("export ").partition("=")
            if not separator:
                raise RuntimeError(f"Invalid line in {path}: {raw_line!r}")
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            os.environ.setdefault(key.strip(), value)

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(f"Missing OPENAI_API_KEY in {path}")
