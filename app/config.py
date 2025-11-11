"""Configuration helpers for the Awaz Urdu voice bot."""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclasses.dataclass(slots=True)
class ModelConfig:
    """Configuration describing the model bundle used by the voice bot."""

    stt_model: str = dataclasses.field(
        default_factory=lambda: os.getenv("AWAZ_STT_MODEL", "Systran/faster-whisper-large-v3")
    )
    llm_model: str = dataclasses.field(
        default_factory=lambda: os.getenv("AWAZ_LLM_MODEL", "tiiuae/falcon-7b-instruct")
    )
    tts_model: str = dataclasses.field(
        default_factory=lambda: os.getenv("AWAZ_TTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
    )
    tts_speaker: Optional[str] = dataclasses.field(default_factory=lambda: os.getenv("AWAZ_TTS_SPEAKER"))


@dataclasses.dataclass(slots=True)
class LiveKitConfig:
    """Runtime credentials required to connect to a LiveKit deployment."""

    url: str = dataclasses.field(default_factory=lambda: os.getenv("LIVEKIT_URL", "ws://127.0.0.1:7880"))
    api_key: str = dataclasses.field(default_factory=lambda: os.getenv("LIVEKIT_API_KEY", "devkey"))
    api_secret: str = dataclasses.field(default_factory=lambda: os.getenv("LIVEKIT_API_SECRET", "devsecret"))
    room: str = dataclasses.field(default_factory=lambda: os.getenv("LIVEKIT_ROOM", "awaz-demo"))
    agent_identity: str = dataclasses.field(default_factory=lambda: os.getenv("LIVEKIT_AGENT_IDENTITY", "awaz-assistant"))


@dataclasses.dataclass(slots=True)
class RuntimeConfig:
    """Aggregated configuration for the application."""

    livekit: LiveKitConfig = dataclasses.field(default_factory=LiveKitConfig)
    models: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    use_gpu: bool = dataclasses.field(default_factory=lambda: os.getenv("AWAZ_USE_GPU", "0") == "1")
    voice_temperature: float = dataclasses.field(default_factory=lambda: float(os.getenv("AWAZ_VOICE_TEMPERATURE", "0.8")))
    response_language: str = dataclasses.field(default_factory=lambda: os.getenv("AWAZ_RESPONSE_LANGUAGE", "ur"))


def load_config(env_file: str | os.PathLike[str] | None = ".env") -> RuntimeConfig:
    """Load configuration from the environment, optionally reading a ``.env`` file."""

    if env_file:
        env_path = Path(env_file)
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)
    return RuntimeConfig()


__all__ = ["ModelConfig", "LiveKitConfig", "RuntimeConfig", "load_config"]
