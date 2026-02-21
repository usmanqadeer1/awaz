import asyncio
import io
import os
import subprocess
import tempfile
from dataclasses import dataclass

import httpx
import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RoomInputOptions,
    WorkerOptions,
    cli,
)
from livekit.plugins import silero


load_dotenv()


@dataclass
class AppConfig:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    whisper_model: str = os.getenv("WHISPER_MODEL", "base.en")
    piper_model_path: str = os.getenv("PIPER_MODEL_PATH", "./models/en_US-lessac-medium.onnx")
    piper_config_path: str = os.getenv("PIPER_CONFIG_PATH", "./models/en_US-lessac-medium.onnx.json")


class LocalEnglishSTT:
    """Tiny STT helper around Faster-Whisper for English-only transcription."""

    def __init__(self, model_name: str):
        self._model = WhisperModel(model_name, device="cpu", compute_type="int8")

    def transcribe(self, pcm16_mono: bytes, sample_rate: int = 16000) -> str:
        audio = np.frombuffer(pcm16_mono, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self._model.transcribe(audio, language="en", vad_filter=True)
        return " ".join(seg.text.strip() for seg in segments).strip()


class LocalOllamaLLM:
    """Small async client for a local Ollama chat endpoint."""

    def __init__(self, base_url: str, model: str):
        self._base_url = base_url.rstrip("/")
        self._model = model

    async def reply(self, user_text: str) -> str:
        system = (
            "You are a live voice assistant. "
            "Respond in concise, natural spoken English only. "
            "If user speaks non-English, politely ask them to continue in English."
        )
        payload = {
            "model": self._model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ],
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{self._base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        return data.get("message", {}).get("content", "Sorry, I could not produce a response.").strip()


class PiperTTS:
    """Generate wav bytes by shelling out to local `piper` CLI."""

    def __init__(self, model_path: str, config_path: str):
        self.model_path = model_path
        self.config_path = config_path

    def synthesize(self, text: str) -> bytes:
        with tempfile.TemporaryDirectory() as td:
            out_wav = os.path.join(td, "reply.wav")
            cmd = [
                "piper",
                "--model",
                self.model_path,
                "--config",
                self.config_path,
                "--output_file",
                out_wav,
            ]
            proc = subprocess.run(cmd, input=text.encode("utf-8"), check=True, capture_output=True)
            if proc.returncode != 0:
                raise RuntimeError("Piper synthesis failed")
            with open(out_wav, "rb") as f:
                return f.read()


def wav_to_pcm16_mono(wav_bytes: bytes) -> tuple[bytes, int]:
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="int16")
    if len(data.shape) > 1:
        data = data[:, 0]
    return data.tobytes(), sr


class EnglishOfflineAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are an English-only offline voice assistant. "
                "Keep answers friendly, brief, and natural for speech."
            )
        )


async def entrypoint(ctx: JobContext):
    cfg = AppConfig()

    stt = LocalEnglishSTT(cfg.whisper_model)
    llm = LocalOllamaLLM(cfg.ollama_base_url, cfg.ollama_model)
    tts = PiperTTS(cfg.piper_model_path, cfg.piper_config_path)

    await ctx.connect()

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=stt,
        llm=llm,
        tts=tts,
    )

    await session.start(
        room=ctx.room,
        agent=EnglishOfflineAgent(),
        room_input_options=RoomInputOptions(),
    )

    await session.generate_reply(
        instructions="Greet the user and tell them you are ready to chat in English."
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
