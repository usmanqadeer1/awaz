"""Wrappers around open-source speech and language models used by the assistant."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from faster_whisper import WhisperModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import RuntimeConfig

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    """Represents a recognised speech segment."""

    text: str
    is_final: bool


class StreamingTranscriber:
    """Simple helper to consume PCM16 audio chunks and yield transcriptions."""

    def __init__(self, model: WhisperModel, min_chunk_size: int = 16000 * 2):
        self._model = model
        self._buffer = np.array([], dtype=np.float32)
        self._min_chunk_size = min_chunk_size

    def accept_frame(self, frame: np.ndarray) -> Iterable[TranscriptSegment]:
        """Feed a PCM frame to the recogniser."""

        self._buffer = np.concatenate([self._buffer, frame])
        if self._buffer.shape[0] < self._min_chunk_size:
            return []
        segments, _ = self._model.transcribe(
            self._buffer,
            language="ur",
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=True,
        )
        results: list[TranscriptSegment] = []
        for segment in segments:
            results.append(TranscriptSegment(text=segment.text.strip(), is_final=True))
        self._buffer = np.array([], dtype=np.float32)
        return results


class UrduLLM:
    """Thin wrapper around an open-source causal language model."""

    def __init__(self, model_name: str, use_gpu: bool = False):
        logger.info("Loading language model %s", model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        device = "cuda" if use_gpu else "cpu"
        self._model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto" if use_gpu else None)
        self._model.to(device)
        self._device = device

    def complete(self, prompt: str) -> str:
        """Generate a response in Urdu for the provided prompt."""

        messages = (
            "آپ ایک مددگار اردو معاون ہیں جو معلوماتی، مختصر اور شائستہ جوابات دیتے ہیں۔\n"
            "صارف کے ان پٹ کا براہِ راست جواب دیں اور رومن اردو کے بجائے معیاری اردو رسم الخط استعمال کریں۔\n\n"
        )
        input_ids = self._tokenizer(messages + prompt, return_tensors="pt").to(self._device)
        generated = self._model.generate(
            **input_ids,
            do_sample=True,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        text = self._tokenizer.decode(generated[0], skip_special_tokens=True)
        return text[len(messages) :].strip()


class UrduTTS:
    """Wrapper around the Coqui TTS library for multilingual synthesis."""

    def __init__(self, model_name: str, speaker: str | None = None, temperature: float = 0.8, use_gpu: bool = False):
        from TTS.api import TTS  # imported lazily to keep import cost low

        logger.info("Loading TTS model %s", model_name)
        self._tts = TTS(model_name)
        if use_gpu:
            try:
                self._tts.to("cuda")
            except AttributeError:  # pragma: no cover - depends on backend implementation
                logger.warning("TTS backend does not expose a .to() method; running on default device")
        self._speaker = speaker
        self._temperature = temperature

    async def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        loop = asyncio.get_running_loop()

        def _synth() -> tuple[np.ndarray, int]:
            audio = self._tts.tts(
                text=text,
                speaker=self._speaker,
                language="ur",
                split_sentences=True,
                temperature=self._temperature,
            )
            sample_rate = getattr(self._tts.synthesizer, "output_sample_rate", 24000)
            return audio.astype(np.float32), sample_rate

        return await loop.run_in_executor(None, _synth)


@dataclass
class ModelBundle:
    """Collection of helper classes instantiated from a configuration."""

    transcriber: StreamingTranscriber
    llm: UrduLLM
    tts: UrduTTS


async def build_model_bundle(config: RuntimeConfig) -> ModelBundle:
    """Initialise all models using the provided configuration."""

    logger.info("Loading Whisper model %s", config.models.stt_model)
    whisper = WhisperModel(config.models.stt_model, device="cuda" if config.use_gpu else "cpu")
    transcriber = StreamingTranscriber(whisper)

    llm = UrduLLM(config.models.llm_model, use_gpu=config.use_gpu)
    tts = UrduTTS(
        config.models.tts_model,
        speaker=config.models.tts_speaker,
        temperature=config.voice_temperature,
        use_gpu=config.use_gpu,
    )
    return ModelBundle(transcriber=transcriber, llm=llm, tts=tts)


def pcm16_from_frame(frame) -> np.ndarray:
    """Convert a LiveKit audio frame to mono float32 samples."""

    import livekit.rtc as rtc

    if not isinstance(frame, rtc.AudioFrame):
        raise TypeError("Expected a LiveKit AudioFrame")
    if frame.num_channels != 1:
        audio = frame.resample(frame.sample_rate, 1)
    else:
        audio = frame
    pcm = np.frombuffer(audio.data, dtype=np.int16).astype(np.float32) / 32768.0
    return pcm


def resample_audio(samples: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resample a waveform using linear interpolation."""

    if from_rate == to_rate:
        return samples
    duration = samples.shape[0] / float(from_rate)
    target_length = int(duration * to_rate)
    if target_length <= 0:
        return samples
    x_old = np.linspace(0.0, duration, num=samples.shape[0], endpoint=False)
    x_new = np.linspace(0.0, duration, num=target_length, endpoint=False)
    return np.interp(x_new, x_old, samples)


def pcm16_to_bytes(samples: np.ndarray) -> bytes:
    """Convert float32 samples back to 16-bit PCM bytes."""

    clamped = np.clip(samples, -1.0, 1.0)
    pcm = (clamped * 32767.0).astype(np.int16)
    return pcm.tobytes()


__all__ = [
    "ModelBundle",
    "TranscriptSegment",
    "StreamingTranscriber",
    "UrduLLM",
    "UrduTTS",
    "build_model_bundle",
    "pcm16_from_frame",
    "pcm16_to_bytes",
    "resample_audio",
]
