"""LiveKit agent that wires speech-to-text, LLM, and TTS together."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

import numpy as np
import livekit.rtc as rtc
from livekit.agents import AutoSubscribe, JobRequest, WorkerOptions, cli

from .config import RuntimeConfig, load_config
from .models import (
    ModelBundle,
    TranscriptSegment,
    build_model_bundle,
    pcm16_from_frame,
    pcm16_to_bytes,
    resample_audio,
)

logger = logging.getLogger(__name__)


async def _respond_to_segment(
    segment: TranscriptSegment,
    models: ModelBundle,
    send_audio: Callable[[np.ndarray, int], Awaitable[None]],
    response_language: str,
) -> None:
    if not segment.is_final or not segment.text.strip():
        return
    user_text = segment.text.strip()
    logger.info("User said: %s", user_text)
    reply_prompt = f"صارف: {user_text}\nمعاون:" if response_language == "ur" else f"User: {user_text}\nAssistant:"
    response_text = models.llm.complete(reply_prompt)
    logger.info("Assistant response: %s", response_text)
    samples, sample_rate = await models.tts.synthesize(response_text)
    await send_audio(samples, sample_rate)


async def _handle_audio_stream(
    stream: rtc.AudioStream,
    models: ModelBundle,
    send_audio: Callable[[np.ndarray, int], Awaitable[None]],
    response_language: str,
) -> None:
    async for frame in stream:
        samples = pcm16_from_frame(frame)
        for segment in models.transcriber.accept_frame(samples):
            await _respond_to_segment(segment, models, send_audio, response_language)


async def run_agent(job: JobRequest, config: RuntimeConfig) -> None:
    await job.connect()
    logger.info("Connected to room %s", job.room.name)

    models = await build_model_bundle(config)
    output_sample_rate = 24000
    source = rtc.AudioSource(output_sample_rate, 1)
    track = rtc.LocalAudioTrack.create_audio_track("awaz-voice", source)
    await job.room.local_participant.publish_track(track)

    send_lock = asyncio.Lock()

    async def send_audio(samples: np.ndarray, sample_rate: int) -> None:
        async with send_lock:
            waveform = resample_audio(samples, sample_rate, output_sample_rate)
            pcm_bytes = pcm16_to_bytes(waveform)
            frame = rtc.AudioFrame(
                data=pcm_bytes,
                sample_rate=output_sample_rate,
                num_channels=1,
                bytes_per_sample=2,
                samples_per_channel=waveform.shape[0],
            )
            source.capture_frame(frame)

    @job.room.on("track_subscribed")
    def _on_track(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant) -> None:
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        logger.info("Subscribed to audio from %s", participant.identity)
        audio_stream = rtc.AudioStream(track)
        asyncio.create_task(_handle_audio_stream(audio_stream, models, send_audio, config.response_language))

    await job.wait_for_disconnect()


async def entrypoint(job: JobRequest) -> None:
    config = load_config()
    await run_agent(job, config)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            auto_subscribe=AutoSubscribe.AUDIO_ONLY,
        )
    )


if __name__ == "__main__":
    main()
