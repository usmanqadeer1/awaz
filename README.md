# Awaz – Urdu LiveKit Voice Assistant

Awaz is a proof-of-concept conversational voice bot that speaks Urdu. It connects to a [LiveKit](https://livekit.io/) room, listens to remote participant audio, transcribes the speech using open-source models, generates Urdu responses with an open-source large language model, and speaks the reply back in real time using neural text-to-speech (TTS).

The entire pipeline is built on open technologies:

- **Speech-to-text (STT):** [`Systran/faster-whisper-large-v3`](https://huggingface.co/Systran/faster-whisper-large-v3)
- **Language model:** [`tiiuae/falcon-7b-instruct`](https://huggingface.co/tiiuae/falcon-7b-instruct)
- **Text-to-speech:** [`tts_models/multilingual/multi-dataset/xtts_v2`](https://huggingface.co/coqui/XTTS-v2)

All components can be swapped for other open-source models through configuration variables.

## Project layout

```text
app/
├── agent.py      # LiveKit worker that wires STT, LLM, and TTS together
├── config.py     # Environment-driven configuration helpers
├── models.py     # Lightweight wrappers around the selected models
└── __init__.py
pyproject.toml     # Python project metadata and dependencies
```

## Prerequisites

1. Python 3.10 or newer.
2. A running LiveKit server (local or hosted). The [LiveKit quick start](https://docs.livekit.io) explains how to set up a development instance.
3. GPU access is strongly recommended for low-latency inference, but the code also works on CPU (with higher latency).
4. Sufficient disk space and RAM to load the models listed above. Feel free to substitute lighter models if needed.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[cli]
```

The optional `cli` dependency installs `python-dotenv` so that environment variables can be defined in a `.env` file.

## Configuration

Create a `.env` file (or export the variables in your shell) with the following values:

```ini
LIVEKIT_URL=ws://127.0.0.1:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=devsecret
LIVEKIT_ROOM=awaz-demo
LIVEKIT_AGENT_IDENTITY=awaz-assistant

# Model overrides (all optional)
AWAZ_USE_GPU=1
AWAZ_RESPONSE_LANGUAGE=ur
AWAZ_VOICE_TEMPERATURE=0.8
AWAZ_STT_MODEL=Systran/faster-whisper-large-v3
AWAZ_LLM_MODEL=tiiuae/falcon-7b-instruct
AWAZ_TTS_MODEL=tts_models/multilingual/multi-dataset/xtts_v2
AWAZ_TTS_SPEAKER=
```

> **Tip:** Replace the model identifiers with lighter variants if you are testing on limited hardware. For example, you can use `Systran/faster-whisper-small` for STT or a 3B LLM served via [Ollama](https://ollama.com/) with the [`llama.cpp`](https://github.com/ggerganov/llama.cpp) backend.

## Running the agent

1. Ensure your LiveKit server is running and the credentials in `.env` are correct.
2. Start the Awaz worker:

   ```bash
   python -m app.agent
   ```

   The worker connects to the specified LiveKit room and publishes an audio track named `awaz-voice`.

3. Join the same room from a LiveKit-compatible client (for example, the official LiveKit web demo). When you speak in Urdu, the assistant transcribes your voice, generates an Urdu reply, and streams the synthesized audio back into the room.

### Local loop testing (optional)

If you want to test the speech pipeline without LiveKit, you can call the helper modules directly. For instance, to run a one-off transcription/response cycle:

```python
import asyncio

from app.config import load_config
from app.models import build_model_bundle

config = load_config(None)
models = asyncio.run(build_model_bundle(config))
text = "السلام علیکم، آپ کیسے ہیں؟"
reply = models.llm.complete(text)
print(reply)
```

## Customisation

- **Different models:** Override the model environment variables to use alternative open-source checkpoints. Make sure the replacements support Urdu (or your desired language).
- **LLM endpoint:** If you are hosting an open-source LLM behind an inference server such as [Text Generation Inference](https://github.com/huggingface/text-generation-inference) or [vLLM](https://github.com/vllm-project/vllm), adapt `UrduLLM.complete` to call the HTTP endpoint instead of running the model locally.
- **Conversational state:** The current implementation treats each recognised utterance independently. You can add a dialogue memory (e.g. a list of past turns) before calling `UrduLLM.complete` to maintain context.
- **Voice cloning:** Provide an appropriate `AWAZ_TTS_SPEAKER` value to use a specific voice embedding supported by XTTS v2, or fine-tune your own Urdu voice.

## Limitations

- The selected default models are large and resource-intensive; adjust them for your hardware.
- Latency depends on compute resources, model size, and network conditions.
- The simple interpolation-based resampler in `app.models` is adequate for prototyping but should be replaced with a high-quality resampling library (e.g. `torchaudio` or `librosa`) for production use.

## License

This project is provided as-is for demonstration purposes. Each referenced model has its own license—ensure you comply with them when deploying Awaz.
