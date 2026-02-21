# awaz - Offline English Live Voice Agent (LiveKit)

This repository now includes a **local/offline-first LiveKit voice agent** that can talk to you in real time (English only for v1).

## What this build does

- Uses **LiveKit** for low-latency live audio transport.
- Uses **local models/services** for speech + reasoning:
  - **STT**: Faster-Whisper (English model)
  - **LLM**: Ollama (local model, e.g. `llama3.1:8b`)
  - **TTS**: Piper (English voice)
- Restricts the assistant to **English responses** for this first version.

> Note: Truly offline operation requires models to be downloaded once ahead of time.

---

## Architecture

`Mic (client) -> LiveKit room -> Python agent -> STT (local) -> LLM (local) -> TTS (local) -> LiveKit room -> Speaker (client)`

---

## Prerequisites

- Python 3.10+
- A running LiveKit server (self-hosted is recommended for offline use)
- Ollama installed locally (`ollama serve`)
- Piper installed locally (CLI binary available as `piper`)
- Whisper model files available locally (for Faster-Whisper)

---

## Environment

Copy `.env.example` to `.env` and set values:

```bash
cp .env.example .env
```

Important fields:

- `LIVEKIT_URL` (e.g. `ws://localhost:7880`)
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`
- `OLLAMA_MODEL` (English-capable local model)
- `PIPER_MODEL_PATH` (English `.onnx` voice model)

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Run

```bash
python agent.py dev
```

This starts a LiveKit worker process and waits for a job. Attach from your LiveKit client/web app and talk naturally.

---

## Notes for English-only v1

- The system prompt enforces English responses.
- Whisper is configured with `language="en"`.
- Piper model should be an English voice.

---

## Next improvements

- Add interruption handling and barge-in tuning.
- Add wake-word support.
- Add conversational memory and RAG.
- Add multilingual support in v2.
