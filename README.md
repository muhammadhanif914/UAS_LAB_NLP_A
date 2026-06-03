
# UAS_LAB_NLP_A
=======
# voice-cs-system

Voice CS System — pipeline speech-to-speech untuk code-switching (ID / EN / AR).

Quick start
1. Buat virtualenv dan install dependency:

```bash
python -m venv env
env\Scripts\activate    # Windows
pip install -r requirements.txt
pip install transformers==5.0.0
```

2. Salin `.env.example` ke `.env` dan isi `GEMINI_API_KEY` serta variabel lain sesuai kebutuhan.

3. Siapkan `whisper.cpp` build dan model:

```bash
# letakkan binary whisper-cli di app/whisper.cpp/build/bin/Release/whisper-cli.exe (Windows)
# atau app/whisper.cpp/build/bin/whisper-cli (Linux/macOS)
# letakkan model ggml di app/whisper.cpp/models/ (mis. ggml-base.bin)
```

4. Jika punya model Coqui TTS lokal, letakkan file checkpoint di `app/coqui_tts/` dan isi `config.json`.
	Jika tidak, sistem akan default ke model remote `tts_models/id/cv/vits`.

Run server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

API endpoints
- `POST /voice-chat` : upload audio (.wav) untuk pipeline lengkap
- `POST /transcribe-only` : hanya STT
- `GET /health` : cek status

Files of interest
- [app/main.py](app/main.py#L1-L1)
- [app/stt.py](app/stt.py#L1-L1)
- [app/llm.py](app/llm.py#L1-L1)
- [app/tts.py](app/tts.py#L1-L1)

If you'd like, I can now attempt to: (a) run a dry-check of environment variables in `.env`, (b) test-load the TTS model (no audio), or (c) prepare a small test script to exercise STT→LLM→TTS locally.

