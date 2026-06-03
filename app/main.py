
import os
import uuid
import logging
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse

from app.stt import transcribe
from app.llm import generate_response
from app.tts import synthesize, _get_tts
from app.utils import normalize_text, detect_languages, text_to_phoneme

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Inisialisasi FastAPI ──────────────────────────────────────────────────────
app = FastAPI(
    title="Voice CS System API",
    description="Pipeline STT → LLM → TTS untuk chatbot suara multibahasa (ID/EN/AR)",
    version="1.0.0"
)

TEMP_DIR = tempfile.gettempdir()


# ── Preload model saat startup ────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Preload TTS model saat backend start agar request pertama tidak lambat."""
    logger.info("[MAIN] Pre-loading TTS model saat startup...")
    print("[MAIN] Pre-loading TTS model, harap tunggu...")
    _get_tts()
    print("[MAIN] TTS model siap. Backend siap menerima request.")


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "Voice CS System API berjalan."}


# ── Endpoint utama ────────────────────────────────────────────────────────────
@app.post("/voice-chat")
async def voice_chat(file: UploadFile = File(...)):
    """
    Menerima file audio dari frontend, memproses melalui pipeline
    STT → Text Processing → LLM → TTS, dan mengembalikan audio respons.

    Args:
        file : file audio WAV dari frontend (Gradio)

    Returns:
        FileResponse berisi file audio hasil TTS
    """
    input_audio_path  = os.path.join(TEMP_DIR, f"input_{uuid.uuid4().hex}.wav")
    output_audio_path = os.path.join(TEMP_DIR, f"output_{uuid.uuid4().hex}.wav")

    try:
        # Simpan audio input ke disk
        audio_bytes = await file.read()
        with open(input_audio_path, "wb") as f:
            f.write(audio_bytes)
        logger.info(f"[MAIN] Audio diterima: {input_audio_path}")

        # ── Step 1: STT ───────────────────────────────────────────────
        logger.info("[MAIN] STT — transkripsi audio...")
        stt_result = transcribe(input_audio_path)
        raw_text   = stt_result.get("text", "").strip()

        if not raw_text:
            raise HTTPException(
                status_code=400,
                detail="Transkripsi gagal atau audio tidak terdeteksi."
            )
        logger.info(f"[MAIN] STT hasil: {raw_text}")

        # ── Step 2: Text Processing ───────────────────────────────────
        normalized = normalize_text(raw_text)
        lang_info  = detect_languages(normalized)
        logger.info(f"[MAIN] Bahasa terdeteksi: {lang_info['languages_detected']}")
        logger.info(f"[MAIN] Code-switching: {lang_info['is_code_switching']}")

        # ── Step 3: LLM ───────────────────────────────────────────────
        logger.info("[MAIN] LLM — generate respons Gemini...")
        llm_response = generate_response(normalized)

        if not llm_response or llm_response.startswith("[ERROR]"):
            raise HTTPException(
                status_code=500,
                detail=f"Gemini gagal: {llm_response}"
            )
        logger.info(f"[MAIN] LLM respons: {llm_response}")

        # ── Step 4: Konversi ke fonem ─────────────────────────────────
        tts_text = text_to_phoneme(llm_response)
        logger.info(f"[MAIN] TTS input (fonem): {tts_text}")

        # ── Step 5: TTS ───────────────────────────────────────────────
        logger.info("[MAIN] TTS — sintesis suara...")
        synthesize(tts_text, output_audio_path)

        if not os.path.exists(output_audio_path):
            raise HTTPException(
                status_code=500,
                detail="File audio output tidak berhasil dibuat."
            )
        logger.info(f"[MAIN] Audio output siap: {output_audio_path}")

        return FileResponse(
            path=output_audio_path,
            media_type="audio/wav",
            filename="response.wav"
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"[MAIN] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan: {str(e)}")

    finally:
        if os.path.exists(input_audio_path):
            os.remove(input_audio_path)

    