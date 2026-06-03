import os
import re
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Path konfigurasi ──────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent        # → app/
ROOT_DIR    = BASE_DIR.parent                        # → voice-cs-system/

# Cari whisper.cpp di app/ dulu, kalau tidak ada cari di root
WHISPER_DIR = BASE_DIR / "whisper.cpp"
if not WHISPER_DIR.exists():
    WHISPER_DIR = ROOT_DIR / "whisper.cpp"

# Path binary whisper-cli (Windows: Release/whisper-cli.exe)
WHISPER_BIN = WHISPER_DIR / "build" / "bin" / "Release" / "whisper-cli.exe"
if not WHISPER_BIN.exists():
    # Fallback Linux/macOS
    WHISPER_BIN = WHISPER_DIR / "build" / "bin" / "whisper-cli"

MODEL_FILE    = os.getenv("WHISPER_MODEL_FILE", "ggml-base.bin")
WHISPER_MODEL = WHISPER_DIR / "models" / MODEL_FILE



# ── Fungsi utama ──────────────────────────────────────────────────────────────
def transcribe(audio_path: str, language: str = None) -> dict:
    """
    Transkripsi file audio (.wav) ke teks menggunakan whisper.cpp CLI.

    Args:
        audio_path : path ke file .wav
        language   : kode bahasa opsional ('id','en','ar'). None = auto-detect.

    Returns:
        dict: text, language, no_speech_prob, segments
    """
    audio_path = str(Path(audio_path).resolve())

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"[STT] File audio tidak ditemukan: {audio_path}")
    if not WHISPER_BIN.exists():
        raise RuntimeError(
            f"[STT] whisper-cli tidak ditemukan di: {WHISPER_BIN}\n"
            f"      Jalankan build dulu:\n"
            f"      cd {WHISPER_DIR}\n"
            f"      cmake -B build -DCMAKE_BUILD_TYPE=Release\n"
            f"      cmake --build build --config Release"
        )
    if not WHISPER_MODEL.exists():
        raise RuntimeError(
            f"[STT] Model tidak ditemukan: {WHISPER_MODEL}\n"
            f"      Download model dulu dari:\n"
            f"      https://huggingface.co/ggerganov/whisper.cpp"
        )

    cmd = [
        str(WHISPER_BIN),
        "-m", str(WHISPER_MODEL),
        "-f", audio_path,
        "--output-txt",
        "--no-timestamps",
    ]
    if not language:
        language = "id"
    cmd += ["-l", language]

    print(f"[STT] Transkripsi: {os.path.basename(audio_path)}")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("[STT] Timeout. Coba model yang lebih kecil.")

    # whisper-cli simpan output ke <audio>.txt
    txt_file = audio_path + ".txt"
    text = ""
    if os.path.exists(txt_file):
        with open(txt_file, encoding="utf-8") as f:
            text = f.read().strip()
        os.remove(txt_file)
    else:
        text = proc.stdout.strip()

    text = _clean(text)
    print(f"[STT] Hasil: {text}")

    return {
        "text": text,
        "language": language or "auto",
        "no_speech_prob": 0.0,
        "segments": [],
    }


def _clean(text: str) -> str:
    """Bersihkan artefak output whisper.cpp."""
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\d{2}:\d{2}[.:]\d{3}\s*-->\s*\d{2}:\d{2}[.:]\d{3}", "", text)
    return " ".join(text.split())


# ── Test langsung ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        test_audio = ROOT_DIR / "data" / "corpus" / "audio" / "2222_audio2.wav"
        hasil = transcribe(str(test_audio))
    else:
        hasil = transcribe(sys.argv[1])
    print("\n=== HASIL ===")
    print(f"Teks   : {hasil['text']}")
    print(f"Bahasa : {hasil['language']}")