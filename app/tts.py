"""
tts.py - Text-to-Speech module menggunakan model Indonesian-TTS dari Wikidepia
Arsitektur: VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech)
Model: https://github.com/Wikidepia/indonesian-tts
"""

import os
import re
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Konfigurasi path model
# ─────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, "coqui_utils", "checkpoint_1260000-inference.pth")
CONFIG_PATH   = os.path.join(BASE_DIR, "coqui_utils", "config.json")
SPEAKERS_PATH = os.path.join(BASE_DIR, "coqui_utils", "speakers.pth")
SPEAKER_IDX   = "wibowo"   # speaker default (tersedia di speakers.pth)

# Singleton agar model hanya dimuat sekali
_tts_instance = None


def _get_tts():
    """
    Memuat model TTS dan mengembalikannya sebagai singleton.
    Model hanya akan dimuat satu kali selama aplikasi berjalan.
    """
    global _tts_instance

    if _tts_instance is not None:
        return _tts_instance

    # Import di sini agar error import tidak menghentikan seluruh aplikasi
    from TTS.api import TTS

    logger.info("[TTS] Memuat model Indonesian-TTS dari Wikidepia...")
    print("[TTS] Memuat model, harap tunggu...")

    _tts_instance = TTS(
        model_path=MODEL_PATH,
        config_path=CONFIG_PATH,
        progress_bar=False,
        gpu=False          # Ganti True jika menggunakan GPU
    )

    logger.info("[TTS] Model berhasil dimuat.")
    print("[TTS] Model siap digunakan.")
    return _tts_instance


# ─────────────────────────────────────────────
# Konversi teks ke fonem sederhana (opsional)
# ─────────────────────────────────────────────
def _clean_text(text: str) -> str:
    """
    Membersihkan teks dari karakter yang tidak didukung model.
    Model VITS ini menggunakan input fonemik, sehingga beberapa
    karakter khusus perlu dihapus atau diganti.
    """
    # Hapus karakter selain huruf, spasi, tanda baca dasar
    text = re.sub(r"[^\w\s,.!?'-]", "", text)
    # Normalisasi spasi berlebih
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────
# Fungsi utama sintesis suara
# ─────────────────────────────────────────────
def synthesize(text: str, output_path: str, speaker: str = SPEAKER_IDX) -> str:
    """
    Mengkonversi teks menjadi file audio WAV.

    Args:
        text        : Teks yang akan disintesis (Bahasa Indonesia).
        output_path : Path file output WAV, contoh: "output/response.wav"
        speaker     : Nama speaker yang digunakan (default: "wibowo")

    Returns:
        Path file audio yang dihasilkan.

    Raises:
        FileNotFoundError : Jika file model tidak ditemukan.
        RuntimeError      : Jika proses sintesis gagal.
    """
    # Validasi file model
    for label, path in [("Model", MODEL_PATH),
                        ("Config", CONFIG_PATH),
                        ("Speakers", SPEAKERS_PATH)]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[TTS] File {label} tidak ditemukan: {path}\n"
                "Pastikan file model sudah ada di folder app/coqui_utils/"
            )

    # Bersihkan teks
    cleaned = _clean_text(text)
    if not cleaned:
        raise ValueError("[TTS] Teks kosong setelah pembersihan, sintesis dibatalkan.")

    print(f"[TTS] Mensintesis: {cleaned[:60]}{'...' if len(cleaned) > 60 else ''}")
    logger.info(f"[TTS] Mensintesis teks: {cleaned[:60]}")

    # Buat folder output jika belum ada
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Proses TTS
    try:
        tts = _get_tts()
        # Gunakan synthesizer langsung karena versi terbaru TTS API
        # tidak support tts_to_file dengan model lokal multi-speaker
        wav = tts.synthesizer.tts(
            text=cleaned,
            speaker_name=speaker,
        )
        tts.synthesizer.save_wav(wav=wav, path=output_path)
    except Exception as e:
        logger.error(f"[TTS] Gagal mensintesis: {e}")
        raise RuntimeError(f"[TTS] Proses sintesis gagal: {e}") from e

    logger.info(f"[TTS] File audio disimpan di: {output_path}")
    print(f"[TTS] Selesai. File disimpan: {output_path}")
    return output_path


# ─────────────────────────────────────────────
# Test mandiri
# ─────────────────────────────────────────────
if __name__ == "__main__":
    out = synthesize(
        "Halo, sistem text-to-speech siap diunakan.",
        "test2_tts.wav"
    )
    print(f"Output: {out}")