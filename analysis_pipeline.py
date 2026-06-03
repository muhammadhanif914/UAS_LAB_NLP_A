"""
analisis_pipeline.py
Pipeline STT → Text Processing → LLM → TTS untuk seluruh audio korpus.

Alur per audio:
  1. STT          → transkrip disimpan ke File 1 (transcript_USER.jsonl)
  2. Text Process → normalisasi + deteksi bahasa (kolom baru di File 1)
  3. LLM          → respons disimpan sebagai kolom baru di File 1
  4. TTS          → teks fonem → audio WAV disimpan ke output_tts/

Output:
  - data/corpus/transcripts/transcript_{user}.jsonl  ← transkrip + LLM per user
  - data/corpus/output_tts/tts_{nama}.wav            ← audio TTS
  - log/hasil_analisis.json                          ← ringkasan semua audio
  - log/hasil_analisis.csv                           ← transkrip + LLM dalam CSV

Jalankan dari root proyek:
    python analisis_pipeline.py
"""

import csv
import json
import time
from pathlib import Path
from datetime import datetime

from app.stt import transcribe
from app.llm import generate_response, MODEL
from app.tts import synthesize
from app.utils import (
    normalize_text,
    detect_languages,
    segment_by_language,
    text_to_phoneme,
    save_transcript,
    log_pipeline_result,
    Timer,
)

# ── Konfigurasi ────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent
AUDIO_DIR   = ROOT_DIR / "data" / "corpus" / "audio"
OUTPUT_DIR  = ROOT_DIR / "data" / "corpus" / "output_tts"
RESULT_FILE = ROOT_DIR / "log" / "hasil_analisis.json"
CSV_FILE    = ROOT_DIR / "log" / "hasil_analisis.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(ROOT_DIR / "log").mkdir(exist_ok=True)

DELAY_BETWEEN_AUDIO = 3  # detik, hindari rate limit LLM

# Kolom CSV — sesuai format output yang diinginkan
CSV_FIELDS = [
    "file", "utterance_id", "stt_transcript", "llm_response",
]


# ── Simpan CSV (dipanggil tiap audio selesai) ─────────────────────────────────
def _save_csv(results: list) -> None:
    """Tulis ulang CSV dengan semua hasil yang ada sejauh ini."""
    rows = []
    for r in results:
        filename     = r.get("audio_input", "")
        # utterance_id: bagian setelah _ dan sebelum .wav, contoh: 2128_audio1.wav -> audio1
        utterance_id = filename.replace(".wav", "").split("_", 1)[-1] if "_" in filename else filename
        rows.append({
            "file":         filename,
            "utterance_id": utterance_id,
            "stt_transcript": r.get("stt", {}).get("normalized_text", ""),
            "llm_response":   r.get("llm", {}).get("response_text", ""),
        })
    if not rows:
        return
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  ✅ CSV diperbarui → {CSV_FILE.name} ({len(rows)} baris)")


# ── Proses satu file audio ─────────────────────────────────────────────────────
def proses_satu_audio(audio_path: Path) -> dict:
    filename = audio_path.name
    user_id  = filename.split("_")[0] if "_" in filename else "unknown"

    print(f"\n{'='*55}")
    print(f"  File : {filename}  |  User : {user_id}")
    print(f"{'='*55}")

    result = {
        "audio_input": filename,
        "user_id":     user_id,
        "timestamp":   datetime.now().isoformat(),
        "stt":         {},
        "lang_info":   {},
        "llm":         {},
        "tts":         {},
        "latency":     {},
        "error":       None,
    }

    total_start = time.perf_counter()

    try:
        # ── Step 1: STT ────────────────────────────────────────────────
        print("\n[1/3] STT — Transkripsi audio...")
        with Timer() as t:
            stt_raw = transcribe(str(audio_path))

        raw_text   = stt_raw.get("text", "")
        normalized = normalize_text(raw_text)
        result["stt"] = {
            "raw_text":        raw_text,
            "normalized_text": normalized,
            "language":        stt_raw.get("language", ""),
            "latency_s":       round(t.elapsed, 3),
        }
        result["latency"]["stt_s"] = round(t.elapsed, 3)
        print(f"    ✔ Teks    : {normalized}")
        print(f"    ✔ Latency : {t.elapsed:.2f}s")

        # ── Step 2: Text Processing ────────────────────────────────────
        lang_info = detect_languages(normalized)
        segments  = segment_by_language(normalized)
        result["lang_info"] = {
            "languages_detected": lang_info["languages_detected"],
            "is_code_switching":  lang_info["is_code_switching"],
            "lang_counts":        lang_info["lang_counts"],
            "segments":           segments,
        }
        print(f"    ✔ Bahasa  : {lang_info['languages_detected']}")
        print(f"    ✔ CS      : {lang_info['is_code_switching']}")

        # ── Step 3: LLM ────────────────────────────────────────────────
        print("\n[2/3] LLM — Generate respons Gemini...")
        with Timer() as t:
            response_text = generate_response(normalized)

        result["llm"] = {
            "response_text": response_text,
            "model":         MODEL,
            "latency_s":     round(t.elapsed, 3),
        }
        result["latency"]["llm_s"] = round(t.elapsed, 3)
        print(f"    ✔ Respons : {response_text[:80]}{'...' if len(response_text) > 80 else ''}")
        print(f"    ✔ Latency : {t.elapsed:.2f}s")

        # ── Step 4: TTS ────────────────────────────────────────────────
        print("\n[3/3] TTS — Sintesis suara...")
        tts_text = text_to_phoneme(response_text)
        print(f"    ✔ Fonem   : {tts_text[:80]}{'...' if len(tts_text) > 80 else ''}")

        tts_out = OUTPUT_DIR / f"tts_{filename}"
        with Timer() as t:
            synthesize(tts_text, str(tts_out))

        result["tts"] = {
            "input_text":  tts_text,
            "output_file": str(tts_out),
            "latency_s":   round(t.elapsed, 3),
        }
        result["latency"]["tts_s"] = round(t.elapsed, 3)
        print(f"    ✔ Output  : {tts_out.name}")
        print(f"    ✔ Latency : {t.elapsed:.2f}s")

        # Simpan transkrip lengkap sekali di akhir (STT + LLM + TTS)
        save_transcript(
            user_id=user_id, audio_filename=filename,
            raw_text=raw_text, normalized_text=normalized,
            lang_info=lang_info, segments=segments,
            llm_response=response_text,
            tts_text=tts_text,
        )

        # Total latency
        total = round(time.perf_counter() - total_start, 3)
        result["latency"]["total_s"] = total
        print(f"\n  ✅ Selesai | Total latency: {total}s")

        # Log pipeline
        log_pipeline_result(
            audio_filename=filename, stt_result=stt_raw,
            llm_response=response_text, tts_output_path=str(tts_out),
            tts_input_text=tts_text, latency=total,
            lang_info=lang_info,
            response_lang_info=detect_languages(response_text),
        )

    except Exception as e:
        result["error"]              = str(e)
        result["latency"]["total_s"] = round(time.perf_counter() - total_start, 3)
        print(f"\n  ❌ ERROR: {e}")

    return result


# ── Ringkasan statistik ────────────────────────────────────────────────────────
def _ringkasan(results: list) -> dict:
    sukses    = [r for r in results if not r["error"]]
    gagal     = [r for r in results if r["error"]]
    latencies = [r["latency"].get("total_s", 0) for r in sukses]
    cs_count  = sum(1 for r in sukses if r["lang_info"].get("is_code_switching"))
    return {
        "total_audio":             len(results),
        "berhasil":                len(sukses),
        "gagal":                   len(gagal),
        "avg_latency_s":           round(sum(latencies) / len(latencies), 3) if latencies else None,
        "min_latency_s":           round(min(latencies), 3) if latencies else None,
        "max_latency_s":           round(max(latencies), 3) if latencies else None,
        "code_switching_detected": cs_count,
        "error_files":             [r["audio_input"] for r in gagal],
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*55)
    print("  ANALISIS PIPELINE — Voice CS System")
    print("="*55)

    audio_files = sorted(AUDIO_DIR.glob("*.wav"))
    if not audio_files:
        print(f"\n❌ Tidak ada file .wav di: {AUDIO_DIR}")
        return

    print(f"\n  Ditemukan : {len(audio_files)} file audio")
    print(f"  Output TTS: {OUTPUT_DIR}")
    print(f"  Log       : {RESULT_FILE}")

    results = []
    for i, audio_path in enumerate(audio_files, 1):
        print(f"\n[Audio {i}/{len(audio_files)}]")
        results.append(proses_satu_audio(audio_path))

        # Simpan CSV setiap audio selesai — aman jika pipeline error di tengah
        _save_csv(results)

        if i < len(audio_files):
            print(f"  Jeda {DELAY_BETWEEN_AUDIO}s...")
            time.sleep(DELAY_BETWEEN_AUDIO)

    # Ringkasan akhir
    ringkasan = _ringkasan(results)
    print("\n" + "="*55)
    print("  RINGKASAN HASIL")
    print("="*55)
    print(f"  Total audio    : {ringkasan['total_audio']}")
    print(f"  Berhasil       : {ringkasan['berhasil']}")
    print(f"  Gagal          : {ringkasan['gagal']}")
    print(f"  Avg latency    : {ringkasan['avg_latency_s']}s")
    print(f"  Code-switching : {ringkasan['code_switching_detected']} audio")
    if ringkasan["error_files"]:
        print(f"  File error     : {ringkasan['error_files']}")
    print("="*55)

    # Simpan JSON
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "run_timestamp": datetime.now().isoformat(),
            "ringkasan":     ringkasan,
            "detail":        results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n  ✅ JSON      → {RESULT_FILE.name}")
    print(f"  ✅ CSV       → {CSV_FILE.name}")
    print(f"  ✅ Audio TTS → {OUTPUT_DIR.name}/")
    print(f"  ✅ Transkrip → data/corpus/transcripts/")


if __name__ == "__main__":
    main()