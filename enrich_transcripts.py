"""
enrich_transcripts.py
Lanjutkan dari transcripts_clean.csv → LLM → TTS → 4 file output

Output:
    1. log/transcripts_enriched.csv
    2. log/transcript_{user}.jsonl
    3. log/hasil_analisis.jsonl
    4. log/output_tts/tts_NAMA.wav
"""

import sys
import os

# ── PENTING (Windows): import torch/TTS LEBIH AWAL sebelum library lain
# agar DLL c10.dll berhasil diinisialisasi sebelum library audio lain (soundfile, dll)
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

try:
    import torch  # pre-load torch agar c10.dll sudah ready
except Exception:
    pass

from app.tts import synthesize as _tts_synthesize   # pre-load TTS singleton

# ── Baru import library lain setelah torch/TTS siap ──────────────────────────
import csv
import json
import time
import shutil
import pandas as pd
from datetime import datetime

from app.llm   import generate_response
from app.utils import text_to_phoneme, save_transcript, detect_languages, segment_by_language

# ─────────────────────────────────────────────────────────────────────────────
INPUT_CSV      = ROOT_DIR / "data" / "corpus" / "transcripts" / "transcripts_clean.csv"
LOG_DIR        = ROOT_DIR / "log"
OUTPUT_CSV     = LOG_DIR / "transcripts_enriched.csv"
TRANSCRIPT_DIR = LOG_DIR
TTS_OUT_DIR    = LOG_DIR / "output_tts"
HASIL_JSONL    = LOG_DIR / "hasil_analisis.jsonl"
TEMP_DIR       = ROOT_DIR / "temp"

LOG_DIR.mkdir(parents=True, exist_ok=True)
TTS_OUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

DELAY = 6


# ── TTS ───────────────────────────────────────────────────────────────────────
def run_tts(text: str, output_path: str) -> float:
    """Jalankan TTS langsung via fungsi yang sudah di-pre-load."""
    t0 = time.perf_counter()

    out = _tts_synthesize(text, output_path)

    elapsed = round(time.perf_counter() - t0, 3)

    # Pindahkan file jika belum ada di target
    final = Path(output_path)
    if not final.exists():
        for candidate in [
            TEMP_DIR / Path(output_path).name,
            TEMP_DIR / f"tts_{Path(output_path).stem}.wav",
        ]:
            if candidate.exists():
                shutil.move(str(candidate), str(final))
                break
        if isinstance(out, str) and Path(out).exists() and not final.exists():
            shutil.move(out, str(final))

    return elapsed


# ── Helper simpan CSV ─────────────────────────────────────────────────────────
def simpan_csv(rows: list):
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"   💾 CSV → {OUTPUT_CSV.name} ({len(rows)} baris)")


# ── Helper simpan JSONL ───────────────────────────────────────────────────────
def simpan_jsonl(entry: dict):
    with open(HASIL_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"   💾 Log → hasil_analisis.jsonl")


def get_user_id(filename: str) -> str:
    parts = Path(filename).stem.split("_", 1)
    return parts[0] if len(parts) == 2 else "unknown"

def get_utterance_id(filename: str) -> str:
    parts = Path(filename).stem.split("_", 1)
    return parts[1] if len(parts) == 2 else Path(filename).stem


# ── Proses satu baris ─────────────────────────────────────────────────────────
def proses_satu(row: dict) -> dict:
    filename     = str(row.get("audio_input", row.get("file", ""))).strip()
    text         = str(row.get("text", row.get("stt_transcript",
                   row.get("normalized_text", "")))).strip()
    user_id      = get_user_id(filename)
    utterance_id = get_utterance_id(filename)

    print(f"\n{'─'*55}")
    print(f"  File : {filename} | User: {user_id}")
    print(f"  STT  : {text[:70]}")
    print(f"{'─'*55}")

    out = {
        **row,
        "utterance_id":         utterance_id,
        "user_id":              user_id,
        "timestamp":            datetime.now().isoformat(),
        "llm_response":         "",
        "llm_response_phoneme": "",
        "tts_output_file":      "",
        "latency_llm_s":        0.0,
        "latency_tts_s":        0.0,
        "error":                "",
    }

    lang_info = detect_languages(text)
    segments  = segment_by_language(text)
    out["detected_languages"] = ", ".join(lang_info["languages_detected"])
    out["is_code_switching"]  = lang_info["is_code_switching"]

    # ── LLM ──────────────────────────────────────────────────────────────────
    print(f"\n  [1/2] LLM")
    try:
        t0       = time.perf_counter()
        response = generate_response(text)
        out["latency_llm_s"] = round(time.perf_counter() - t0, 3)
        out["llm_response"]  = response
        print(f"   ✅ ({out['latency_llm_s']}s) {response[:70]}")
    except Exception as e:
        out["error"] = f"LLM error: {e}"
        print(f"   ❌ {e}")
        _simpan_semua(out, user_id, filename, text, lang_info, segments)
        return out

    # ── Fonem ─────────────────────────────────────────────────────────────────
    print(f"\n  [2/2] Fonem → TTS")
    phoneme = text_to_phoneme(response)
    out["llm_response_phoneme"] = phoneme
    print(f"   ✅ Fonem: {phoneme[:70]}")

    # ── TTS ───────────────────────────────────────────────────────────────────
    tts_filename = f"tts_{Path(filename).stem}.wav"
    tts_target   = TTS_OUT_DIR / tts_filename

    try:
        elapsed = run_tts(phoneme, str(tts_target))
        out["latency_tts_s"]   = elapsed
        out["tts_output_file"] = tts_filename
        print(f"   ✅ TTS ({elapsed}s) → log/output_tts/{tts_filename}")
    except Exception as e:
        out["error"] = f"TTS error: {e}"
        print(f"   ❌ TTS error: {e}")

    _simpan_semua(out, user_id, filename, text, lang_info, segments)
    return out


def _simpan_semua(out, user_id, filename, text, lang_info, segments):
    # File 2: transcript_{user}.jsonl → log/
    try:
        save_transcript(
            user_id=user_id,
            audio_filename=filename,
            raw_text=text,
            normalized_text=out.get("normalized_text", text),
            lang_info=lang_info,
            segments=segments,
            llm_response=out.get("llm_response", ""),
            tts_text=out.get("llm_response_phoneme", ""),
            pos_tags=out.get("pos_tags", ""),
        )
        print(f"   💾 Transkrip → log/transcript_{user_id}.jsonl")
    except Exception as e:
        print(f"   ⚠️  Gagal simpan transkrip: {e}")

    # File 3: hasil_analisis.jsonl → log/
    simpan_jsonl({
        "timestamp":          out.get("timestamp"),
        "audio_input":        filename,
        "user_id":            user_id,
        "utterance_id":       out.get("utterance_id"),
        "stt_text":           text,
        "detected_languages": out.get("detected_languages"),
        "is_code_switching":  out.get("is_code_switching"),
        "llm_response":       out.get("llm_response"),
        "tts_output_file":    out.get("tts_output_file"),
        "latency_llm_s":      out.get("latency_llm_s"),
        "latency_tts_s":      out.get("latency_tts_s"),
        "error":              out.get("error"),
    })


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*55}")
    print("  ENRICH: CSV → LLM → TTS → 4 Output")
    print(f"{'='*55}")

    if not INPUT_CSV.exists():
        print(f"\n❌ File tidak ditemukan: {INPUT_CSV}")
        return

    df    = pd.read_csv(INPUT_CSV)
    total = len(df)
    print(f"\n  Input  : {INPUT_CSV.name} ({total} baris)")
    print(f"  ⚠️  Auto-save aktif: hasil langsung tersimpan per baris!\n")

    # Resume: skip yang sudah berhasil
    sudah = set()
    if HASIL_JSONL.exists():
        with open(HASIL_JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        e = json.loads(line)
                        if not e.get("error"):
                            sudah.add(e.get("audio_input", ""))
                    except Exception:
                        pass
    if sudah:
        print(f"  ⏭️  {len(sudah)} audio sudah diproses sebelumnya, akan di-skip.\n")

    semua_rows = []
    berhasil = gagal = 0

    for i, row in df.iterrows():
        filename = str(row.get("audio_input", row.get("file", ""))).strip()

        if filename in sudah:
            print(f"\n[{i+1}/{total}] ⏭️  Skip: {filename}")
            semua_rows.append(row.to_dict())
            continue

        print(f"\n[{i+1}/{total}]", end="")
        hasil = proses_satu(row.to_dict())
        semua_rows.append(hasil)

        if hasil.get("error"):
            gagal += 1
        else:
            berhasil += 1

        # File 1: simpan CSV per baris
        simpan_csv(semua_rows)

        if i < total - 1:
            print(f"\n  ⏳ Jeda {DELAY}s...")
            time.sleep(DELAY)

    print(f"\n{'='*55}")
    print(f"  SELESAI | Berhasil: {berhasil} | Gagal: {gagal} | Skip: {len(sudah)}")
    print(f"{'='*55}")
    print(f"\n📂 Output:")
    print(f"  1. {OUTPUT_CSV}")
    print(f"  2. {TRANSCRIPT_DIR}/transcript_{{user}}.jsonl")
    print(f"  3. {HASIL_JSONL}")
    print(f"  4. {TTS_OUT_DIR}/tts_*.wav")


if __name__ == "__main__":
    main()