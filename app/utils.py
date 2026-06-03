"""
app/utils.py
Fungsi utilitas: normalisasi, deteksi bahasa, POS tagging,
segment_by_language, text_to_phoneme, save_transcript, log_pipeline_result, Timer
"""

import os
import re
import json
import time
from pathlib import Path
from datetime import datetime

# ── Path ──────────────────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).resolve().parent.parent
LOG_DIR        = ROOT_DIR / "log"
TRANSCRIPT_DIR = LOG_DIR                          # ← semua transcript ke log/
LOG_DIR.mkdir(exist_ok=True)
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

# ── Kosakata ──────────────────────────────────────────────────────────────────
_ID_WORDS = {
    "dan","yang","ini","itu","dengan","untuk","adalah","di","ke","dari",
    "tidak","pada","juga","akan","sudah","bisa","kita","saya","kami",
    "mereka","ada","atau","tapi","karena","jadi","agar","bila","kalau",
    "mau","aku","kamu","dia","bagaimana","berapa","siapa","kapan",
    "dimana","mengapa","apakah","sangat","lebih","setelah","sebelum",
    "semua","setiap","banyak","tentang","antara","dalam","anda","bantu",
    "minta","jelaskan","proses","cara","secara","benar","lah","pun",
}
_EN_WORDS = {
    "the","is","are","and","this","that","with","for","not","have","can",
    "will","but","from","they","we","you","it","at","be","do","was","were",
    "how","what","when","where","who","why","my","your","our","their",
    "get","help","book","flight","hotel","visa","apply","arrange","prepare",
    "explain","schedule","transport","simple","include","budget","online",
    "step","by","next","week","tomorrow","today","want","need","just","also",
}
_AR_CHARS = set("ابتثجحخدذرزسشصضطظعغفقكلمنهويءأإآةىئؤ")
_AR_WORDS = {
    "insya","allah","alhamdulillah","bismillah","masya","subhanallah",
    "wallahi","yalla","habibi","ila","min","umrah","haji",
    "makkah","madinah","jeddah",
}

_POS_ID = {
    "saya":"PRON","aku":"PRON","kamu":"PRON","dia":"PRON","kami":"PRON",
    "kita":"PRON","mereka":"PRON","anda":"PRON",
    "adalah":"VERB","ada":"VERB","bisa":"VERB","mau":"VERB","akan":"VERB",
    "sudah":"VERB","minta":"VERB","bantu":"VERB","jelaskan":"VERB",
    "dan":"CONJ","atau":"CONJ","tapi":"CONJ","karena":"CONJ","agar":"CONJ",
    "di":"PREP","ke":"PREP","dari":"PREP","untuk":"PREP","dengan":"PREP",
    "pada":"PREP","dalam":"PREP","tentang":"PREP",
    "bagaimana":"WH","berapa":"WH","siapa":"WH","kapan":"WH",
    "dimana":"WH","mengapa":"WH","apakah":"WH","apa":"WH",
    "tidak":"ADV","juga":"ADV","sangat":"ADV","lebih":"ADV","hanya":"ADV",
    "ini":"DET","itu":"DET","yang":"REL","semua":"DET",
}
_POS_EN = {
    "i":"PRON","you":"PRON","he":"PRON","she":"PRON","we":"PRON",
    "they":"PRON","it":"PRON","my":"PRON","your":"PRON","our":"PRON",
    "is":"VERB","are":"VERB","was":"VERB","have":"VERB","can":"VERB",
    "will":"VERB","get":"VERB","help":"VERB","book":"VERB","apply":"VERB",
    "prepare":"VERB","explain":"VERB","arrange":"VERB","want":"VERB",
    "and":"CONJ","or":"CONJ","but":"CONJ","so":"CONJ",
    "in":"PREP","at":"PREP","on":"PREP","to":"PREP","for":"PREP",
    "from":"PREP","with":"PREP","by":"PREP","of":"PREP",
    "the":"DET","a":"DET","an":"DET","this":"DET","that":"DET",
    "what":"WH","when":"WH","where":"WH","who":"WH","how":"WH",
    "not":"ADV","also":"ADV","very":"ADV","just":"ADV","now":"ADV",
}
_POS_AR = {
    "insya":"PART","allah":"NOUN","alhamdulillah":"INTJ",
    "bismillah":"INTJ","masya":"PART","subhanallah":"INTJ",
    "wallahi":"PART","yalla":"ADV","habibi":"NOUN",
    "ila":"PREP","min":"PREP","umrah":"NOUN","haji":"NOUN",
    "makkah":"NOUN","madinah":"NOUN","jeddah":"NOUN",
}


# ── 1. Normalisasi ────────────────────────────────────────────────────────────
def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"[^\w\s\'\-\.,!?]", " ", text)
    text = re.sub(r"\s+", " ", text)
    for wrong, correct in {
        "inshaallah":"insya allah","inshallah":"insya allah",
        "insyallah":"insya allah","alhamdulilah":"alhamdulillah",
    }.items():
        text = text.replace(wrong, correct)
    return text.strip()


# ── 2. Deteksi bahasa ─────────────────────────────────────────────────────────
def detect_languages(text: str) -> dict:
    words  = text.lower().split()
    counts = {"id": 0, "en": 0, "ar": 0, "unknown": 0}
    for word in words:
        clean = re.sub(r"[^\w]", "", word)
        if not clean:
            continue
        if any(c in _AR_CHARS for c in clean) or clean in _AR_WORDS:
            counts["ar"] += 1
        elif clean in _ID_WORDS or clean in _POS_ID:
            counts["id"] += 1
        elif clean in _EN_WORDS or clean in _POS_EN:
            counts["en"] += 1
        else:
            counts["unknown"] += 1
    detected = [l for l in ("id","en","ar") if counts[l] > 0]
    return {
        "languages_detected": detected,
        "is_code_switching":  len(detected) > 1,
        "lang_counts":        counts,
    }


# ── 3. POS Tagging ────────────────────────────────────────────────────────────
def pos_tagging(text: str) -> str:
    """Tandai setiap kata: kata/LANG_POS. Contoh: aku/ID_PRON book/EN_VERB"""
    words, tagged = text.lower().split(), []
    for word in words:
        clean = re.sub(r"[^\w]", "", word)
        if not clean:
            continue
        if any(c in _AR_CHARS for c in clean) or clean in _AR_WORDS:
            lang, pos = "AR", _POS_AR.get(clean, "NOUN")
        elif clean in _ID_WORDS or clean in _POS_ID:
            lang, pos = "ID", _POS_ID.get(clean, "NOUN")
        elif clean in _EN_WORDS or clean in _POS_EN:
            lang, pos = "EN", _POS_EN.get(clean, "NOUN")
        else:
            lang, pos = "ID", "NOUN"
        tagged.append(f"{clean}/{lang}_{pos}")
    return " ".join(tagged)


# ── 4. Segment by language ────────────────────────────────────────────────────
def segment_by_language(text: str) -> list:
    """
    Pisahkan teks menjadi segmen berdasarkan bahasa.
    Contoh: [{"lang":"id","text":"aku mau"},{"lang":"en","text":"book flight"}]
    """
    words = text.lower().split()
    segments = []
    current_lang = None
    current_words = []

    for word in words:
        clean = re.sub(r"[^\w]", "", word)
        if not clean:
            continue
        if any(c in _AR_CHARS for c in clean) or clean in _AR_WORDS:
            lang = "ar"
        elif clean in _ID_WORDS or clean in _POS_ID:
            lang = "id"
        elif clean in _EN_WORDS or clean in _POS_EN:
            lang = "en"
        else:
            lang = current_lang or "id"

        if lang != current_lang:
            if current_words:
                segments.append({"lang": current_lang, "text": " ".join(current_words)})
            current_lang  = lang
            current_words = [clean]
        else:
            current_words.append(clean)

    if current_words:
        segments.append({"lang": current_lang, "text": " ".join(current_words)})

    return segments


# ── 5. Text to Phoneme ────────────────────────────────────────────────────────
def text_to_phoneme(text: str) -> str:
    """Konversi teks ke fonem menggunakan phonemizer + espeak-ng."""
    try:
        from phonemizer import phonemize
        result = phonemize(
            text,
            backend="espeak",
            language="id",
            with_stress=True,
            language_switch="remove-flags",
        )
        return result
    except Exception as e:
        print(f"[UTILS] ⚠️  phonemizer gagal ({e}), pakai teks asli.")
        return text


# ── 6. Save transcript ────────────────────────────────────────────────────────
def save_transcript(
    user_id: str,
    audio_filename: str,
    raw_text: str,
    normalized_text: str,
    lang_info: dict,
    segments: list,
    llm_response: str = "",
    tts_text: str = "",
    pos_tags: str = "",
):
    """
    Simpan/update satu entri transkrip ke log/transcript_{user_id}.jsonl
    """
    filepath = TRANSCRIPT_DIR / f"transcript_{user_id}.jsonl"

    entry = {
        "filename":           audio_filename,
        "user_id":            user_id,
        "timestamp":          datetime.now().isoformat(),
        "stt_raw":            raw_text,
        "normalized_text":    normalized_text,
        "detected_languages": lang_info.get("languages_detected", []),
        "is_code_switching":  lang_info.get("is_code_switching", False),
        "lang_counts":        lang_info.get("lang_counts", {}),
        "segments":           segments,
        "pos_tags":           pos_tags,
        "llm_response":       llm_response,
        "tts_phoneme_input":  tts_text,
    }

    # Baca existing, update jika sudah ada
    existing = []
    if filepath.exists():
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        existing.append(json.loads(line))
                    except Exception:
                        pass

    updated = False
    for i, e in enumerate(existing):
        if e.get("filename") == audio_filename:
            existing[i] = entry
            updated = True
            break
    if not updated:
        existing.append(entry)

    with open(filepath, "w", encoding="utf-8") as f:
        for e in existing:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


# ── 7. Log pipeline result ────────────────────────────────────────────────────
def log_pipeline_result(
    audio_filename: str,
    stt_result: dict,
    llm_response: str,
    tts_output_path: str,
    tts_input_text: str,
    latency: float,
    lang_info: dict = None,
    response_lang_info: dict = None,
):
    """Append satu entri log pipeline ke log/pipeline_log.jsonl"""
    log_file = LOG_DIR / "pipeline_log.jsonl"
    entry = {
        "timestamp":        datetime.now().isoformat(),
        "audio_input":      audio_filename,
        "latency_total_s":  round(latency, 3),
        "stt_text":         stt_result.get("text", ""),
        "llm_response":     llm_response,
        "tts_output":       os.path.basename(tts_output_path) if tts_output_path else "",
        "tts_phoneme":      tts_input_text[:100],
        "lang_info":        lang_info or {},
        "response_lang":    response_lang_info or {},
    }
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── 8. Timer ──────────────────────────────────────────────────────────────────
class Timer:
    def __enter__(self):
        self._start = time.perf_counter()
        return self
    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = "Aku mau book flight ke Jeddah, insya Allah minggu depan."
    print("Normalized :", normalize_text(sample))
    print("Lang detect:", detect_languages(normalize_text(sample)))
    print("POS tags   :", pos_tagging(normalize_text(sample)))
    print("Segments   :", segment_by_language(normalize_text(sample)))