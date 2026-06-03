
import os
import time
from google import genai
from google.genai import types
from pydantic import TypeAdapter
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "gemma-4-26b-a4b-it").strip()
if not MODEL.startswith("models/"):
    MODEL = f"models/{MODEL}"

# ── Load API keys ─────────────────────────────────────────────────────────────
def _load_api_keys() -> list[str]:
    keys = []
    k = os.getenv("GEMINI_API_KEY", "").strip()
    if k:
        keys.append(k)
    i = 1
    while True:
        k = os.getenv(f"GEMINI_API_KEY_{i}", "").strip()
        if not k:
            break
        if k not in keys:
            keys.append(k)
        i += 1
    if not keys:
        raise EnvironmentError("[LLM] Tidak ada GEMINI_API_KEY ditemukan di .env")
    return keys

API_KEYS          = _load_api_keys()
current_key_index = 0
print(f"[LLM] Model    : {MODEL}")
print(f"[LLM] API Keys : {len(API_KEYS)} key tersedia")

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_INSTRUCTION = "Jawab selalu dalam Bahasa Indonesia baku, tanpa markdown, 2-3 kalimat singkat."

chat_config     = types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION)
history_adapter = TypeAdapter(list[types.Content])

def _make_client(idx: int) -> genai.Client:
    return genai.Client(api_key=API_KEYS[idx])

def _make_chat(client: genai.Client):
    return client.chats.create(model=MODEL, config=chat_config)

client = _make_client(current_key_index)
chat   = _make_chat(client)

# ── Fungsi utama ──────────────────────────────────────────────────────────────
def generate_response(prompt: str) -> str:
    global current_key_index, client, chat
    max_retries = len(API_KEYS)
    for attempt in range(max_retries):
        try:
            print(f"  [LLM] Mencoba key [{current_key_index + 1}]...")
            response = chat.send_message(prompt)
            return response.text.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err or "rate" in err.lower():
                print(f"  [LLM] Key [{current_key_index + 1}] rate limit! Ganti key berikutnya...")
                current_key_index = (current_key_index + 1) % len(API_KEYS)
                client = _make_client(current_key_index)
                chat   = _make_chat(client)
                print(f"  [LLM] Pakai key [{current_key_index + 1}], tunggu 5 detik...")
                time.sleep(5)
            elif "403" in err or "PERMISSION_DENIED" in err:
                print(f"  [LLM] Key [{current_key_index + 1}] ditolak! Ganti key berikutnya...")
                current_key_index = (current_key_index + 1) % len(API_KEYS)
                client = _make_client(current_key_index)
                chat   = _make_chat(client)
            else:
                return f"[ERROR] {err}"
    return "[ERROR] Semua API key habis quota atau gagal."

# ── Test langsung ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        "Aku mau book flight ke Jeddah minggu depan, bisa bantu schedule?",
        "What time is the prayer? Saya mau sholat dulu.",
        "Tolong jelaskan apa itu machine learning.",
    ]
    for teks in test_cases:
        print(f"\nInput : {teks}")
        print(f"Output: {generate_response(teks)}")