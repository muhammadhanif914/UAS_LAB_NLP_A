import os
import requests
import tempfile
import gradio as gr

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://127.0.0.1:8000/voice-chat"
)

CSS = """
.hero {
    text-align: center;
    padding: 25px 0 10px 0;
}
.hero h1 {
    font-size: 2.8rem;
    margin-bottom: 10px;
    font-weight: 700;
}
.hero p {
    opacity: 0.8;
    font-size: 1rem;
}
.footer {
    text-align: center;
    opacity: 0.65;
    padding: 20px;
    font-size: 0.9rem;
}
.pipeline-box {
    border-radius: 14px;
}
"""


def voice_chat(audio_input):
    if audio_input is None:
        return (
            "### ⚠️ Tidak ada audio\n\nSilakan rekam atau upload audio terlebih dahulu.",
            None,
        )

    try:
        with open(audio_input, "rb") as f:
            response = requests.post(
                BACKEND_URL,
                files={"file": ("audio.wav", f, "audio/wav")},
                timeout=300,
            )

        if response.status_code != 200:
            try:
                error_msg = response.json().get("detail", response.text)
            except Exception:
                error_msg = response.text

            return (
                f"### ❌ Pipeline Error\n\n{error_msg}",
                None,
            )

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(response.content)
        tmp.close()

        return (
            "### ✅ Pipeline Selesai\n\n"
            "- ✅ Audio diterima\n"
            "- ✅ Speech-to-Text berhasil\n"
            "- ✅ LLM berhasil menghasilkan respons\n"
            "- ✅ Text-to-Speech berhasil\n\n"
            "Respons audio siap diputar.",
            tmp.name,
        )

    except requests.exceptions.ConnectionError:
        return (
            "### ❌ Backend Tidak Aktif\n\n"
            "Jalankan backend terlebih dahulu:\n\n"
            "```bash\nuvicorn app.main:app --host 0.0.0.0 --port 8000\n```",
            None,
        )

    except requests.exceptions.Timeout:
        return (
            "### ❌ Timeout\n\nProses terlalu lama. Silakan coba lagi.",
            None,
        )

    except Exception as e:
        return (
            f"### ❌ Error\n\n{str(e)}",
            None,
        )


theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
)

with gr.Blocks(theme=theme, css=CSS, title="Voice AI Assistant") as demo:

    gr.HTML(
        """
        <div class="hero">
            <h1>🎙️ Voice AI Assistant</h1>
            <p>
                Speech To Text → LLM → Text To Speech<br>
                Indonesian • English • Arabic
            </p>
        </div>
        """
    )

    with gr.Row():

        with gr.Column(scale=2):
            gr.Markdown("## 🎤 Input Audio")

            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Rekam atau Upload Audio"
            )

            btn = gr.Button(
                "🚀 Process Audio",
                variant="primary",
                size="lg"
            )

        with gr.Column(scale=3):
            gr.Markdown("## 📊 Pipeline Status")

            status_box = gr.Markdown(
                "### Menunggu Input\n\nSilakan rekam suara atau upload file audio."
            )

    gr.Markdown("---")
    gr.Markdown("## 🔊 AI Voice Response")

    audio_output = gr.Audio(
        label="Generated Response",
        autoplay=True,
        type="filepath"
    )

    with gr.Accordion("ℹ️ Cara Menggunakan", open=False):
        gr.Markdown(
            "### Langkah Penggunaan\n\n"
            "1. Klik **Record from Microphone**\n"
            "2. Rekam suara Anda atau upload file audio\n"
            "3. Klik **Process Audio**\n"
            "4. Sistem akan menjalankan:\n"
            "   - Speech-to-Text\n"
            "   - Large Language Model\n"
            "   - Text-to-Speech\n"
            "5. Dengarkan hasil respons pada bagian output\n\n"
            "---\n\n"
            "### Backend\n\n"
            "Pastikan backend berjalan:\n\n"
            "```bash\nuvicorn app.main:app --host 0.0.0.0 --port 8000\n```"
        )

    gr.HTML(
        """
        <div class="footer">
            Voice Code-Switching System 
        </div>
        """
    )

    btn.click(
        fn=voice_chat,
        inputs=[audio_input],
        outputs=[status_box, audio_output],
    )


if __name__ == "__main__":
    demo.launch(
        server_port=7860,
        share=True,
    )

