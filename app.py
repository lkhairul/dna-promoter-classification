import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import gradio as gr
 
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "backend", "model", "promoter_cnn_model.h5")
 
print(f"🔄 Sedang memuat model CNN dari: {MODEL_PATH}...")
try:
    model = load_model(MODEL_PATH)
    print("✅ BERHASIL: Model CNN sukses dimuat ke folder utama!")
except Exception as e:
    print(f"❌ Gagal memuat model. Pastikan file ada di backend/model/. Detail: {e}")
 
def one_hot_encode(sequence):
    mapping = {
        'A': [1, 0, 0, 0], 
        'T': [0, 1, 0, 0], 
        'G': [0, 0, 1, 0], 
        'C': [0, 0, 0, 1]
    }
    sequence = sequence.upper()
    encoded = np.array([mapping[nucleotide] for nucleotide in sequence])
    return encoded
 
def predict_promoter(sequence):
    sequence = sequence.upper().strip()
    
    if len(sequence) != 57:
        raise ValueError("DNA sequence must contain exactly 57 nucleotides.")
        
    encoded_sequence = one_hot_encode(sequence)
    encoded_sequence = np.expand_dims(encoded_sequence, axis=0)
    
    prediction = model.predict(encoded_sequence, verbose=0)[0][0]
    
    if prediction > 0.5:
        label = "Promoter"
        confidence = prediction
    else:
        label = "Non-Promoter"
        confidence = 1 - prediction
        
    return {
        "prediction": label,
        "confidence": round(float(confidence), 4)
    }
 
 
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Mono:wght@400;500&display=swap');
 
/* === GLOBAL === */
body {
    background-color: #0f1117 !important;
    font-family: 'DM Sans', sans-serif !important;
}
.gradio-container {
    background-color: #0f1117 !important;
    border: none !important;
    font-family: 'DM Sans', sans-serif !important;
    max-width: 1000px !important;
    margin: 0 auto !important;
    padding: 0 !important;
}
 
/* === HEADER === */
#title-text {
    background: #1a3a5c !important;
    border-bottom: 1px solid #1e4976 !important;
    padding: 32px 40px 26px !important;
    margin-bottom: 28px !important;
    text-align: center;
    border-radius: 0 !important;
}
#title-text h1 {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1.85rem !important;
    font-weight: 600 !important;
    color: #f0f6ff !important;
    letter-spacing: -0.3px !important;
    margin-bottom: 6px !important;
}
#title-text p {
    font-size: 13px !important;
    color: #7ea8d8 !important;
    font-weight: 400 !important;
    font-family: 'DM Sans', sans-serif !important;
}
 
/* === INFO CARD === */
.info-card {
    background: #0d1e33;
    padding: 12px 16px;
    border-radius: 0 6px 6px 0;
    border-left: 3px solid #2563eb;
    color: #94b4d4;
    font-size: 13.5px;
    line-height: 1.65;
    font-family: 'DM Sans', sans-serif;
}
 
/* === TEXTAREA === */
.dna-input textarea {
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
    letter-spacing: 1.5px !important;
    background-color: #0d1117 !important;
    color: #7dd3fc !important;
    border: 1px solid #1e2d42 !important;
    border-radius: 6px !important;
    line-height: 1.7 !important;
}
.dna-input textarea:focus {
    border-color: #2563eb !important;
    outline: none !important;
}
.dna-input label span {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    letter-spacing: 1.2px !important;
    color: #60a5fa !important;
    text-transform: uppercase !important;
}
 
/* === BUTTONS === */
.submit-btn, button.primary {
    background: #1d4ed8 !important;
    border: 1px solid #2563eb !important;
    color: #eff6ff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    border-radius: 6px !important;
    letter-spacing: 0.2px !important;
}
.submit-btn:hover, button.primary:hover {
    background: #2563eb !important;
}
button.secondary {
    background: #161b27 !important;
    border: 1px solid #1e2d42 !important;
    color: #7ea8d8 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    border-radius: 6px !important;
}
button.secondary:hover {
    background: #1e2d42 !important;
}
 
/* === OUTPUT BOX === */
.output-box {
    background-color: #161b27 !important;
    border-radius: 10px !important;
    border: 1px solid #1e2d42 !important;
    font-family: 'DM Sans', sans-serif !important;
    min-height: 180px;
}
 
/* === FOOTER === */
.gr-markdown p, .prose p {
    font-family: 'DM Sans', sans-serif !important;
    color: #2d4a6b !important;
    font-size: 12px !important;
}
"""
 
def grad_predict(sequence):
    try:
        result = predict_promoter(sequence)
        label = result["prediction"]    
        conf = result["confidence"]    
        
        if "non" in str(label).lower():
            icon = "🔴"
            status = "NON-PROMOTER (-)"
            color = "#ef4444"
        else:
            icon = "🟢"
            status = "PROMOTER (+)"
            color = "#22c55e"
            
        return f"""
        <div style='text-align: center; padding: 20px;'>
            <h2 style='color: {color}; margin-bottom: 10px;'>{icon} {status}</h2>
            <p style='font-size: 1.5rem; color: white;'>Confidence Score: <br>
            <b style='font-size: 2.5rem; color: #2dd4bf;'>{conf:.2%}</b></p>
        </div>
        """
    except Exception as e:
        return f"<div style='color: #f87171; padding: 20px;'>❌ Error: {str(e)}<br><small>Pastikan panjang sekuens pas 57 karakter (A, C, G, T).</small></div>"
 
with gr.Blocks(css=custom_css) as demo:
    with gr.Column(elem_id="title-text"):
        gr.HTML("""
            <div style='font-size:11px; font-weight:500; letter-spacing:2.5px; text-transform:uppercase; color:#60a5fa; margin-bottom:10px;'>
                Genomic Analysis Platform
            </div>
            <h1>🧬 DNA Insight: Promoter Classifier</h1>
        """)
        gr.Markdown("Deep Learning-based Genomic Sequence Analysis · CNN Model")
 
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("<div class='info-card'><b style='color:#93c5fd;'>Instruksi:</b> Masukkan 57 karakter sekuens DNA (A, C, G, T). Model CNN akan mendeteksi apakah sekuens tersebut merupakan area promoter.</div>")
            input_dna = gr.Textbox(
                lines=3, 
                placeholder="Contoh: CTCGTCCTCAATGGCCTCTAAACGGGTCTTGAGGGGTTTTTTGCTG...", 
                label="DNA Sequence (57bp)", 
                elem_classes="dna-input"
            )
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear")
                submit_btn = gr.Button("🚀 Analyze Sequence", variant="primary", elem_classes="submit-btn")
        
        with gr.Column(scale=1):
            output_html = gr.HTML(
                label="Analysis Result", 
                elem_classes="output-box", 
                value="<div style='text-align: center; color: #4b5563; padding: 40px;'>Menunggu input sekuens...</div>"
            )
 
    submit_btn.click(fn=grad_predict, inputs=input_dna, outputs=output_html)
    clear_btn.click(lambda: [None, "<div style='text-align: center; color: #4b5563; padding: 40px;'>Menunggu input sekuens...</div>"], outputs=[input_dna, output_html])
 
    gr.Markdown("---")
    gr.HTML("<div style='text-align: center; color: #2d4a6b; font-family: DM Sans, sans-serif; font-size: 12px;'>Kelompok 3 Tugas Besar KDS © 2026</div>")
 
if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Soft())