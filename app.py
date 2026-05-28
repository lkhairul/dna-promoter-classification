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
body { background-color: #0b0f19; }
.gradio-container { background-color: #0b0f19 !important; border: none !important; }
#title-text { text-align: center; color: #2dd4bf; margin-bottom: 20px; }
#title-text h1 { font-family: 'Poppins', sans-serif; font-weight: 700; font-size: 2.5rem; }
.dna-input textarea { 
    font-family: 'Courier New', monospace !important; 
    font-size: 1.2rem !important; 
    letter-spacing: 2px;
    background-color: #111827 !important;
    color: #2dd4bf !important;
    border: 1px solid #374151 !important;
}
.submit-btn { 
    background: linear-gradient(90deg, #0d9488 0%, #2dd4bf 100%) !important; 
    border: none !important; 
    color: white !important;
    font-weight: bold !important;
}
.output-box { 
    background-color: #111827 !important; 
    border-radius: 15px !important; 
    border: 1px solid #374151 !important; 
    font-size: 1.3rem !important;
}
.info-card {
    background-color: #1e293b;
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid #2dd4bf;
    color: #cbd5e1;
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
        gr.HTML("<h1>🧬 DNA Insight: Promoter Classifier</h1>")
        gr.Markdown("Deep Learning-based Genomic Sequence Analysis Platform")

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("<div class='info-card'><b>Instruksi:</b> Masukkan 57 karakter sekuens DNA (A, C, G, T). Model CNN akan mendeteksi apakah sekuens tersebut merupakan area promoter.</div>")
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
    gr.HTML("<div style='text-align: center; color: #64748b;'>Kelompok Tugas Besar KDS © 2026</div>")

if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Soft())