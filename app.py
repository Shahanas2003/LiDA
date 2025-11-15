import streamlit as st
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

# --- Import your model class (edit if needed) ---
from models.bilstm import BiLSTM  # Edit if your model file is elsewhere

label_map = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

st.title("LiDA Multilingual Sentiment Classifier Demo (Debug Mode)")

# --- Model checkpoint --- 
ckpt_path = "saved/local_run-v2.ckpt"  # Change as needed
model = BiLSTM.load_from_checkpoint(ckpt_path)
model.eval()

# --- Load SBERT (no device arg, solves the error) ---
sbert = SentenceTransformer("stsb-xlm-r-multilingual")

st.sidebar.header("Settings")
language = st.sidebar.selectbox("Language", ["English", "Chinese", "Indonesian"])
text_input = st.text_area("Enter a sentence for sentiment classification:")

if st.button("Show SBERT embedding"):
    emb = sbert.encode([text_input])
    st.write("SBERT embedding shape:", np.array(emb).shape)
    st.write("First 10 values:", emb[0][:10])

if st.button("Predict"):
    emb = sbert.encode([text_input])
    st.write("SBERT embedding (first 10):", emb[0][:10])
    emb_tensor = torch.tensor(emb, dtype=torch.float32)
    with torch.no_grad():
        logits = model(emb_tensor)
        st.write("Raw logits:", logits.numpy())
        probs = torch.softmax(logits, dim=1).numpy()[0]
        st.write("Softmax probabilities:", probs)
        pred = int(np.argmax(probs))
        conf = float(np.max(probs))
        label_name = label_map.get(pred, "Unknown")

    st.write(f"**Predicted class:** {pred} ({label_name})")
    st.write(f"**Confidence:** {conf:.3f}")
    st.write(f"Per-class probabilities: {probs}")

    # Debug info: check if probabilities are almost always neutral
    if np.allclose(probs, 1.0/len(probs), atol=0.1):
        st.warning("Warning: All probabilities nearly equal! Your model might not be trained or is always guessing one class.")

