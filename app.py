import streamlit as st
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

# --- Import your model class ---
from models.bilstm import BiLSTM  # Edit if your model file is elsewhere

label_map = {
    0: "Negative",
    1: "Positive"
}

st.title("LiDA Multilingual Sentiment Classifier")

# --- Model checkpoint --- 
ckpt_path = "saved/local_run-v3.ckpt"  # Change as needed
model = BiLSTM.load_from_checkpoint(ckpt_path)
model.eval()

# --- Load SBERT ---
sbert = SentenceTransformer("stsb-xlm-r-multilingual")

# Main app input
st.sidebar.header("Settings")
language = st.sidebar.selectbox("Language", ["English", "Chinese", "Indonesian"])
text_input = st.text_area("Enter a sentence for sentiment classification:")

if st.button("Predict"):
    emb = sbert.encode([text_input])
    emb_tensor = torch.tensor(emb, dtype=torch.float32)
    with torch.no_grad():
        logits = model(emb_tensor)
        probs = torch.softmax(logits, dim=1).numpy()[0]
        pred = int(np.argmax(probs))
        label_name = label_map.get(pred, "Unknown")
        conf = float(np.max(probs))

    st.write(f"**Predicted Sentiment:** {label_name}")
    st.write(f"**Confidence:** {conf:.2f}")
