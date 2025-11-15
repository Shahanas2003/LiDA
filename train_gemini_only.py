# train_gemini_only.py

from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import numpy as np

# Load Gemini augmentation data
df = pd.read_csv("gemini_augmented.csv")
X = df["sentence"].tolist()
y = df["label"].values

# SBERT Embeddings
model = SentenceTransformer('stsb-xlm-r-multilingual', device='cpu')
X_vec = model.encode(X, batch_size=32, show_progress_bar=True)
X_tensor = torch.tensor(X_vec, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Split for tiny validation/test (just for demonstration)
X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Simple one-layer classifier
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=768, n_classes=2):
        super().__init__()
        self.fc = nn.Linear(input_dim, n_classes)

    def forward(self, x):
        return self.fc(x)

clf = SimpleClassifier(input_dim=X_train.shape[1], n_classes=int(y_tensor.max())+1)
optimizer = torch.optim.Adam(clf.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Quick train loop
for epoch in range(10):
    clf.train()
    optimizer.zero_grad()
    logits = clf(X_train)
    loss = criterion(logits, y_train)
    loss.backward()
    optimizer.step()

    clf.eval()
    with torch.no_grad():
        val_logits = clf(X_val)
        val_preds = torch.argmax(val_logits, dim=1)
        acc = (val_preds == y_val).float().mean().item()
    print(f"Epoch {epoch+1}: Val Acc {acc:.3f}, Loss {loss.item():.3f}")

print("Done training on Gemini-only augmented data.")
