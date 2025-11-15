# train_gemini_merged.py

from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import numpy as np

orig = pd.read_csv("sample_for_gemini.csv")  # (or adjust to your actual train file)
gemini = pd.read_csv("gemini_augmented.csv")
df = pd.concat([orig, gemini], ignore_index=True)

X = df["sentence"].tolist()
y = df["label"].values

model = SentenceTransformer('stsb-xlm-r-multilingual', device='cpu')
X_vec = model.encode(X, batch_size=32, show_progress_bar=True)
X_tensor = torch.tensor(X_vec, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=768, n_classes=2):
        super().__init__()
        self.fc = nn.Linear(input_dim, n_classes)
    def forward(self, x):
        return self.fc(x)

clf = SimpleClassifier(input_dim=X_train.shape[1], n_classes=int(y_tensor.max())+1)
optimizer = torch.optim.Adam(clf.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

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

print("Done training with merged original and Gemini augmented data.")
