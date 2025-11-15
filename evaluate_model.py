from models.bilstm import BiLSTM
from utils.dataset import load_dataset
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd

# Step 1: Load encoder and dataset
encoder = SentenceTransformer('stsb-xlm-r-multilingual', device='cpu')
datasets = load_dataset(encoder, dataset_name='en', sample=1.0, aug=False)

# Step 2: Load trained model
model_path = "E:/LiDA/saved/local_run.ckpt"


model = BiLSTM.load_from_checkpoint(model_path)
model.eval()

# Step 3: Run inference
texts = datasets['test']['text']
labels = datasets['test']['labels']

predictions = []
with torch.no_grad():
    for x in texts:
        pred = torch.argmax(model(x.unsqueeze(0))).item()
        predictions.append(pred)

# Step 4: Save results to CSV
df = pd.DataFrame({'true_label': labels, 'predicted_label': predictions})
df.to_csv("E:/LiDA/saved/test_predictions_all.csv", index=False)
print("✅ Predictions saved to saved/test_predictions1.csv")
