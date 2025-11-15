from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

languages = ['en', 'cn', 'id']
max_samples = 800  # Adjust as you wish (None or a value < total rows for speed)

model = SentenceTransformer('stsb-xlm-r-multilingual', device='cpu')

for lang in languages:
    path = f'datasets/{lang}/train_100.tsv'  # Or use dev.tsv/test.tsv if preferred!
    df = pd.read_csv(path, sep='\t', names=['label', 'text'])
    if max_samples:
        df = df.sample(n=min(max_samples, len(df)), random_state=0)
    print(f"Encoding {len(df)} samples for {lang.upper()}...")
    embeddings = model.encode(df['text'].tolist(), batch_size=32, show_progress_bar=True)
    np.save(f'saved/{lang}_train_enc.npy', embeddings)
    np.save(f'saved/{lang}_train_labels.npy', df['label'].values)
    print(f"✅  Saved: saved/{lang}_train_enc.npy and saved/{lang}_train_labels.npy")

print("Done for all languages!")
