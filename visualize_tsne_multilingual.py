import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

plt.style.use('seaborn-v0_8')

languages = ['en', 'cn', 'id']
language_names = {'en': 'English', 'cn': 'Chinese', 'id': 'Indonesian'}

for lang in languages:
    enc = np.load(f'saved/{lang}_train_enc.npy')
    labels = np.load(f'saved/{lang}_train_labels.npy')
    
    print(f"Running t-SNE for {language_names[lang]} ({lang})...")
    tsne = TSNE(n_components=2, random_state=0, perplexity=30)
    emb_2d = tsne.fit_transform(enc)

    plt.figure(figsize=(7, 6))
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap('Set2', len(unique_labels))
    for i, lab in enumerate(unique_labels):
        plt.scatter(emb_2d[labels == lab, 0], emb_2d[labels == lab, 1], 
                    s=30, alpha=0.7, label=f"Class {lab}", color=colors(i))
    plt.legend()
    plt.title(f"t-SNE of SBERT Embeddings — {language_names[lang]}")
    plt.tight_layout()
    plt.savefig(f'saved/tsne_{lang}_train.png', dpi=300)
    plt.close()
    print(f"✅ Plot saved: saved/tsne_{lang}_train.png")

print("All t-SNE plots generated!")
