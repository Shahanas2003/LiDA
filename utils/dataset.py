import pandas as pd
import numpy as np
import torch
from ae.ae import AutoEncoder
from sentence_transformers import SentenceTransformer, util
import jieba
from utils.eda import eda

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(encoder=None, dataset_name='en', sample=1.0,
                 aug=False, ae_model=None, ae_hidden=768, aug_num=0,
                 da_model=None, aug_type=None, backtrans=False, eda_aug=False):
    sample = int(sample * 100)
    dataset_path = f'datasets/{dataset_name}/'

    if backtrans:
        train_dataset = pd.read_csv(
            dataset_path + f'train_{sample}_backtrans.tsv',
            sep='\t', names=['labels', 'text']
        )[["text", "labels"]]
    else:
        train_dataset = pd.read_csv(
            dataset_path + f'train_{sample}.tsv',
            sep='\t', names=['labels', 'text']
        )[["text", "labels"]]

        if eda_aug:
            aug_data = {'text': [], 'labels': []}
            for i in range(len(train_dataset)):
                text = train_dataset.iloc[i]["text"]
                label = train_dataset.iloc[i]["labels"]
                if dataset_name == 'cn':
                    text_jb = jieba.lcut(text)
                    text = " ".join(text_jb)
                    eda_text = eda(text, alpha_sr=0, alpha_ri=0)
                else:
                    eda_text = eda(text)
                for j in eda_text:
                    aug_data['text'].append(j)
                    aug_data['labels'].append(label)
            train_dataset = pd.concat([train_dataset, pd.DataFrame(aug_data)], ignore_index=True)

    val_dataset = pd.read_csv(dataset_path + 'dev.tsv',
                              sep='\t', names=['labels', 'text'])[['text', 'labels']]
    test_dataset = pd.read_csv(dataset_path + 'test.tsv',
                               sep='\t', names=['labels', 'text'])[['text', 'labels']]

    # Encode text into SBERT embeddings
    train_enc = encoder.encode(train_dataset.text.values, convert_to_tensor=True, batch_size=16)
    val_enc = encoder.encode(val_dataset.text.values, convert_to_tensor=True, batch_size=16)
    test_enc = encoder.encode(test_dataset.text.values, convert_to_tensor=True, batch_size=16)

    train_labels = train_dataset.labels.values
    val_labels = val_dataset.labels.values
    test_labels = test_dataset.labels.values

    if aug:
        print(f"Using augmentation: {aug_type}")
        if aug_type == "linear":
            train_aug = linear(train_enc, aug_num)
        elif aug_type == "ae":
            train_aug = autoencoder(train_enc, ae_model, ae_hidden)
        elif aug_type == "da":
            train_aug = denoising_ae(train_enc, da_model, ae_hidden)
        elif aug_type == "all":
            train_aug1 = linear(train_enc, aug_num)
            train_aug2 = autoencoder(train_enc, ae_model, ae_hidden)
            train_aug3 = denoising_ae(train_enc, da_model, ae_hidden)
            train_aug = torch.cat((train_aug1, train_aug2, train_aug3), 0)
            train_labels = np.concatenate((train_labels, train_labels, train_labels, train_labels))
        else:
            raise ValueError("Invalid augmentation type!")

        train_enc = torch.cat((train_enc, train_aug), 0)
        train_labels = np.concatenate((train_labels, train_labels))

    return {
        "train": {"text": train_enc, "labels": train_labels},
        "val": {"text": val_enc, "labels": val_labels},
        "test": {"text": test_enc, "labels": test_labels}
    }


def linear(embedding, aug_num):
    return embedding + aug_num


# FIXED AutoEncoder augmentation
def autoencoder(embedding, model, hidden):
    ae = AutoEncoder.load_from_checkpoint(
        f'ae/best/ae-quora-den-{model}.ckpt',
        embedding_dim=768,
        hidden_dim=hidden,
        lr=1e-4
    ).to(device)
    ae.eval()

    # Disable gradients for inference under Lightning 2.x
    with torch.no_grad():
        emb_clone = embedding.clone().detach().to(device)
        augmented = ae(emb_clone).detach().to(device)
    return augmented


# FIXED Denoising AutoEncoder
def denoising_ae(embedding, model, hidden):
    da = AutoEncoder.load_from_checkpoint(
        f'ae/best/ae-quora-den-{model}.ckpt',
        embedding_dim=768,
        hidden_dim=hidden,
        lr=1e-4
    ).to(device)
    da.eval()

    with torch.no_grad():
        emb_clone = embedding.clone().detach().to(device)
        augmented = da(emb_clone).detach().to(device)
    return augmented
