import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add_noise(emb, frac, noise_type='zero'):
    length = len(emb)
    if noise_type == 'zero':
        r = np.random.randint(0, length, int(length * frac))
        for i in r:
            emb[i] = 0.
    elif noise_type == 'gaussian':
        mean_emb = torch.mean(emb).to(device)
        std_emb = torch.std(emb).to(device) + 2
        noise = torch.normal(mean_emb, std_emb, size=(1, emb.shape[-1])).to(device)
        emb = emb + noise
    return emb.to(device)


class MyDataset(Dataset):
    def __init__(self, text1, text2):
        self.text1 = text1
        self.text2 = text2

    def __len__(self):
        return len(self.text1)

    def __getitem__(self, idx):
        return {'text1': self.text1[idx], 'text2': self.text2[idx]}


class MyDataModule(pl.LightningDataModule):
    def __init__(self, datasets, encoder, batch_size, denoising=False):
        super().__init__()
        self.train1 = datasets['train1'].to(device)
        self.train2 = datasets['train2'].to(device)
        self.val1 = datasets['val1'].to(device)
        self.val2 = datasets['val2'].to(device)
        if denoising:
            self.train1 = add_noise(self.train1, 0.5, 'gaussian')
            self.train2 = self.train1
            self.val2 = self.val1

        self.train_dataset = MyDataset(self.train1, self.train2)
        self.val_dataset = MyDataset(self.val1, self.val2)
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=4)


class AutoEncoder(pl.LightningModule):
    def __init__(self, embedding_dim, hidden_dim, lr):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, x):
        with torch.set_grad_enabled(True):
            x = x.to(device)
            z = self.encoder(x)
            x_hat = self.decoder(z)
        return x_hat

    def training_step(self, batch, batch_idx):
        x = batch['text1']
        y = batch['text2']
        x_hat = self(x)
        loss = F.mse_loss(x_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['text1']
        y = batch['text2']
        x_hat = self(x)
        loss = F.mse_loss(x_hat, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.9)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }
