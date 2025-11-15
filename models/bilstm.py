import torch
import torch.nn.functional as F
from torch import nn
from sklearn import metrics
import pytorch_lightning as pl
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BiLSTM(pl.LightningModule):
    def __init__(self, embedding_dim, hidden_dim, num_labels, dropout, lr):
        super().__init__()
        self.lr = lr
        self.num_labels = num_labels

        self.rnn = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=1,
            bidirectional=True, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim * 2, num_labels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.save_hyperparameters()

    def forward(self, x):
        # x shape: (batch_size, embedding_dim)
        # reshape to (batch_size, seq_len=1, embedding_dim)
        x = x.unsqueeze(1)
        embedded = self.dropout(x)
        output, (hidden, cell) = self.rnn(embedded)
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        hidden_cat = self.dropout(hidden_cat)
        return self.fc(self.relu(hidden_cat))  # shape: [batch_size, num_labels]

    def training_step(self, batch, batch_idx):
        x = batch['text']
        y = batch['label']
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['text']
        y = batch['label']
        y_hat = self(x)
        preds = torch.argmax(y_hat, dim=-1)
        loss = F.cross_entropy(y_hat, y)
        self.validation_step_outputs.append({'y': y.cpu(), 'y_hat': preds.cpu(), 'loss': loss.item()})
        self.log('val_loss', loss)
        return loss

    def on_validation_epoch_end(self):
        y_true, y_pred = [], []
        for out in self.validation_step_outputs:
            y_true.extend(out['y'].numpy())
            y_pred.extend(out['y_hat'].numpy())
        acc = metrics.accuracy_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred, average='macro' if self.num_labels > 2 else 'binary')
        mcc = metrics.matthews_corrcoef(y_true, y_pred)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)
        self.log('val_mcc', mcc, prog_bar=True)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        x = batch['text']
        y = batch['label']
        y_hat = self(x)
        preds = torch.argmax(y_hat, dim=-1)
        self.test_step_outputs.append({'y': y.cpu(), 'y_hat': preds.cpu()})

    def on_test_epoch_end(self):
        y_true, y_pred = [], []
        for out in self.test_step_outputs:
            y_true.extend(out['y'].numpy())
            y_pred.extend(out['y_hat'].numpy())
        acc = metrics.accuracy_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred, average='macro' if self.num_labels > 2 else 'binary')
        mcc = metrics.matthews_corrcoef(y_true, y_pred)
        self.log('test_acc', acc)
        self.log('test_f1', f1)
        self.log('test_mcc', mcc)
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', patience=2, factor=0.8
            ),
            'monitor': 'val_loss'
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
