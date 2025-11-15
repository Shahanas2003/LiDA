from argparse import ArgumentParser
from utils.my_datamodule import MyDataModule
from utils.dataset import load_dataset
from models.bilstm import BiLSTM
from ae.ae import AutoEncoder

from sentence_transformers import SentenceTransformer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import wandb
from pytorch_lightning.loggers import WandbLogger

import torch
import time
import os


def main(hparams):
    # Safe seed setup
    pl.seed_everything(1)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Optional WandB login
    wandb_mode = "disabled"  # change to "online" if you want to log results
    if wandb_mode != "disabled":
        try:
            wandb.login(host="http://localhost:8081")
        except:
            print("WandB login skipped (offline mode).")

    run = wandb.init(
        project=hparams.project_name or hparams.dataset,
        mode=wandb_mode,
        save_code=True
    )

    wandb_logger = WandbLogger(log_model=False) if wandb_mode != "disabled" else None
    if wandb_mode != "disabled":
        wandb.config.update(hparams)

    print(f"Augmenting: {hparams.augmenting}")
    print(f"SBERT model: {hparams.sbert}")

    encoder = SentenceTransformer(hparams.sbert, device=str(device))

    # ----------------- Load Dataset -----------------
    start = time.time()
    datasets = load_dataset(
        encoder,
        hparams.dataset,
        hparams.sample,
        hparams.augmenting,
        hparams.ae_model,
        hparams.ae_hidden,
        hparams.aug_number,
        hparams.da_model,
        hparams.aug_type,
        hparams.backtrans,
        hparams.eda_aug
    )
    end = time.time()
    print(f"Dataset loaded in {end - start:.2f} seconds")

    print(f"Train samples: {len(datasets['train']['text'])}")
    num_labels = len(set(datasets['train']['labels']))
    print(f"Detected Labels: {num_labels}")

    # ----------------- Data Module -----------------
    dm = MyDataModule(datasets, hparams.batch_size)

    # ----------------- Model -----------------
    model = BiLSTM(768, hparams.hidden_dim, num_labels, hparams.dropout, hparams.lr).to(device)

    # ----------------- Callbacks -----------------
    filename = f"{wandb.run.name if wandb_mode != 'disabled' else 'local_run'}"
    checkpoint_callback = ModelCheckpoint(
        dirpath='saved/',
        monitor='val_mcc',
        mode='max',
        filename=filename
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # ----------------- Trainer -----------------
    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator="cpu" if not torch.cuda.is_available() else "gpu",
        devices=1,
        max_epochs=hparams.epochs,
        deterministic=True,
        callbacks=[checkpoint_callback, lr_monitor],
        num_sanity_val_steps=0  # skip initial validation check
    )

    # ----------------- Train/Test -----------------
    print("Starting Training...")
    trainer.fit(model, dm)

    print("Evaluating on test data...")
    trainer.test(datamodule=dm)
    print(f"Best model checkpoint score: {checkpoint_callback.best_model_score}")

    print("Training complete.")
    if wandb_mode != "disabled":
        run.finish()


if __name__ == "__main__":
    parser = ArgumentParser()

    # Model/Embedding parameters
    parser.add_argument("--sbert", type=str, default="stsb-xlm-r-multilingual")
    parser.add_argument("--model", type=str, default="bilstm")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--hidden_dim", type=int, default=500)

    # Augmentation / dataset
    parser.add_argument("--dataset", type=str, default="en")
    parser.add_argument("--sample", type=float, default=0.1)
    parser.add_argument("--augmenting", type=bool, default=False)
    parser.add_argument("--ae_model", type=str, default=None)
    parser.add_argument("--da_model", type=str, default=None)
    parser.add_argument("--ae_hidden", type=int, default=None)
    parser.add_argument("--aug_number", type=float, default=0.0)
    parser.add_argument("--aug_type", type=str, default=None)
    parser.add_argument("--backtrans", type=bool, default=False)
    parser.add_argument("--eda_aug", type=bool, default=False)

    # Logging/specifications
    parser.add_argument("--project_name", type=str, default="LiDA-CPU-Test")

    args = parser.parse_args()

    # Run Main
    main(args)
