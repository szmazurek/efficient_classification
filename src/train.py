import torch
from models.lightning_module import LightningModel
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)

from lightning.pytorch.strategies.ddp import DDPStrategy

torch.set_float32_matmul_precision("medium")


def train_lightning(train_loader, val_loader, args):
    input_channels = next(iter(train_loader))[0].shape[1]
    num_classes = args.num_classes
    model = LightningModel(
        in_channels=input_channels, num_classes=num_classes, lr=args.lr
    )
    model_checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        mode="min",
        monitor="val_loss",
        dirpath="checkpoints/",
        filename="best_model",
    )

    strategy = DDPStrategy(
        find_unused_parameters=False,
        static_graph=True,
    )
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed",
        strategy=strategy,
        log_every_n_steps=1,
        max_epochs=args.epochs,
        callbacks=model_checkpoint_callback,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, train_loader, val_loader)
    process_rank = int(trainer.global_rank)
    return model_checkpoint_callback.best_model_path, process_rank
