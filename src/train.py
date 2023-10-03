import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies.ddp import DDPStrategy
from models.lightning_module import LightningModel

torch.set_float32_matmul_precision("medium")


def train_lightning(train_loader, args):
    input_channels = next(iter(train_loader))[0].shape[1]
    num_classes = args.num_classes
    model = LightningModel(
        in_channels=input_channels, num_classes=num_classes, lr=args.lr
    )
    model_checkpoint_callback = ModelCheckpoint(
        save_last=True,
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
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, train_loader)
    process_rank = int(trainer.global_rank)
    return model_checkpoint_callback.best_model_path, process_rank
