import lightning.pytorch as pl
import torch
from monai.data import decollate_batch
from monai.inferers import SliceInferer
from monai.optimizers import Novograd
from monai.transforms import Compose
from torchmetrics import Accuracy, AUROC, F1Score, Precision, Recall
from efficientnet_pytorch_3d import EfficientNet3D
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from models.model_original import threeDClassModel


class LightningModel(pl.LightningModule):
    def __init__(
        self,
        in_channels=1,
        num_classes=2,
        lr=1e-3,
    ):
        super().__init__()

        # self.model = EfficientNet3D.from_name(
        #     "efficientnet-b0",
        #     override_params={"num_classes": num_classes},
        #     in_channels=in_channels,
        # )
        self.model = threeDClassModel(
            input_size=in_channels, num_classes=num_classes
        )

        self.loss = BCEWithLogitsLoss()

        self.accuracy = Accuracy("binary")
        self.auroc = AUROC(task="binary")
        self.f1 = F1Score(task="binary")
        self.precision = Precision(task="binary")
        self.recall = Recall(task="binary")

        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()

        y_hat = self(x).squeeze()
        loss = self.loss(y_hat, y)
        acc = self.accuracy(y_hat, y.int())
        auroc = self.auroc(y_hat, y.int())
        f1 = self.f1(y_hat, y.int())
        precision = self.precision(y_hat, y.int())
        recall = self.recall(y_hat, y.int())
        self.log_dict(
            {
                "train_loss": loss,
                "train_acc": acc,
                "train_auroc": auroc,
                "train_f1": f1,
                "train_precision": precision,
                "train_recall": recall,
            },
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()
        y_hat = self(x).squeeze()
        loss = self.loss(y_hat, y)
        acc = self.accuracy(y_hat, y.int())
        auroc = self.auroc(y_hat, y.int())
        f1 = self.f1(y_hat, y.int())
        precision = self.precision(y_hat, y.int())
        recall = self.recall(y_hat, y.int())
        self.log_dict(
            {
                "val_loss": loss,
                "val_acc": acc,
                "val_auroc": auroc,
                "val_f1": f1,
                "val_precision": precision,
                "val_recall": recall,
            },
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            logger=True,
            sync_dist=True,
        )

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()
        y_hat = self(x).squeeze()
        loss = self.loss(y_hat, y)
        acc = self.accuracy(y_hat, y.int())
        auroc = self.auroc(y_hat, y.int())
        f1 = self.f1(y_hat, y.int())
        precision = self.precision(y_hat, y.int())
        recall = self.recall(y_hat, y.int())
        self.log_dict(
            {
                "test_loss": loss,
                "test_acc": acc,
                "test_auroc": auroc,
                "test_f1": f1,
                "test_precision": precision,
                "test_recall": recall,
            },
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            logger=True,
            sync_dist=True,
        )
        return loss

    # def predict_step(self, batch, batch_idx, dataloader_idx=None):
    #     x = batch["image"]
    #     inferer = SliceInferer(
    #         roi_size=(self.in_shape[2], self.in_shape[3]),
    #         spatial_dim=2,
    #         progress=False,
    #     )
    #     y_hat = inferer(x, self.model)
    #     # #### THIS DOES NOT WORK ON OPENNEURO
    #     # THIS IS DUE TO SOME PROBLEMS WITH INVERSION OF
    #     # AFFINE TRANSFORMS. THIS VERSION RUNS ON 3D VOLUMES ONLY FROM
    #     # THE DATASET PROVIDED BY THE CHALLENGE.
    #     # #####
    #     batch_copied = batch.copy()
    #     batch_copied["pred"] = y_hat
    #     batch_copied = [
    #         self.predict_transforms(i) for i in decollate_batch(batch_copied)
    #     ]
    #     return y_hat

    def configure_optimizers(self):
        optimizer = Novograd(
            self.parameters(), lr=self.lr, amsgrad=True, weight_decay=0.001
        )
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=5,
            verbose=True,
            threshold=0.001,
            min_lr=1e-6,
        )
        return optimizer
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "monitor": "val_loss",
        #     },
        # }
