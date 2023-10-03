import lightning.pytorch as pl
from models.model_original import threeDClassModel
from monai.optimizers import Novograd
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torchmetrics import AUROC, Accuracy, F1Score, Precision, Recall


class LightningModel(pl.LightningModule):
    def __init__(
        self,
        in_channels=1,
        num_classes=2,
        lr=1e-3,
    ):
        super().__init__()

        self.model = threeDClassModel(
            input_size=in_channels, num_classes=num_classes
        )
        self.loss = (
            BCEWithLogitsLoss() if num_classes == 1 else CrossEntropyLoss()
        )

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
        # acc = self.accuracy(y_hat, y.int())
        # auroc = self.auroc(y_hat, y.int())
        # f1 = self.f1(y_hat, y.int())
        # precision = self.precision(y_hat, y.int())
        # recall = self.recall(y_hat, y.int())
        # self.log_dict(
        #     {
        #         "train_loss": loss,
        #         "train_acc": acc,
        #         "train_auroc": auroc,
        #         "train_f1": f1,
        #         "train_precision": precision,
        #         "train_recall": recall,
        #     },
        #     on_epoch=True,
        #     on_step=True,
        #     prog_bar=True,
        #     logger=True,
        #     sync_dist=True,
        # )
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

    def configure_optimizers(self):
        optimizer = Novograd(
            self.parameters(),
            lr=self.lr,
            amsgrad=True,
            weight_decay=0.001,
        )

        return optimizer
