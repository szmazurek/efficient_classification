import torch
from torch import nn
import numpy as np
from torchmetrics import AUROC, F1Score, Precision, Recall
import datetime
import matplotlib.pyplot as plt
from models.model_original import threeDClassModel
from efficientnet_pytorch_3d import EfficientNet3D
from torchsummary import summary
from monai.optimizers import Novograd
from models.lightning_module import LightningModel
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.strategies.ddp import DDPStrategy

torch.set_float32_matmul_precision("medium")


def train_lightning(train_loader, val_loader, args):
    input_channels = next(iter(train_loader))[0].shape[1]
    num_classes = args.num_classes
    model = LightningModel(
        in_channels=input_channels, num_classes=num_classes, lr=args.lr
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=10,
        verbose=True,
        mode="min",
    )
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        mode="min",
        monitor="val_loss",
        dirpath="checkpoints/",
        filename="best_model",
    )
    strategy = pl.strategies.DDPStrategy(
        find_unused_parameters=False,
        static_graph=True,
    )
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed",
        strategy=strategy,
        max_epochs=args.epochs,
        callbacks=[early_stop_callback, model_checkpoint_callback],
    )
    trainer.fit(model, train_loader, val_loader)
    return model_checkpoint_callback.best_model_path


def train_loop_class(train_loader, val_loader, args):
    input_channels = next(iter(train_loader))[0].shape[1]
    num_classes = next(iter(train_loader))[1].shape[-1]
    # model = threeDClassModel(
    #     input_size=input_channels, num_classes=num_classes
    # )
    model = EfficientNet3D.from_name(
        "efficientnet-b0",
        override_params={"num_classes": num_classes},
        in_channels=input_channels,
    )
    model = model.cuda()

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)  # 0.001
    optimizer = Novograd(model.parameters(), lr=args.lr)
    datestr = str(datetime.datetime.now())
    print("this run has datestr " + datestr)
    tr_accs, tr_losses, tr_aucs, tr_f1ss, tr_precisions, tr_recalls = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    val_accs, val_aucs, val_f1ss, val_precisions, val_recalls = (
        [],
        [],
        [],
        [],
        [],
    )
    best_val = 0.0
    for ep in range(args.epochs):
        print("Epoch " + str(ep))
        print("Training")
        (
            model,
            tr_acc,
            tr_loss,
            tr_auc,
            tr_f1s,
            tr_precision,
            tr_recall,
        ) = train_fnc(model, train_loader, optim=optimizer)
        tr_accs.append(tr_acc)
        tr_losses.append(tr_loss.cpu().detach().numpy())
        tr_aucs.append(tr_auc.cpu().detach().numpy())
        tr_f1ss.append(tr_f1s.cpu().detach().numpy())
        tr_precisions.append(tr_precision.cpu().detach().numpy())
        tr_recalls.append(tr_recall.cpu().detach().numpy())
        val_acc, val_auc, val_f1s, val_precision, val_recall = val_fnc(
            model, val_loader
        )
        val_accs.append(val_acc)
        val_aucs.append(val_auc.cpu().detach().numpy())
        val_f1ss.append(val_f1s.cpu().detach().numpy())
        val_precisions.append(val_precision.cpu().detach().numpy())
        val_recalls.append(val_recall.cpu().detach().numpy())
        if val_f1s.cpu().detach().numpy() > best_val:
            print("save new best model")
            torch.save(model, str(datestr) + ".pth")
            best_val = val_f1s.cpu().detach().numpy()
    # plt.subplot(2, 6, 1)
    # plt.plot(np.arange(args.epochs), tr_losses)
    # plt.title("Train Loss")
    # plt.subplot(2, 6, 2)
    # plt.plot(np.arange(args.epochs), tr_accs)
    # plt.title("Train Accuracy")
    # plt.subplot(2, 6, 3)
    # plt.plot(np.arange(args.epochs), tr_aucs)
    # plt.title("Train AUC")
    # plt.subplot(2, 6, 4)
    # plt.plot(np.arange(args.epochs), tr_f1ss)
    # plt.title("Train F1")
    # plt.subplot(2, 6, 5)
    # plt.plot(np.arange(args.epochs), tr_precisions)
    # plt.title("Train Precision")
    # plt.subplot(2, 6, 6)
    # plt.plot(np.arange(args.epochs), tr_recalls)
    # plt.title("Train Recall")
    #
    # plt.subplot(2, 6, 8)
    # plt.plot(np.arange(args.epochs), val_accs)
    # plt.title("Val Accuracy")
    # plt.subplot(2, 6, 9)
    # plt.plot(np.arange(args.epochs), val_aucs)
    # plt.title("Val AUC")
    # plt.subplot(2, 6, 10)
    # plt.plot(np.arange(args.epochs), val_f1ss)
    # plt.title("Val F1")
    # plt.subplot(2, 6, 11)
    # plt.plot(np.arange(args.epochs), val_precisions)
    # plt.title("Val Precision")
    # plt.subplot(2, 6, 12)
    # plt.plot(np.arange(args.epochs), val_recalls)
    # plt.title("Val Recall")
    # plt.show()
    return str(datestr) + ".pth"


def loss_fcn(gt, pred):
    L_pred = nn.CrossEntropyLoss()(torch.squeeze(pred, dim=-1), gt)
    return L_pred


def train_fnc(trainmodel, data_loader, optim):
    trainmodel.train()
    auc, f1s, precision, recall = [], [], [], []
    correct_mal = 0
    tr_loss = 0
    for i, (x, y_mal) in enumerate(data_loader):
        x, y_mal = x.to("cuda", dtype=torch.float), y_mal.to(
            "cuda", dtype=torch.float
        )
        optim.zero_grad()
        pred_mal = trainmodel(x)

        loss = loss_fcn(y_mal, pred_mal)

        loss.backward()
        optim.step()
        tr_loss += loss
        auroc = AUROC(task="multiclass", num_classes=2).to("cuda")
        auc.append(auroc(pred_mal, torch.argmax(y_mal, dim=1)))
        f1score = F1Score(task="multiclass", num_classes=2).to("cuda")
        f1s.append(f1score(pred_mal, torch.argmax(y_mal, dim=1)))
        precisionscore = Precision(
            task="multiclass", average="macro", num_classes=2
        ).to("cuda")
        precision.append(precisionscore(pred_mal, torch.argmax(y_mal, dim=1)))
        recallscore = Recall(
            task="multiclass", average="macro", num_classes=2
        ).to("cuda")
        recall.append(recallscore(pred_mal, torch.argmax(y_mal, dim=1)))

    return (
        trainmodel,
        correct_mal / len(data_loader.dataset),
        tr_loss / len(data_loader.dataset),
        sum(auc) / len(auc),
        sum(f1s) / len(f1s),
        sum(precision) / len(precision),
        sum(recall) / len(recall),
    )


def val_fnc(testmodel, data_loader):
    testmodel.eval()
    auc, f1s, precision, recall = [], [], [], []
    correct_mal = 0
    with torch.no_grad():
        for i, (x, y_mal) in enumerate(data_loader):
            x, y_mal = x.to("cuda", dtype=torch.float), y_mal.to(
                "cuda", dtype=torch.float
            )

            pred_mal = testmodel(x)

            auroc = AUROC(task="multiclass", num_classes=2).to("cuda")
            auc.append(auroc(pred_mal, torch.argmax(y_mal, dim=1)))
            f1score = F1Score(task="multiclass", num_classes=2).to("cuda")
            f1s.append(f1score(pred_mal, torch.argmax(y_mal, dim=1)))
            precisionscore = Precision(
                task="multiclass", average="macro", num_classes=2
            ).to("cuda")
            precision.append(
                precisionscore(pred_mal, torch.argmax(y_mal, dim=1))
            )
            recallscore = Recall(
                task="multiclass", average="macro", num_classes=2
            ).to("cuda")
            recall.append(recallscore(pred_mal, torch.argmax(y_mal, dim=1)))

    return (
        correct_mal / len(data_loader.dataset),
        sum(auc) / len(auc),
        sum(f1s) / len(f1s),
        sum(precision) / len(precision),
        sum(recall) / len(recall),
    )
