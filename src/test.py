import os
import shutil
import torch
import numpy as np
from models.lightning_module import LightningModel
from torchmetrics import Accuracy, AUROC, F1Score, Precision, Recall


def test_lightning_model(dataloader, args):
    model = LightningModel.load_from_checkpoint(
        args.model_path, num_classes=args.num_classes, in_channels=1
    )
    if os.path.isdir("testing_data_prediction_classification"):
        shutil.rmtree("testing_data_prediction_classification")
    os.makedirs("testing_data_prediction_classification")
    test_fnc_final_lightning(model, dataloader)


def test_fnc_final_lightning(test_model, data_loader):
    test_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for i, (x, scanIDs, nodIDs) in enumerate(data_loader):
            x = x.to(device, dtype=torch.float)

            pred_mal = test_model(x)

            for samplei in range(len((scanIDs))):
                if not os.path.isdir(
                    "testing_data_prediction_classification/scan_"
                    + str(int(scanIDs[samplei].item()))
                ):
                    os.makedirs(
                        "testing_data_prediction_classification/scan_"
                        + str(int(scanIDs[samplei].item()))
                    )
                np.savetxt(
                    "testing_data_prediction_classification/scan_"
                    + str(int(scanIDs[samplei].item()))
                    + "/nodule_"
                    + str(int(nodIDs[samplei].item()))
                    + ".txt",
                    pred_mal[samplei].cpu(),
                    delimiter=",",
                )


def calculate_accuracy_lightning(args):
    y_list = []
    pred_list = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for scan_file in os.listdir(args.testing_data_solution_path):
        for nod_file in os.listdir(
            args.testing_data_solution_path + "/" + scan_file
        ):
            y = np.loadtxt(
                args.testing_data_solution_path
                + "/"
                + scan_file
                + "/"
                + nod_file,
                delimiter=",",
            )
            pred = np.loadtxt(
                "testing_data_prediction_classification/"
                + scan_file
                + "/"
                + nod_file,
                delimiter=",",
            )
            y_list.append(y)
            pred_list.append(pred)

    y = np.argmax(np.stack(y_list), axis=1)
    pred = np.stack(pred_list)
    y = torch.from_numpy(y).to(device)
    pred = torch.from_numpy(pred).to(device)
    acc = Accuracy("binary").to(device)(pred, y).item()
    auc = AUROC(task="binary").to(device)(pred, y).item()
    f1s = F1Score(task="binary").to(device)(pred, y).item()
    precision = Precision(task="binary").to(device)(pred, y).item()
    recall = Recall(task="binary").to(device)(pred, y).item()

    return acc, auc, f1s, precision, recall
