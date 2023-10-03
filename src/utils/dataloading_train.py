import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .loading_helper import generateMyTrainingData


def load_training_data(args):
    if not os.path.isdir("my_training_data"):
        os.makedirs("my_training_data")
        print("Create my training dataset")
        generateMyTrainingData(args)

    h5f = h5py.File("my_training_data/traindata.h5", "r")
    img_train = np.array(h5f["img"])
    tar_label_train = np.array(h5f["tar_label"])

    train_imgs = np.transpose(img_train, (0, 3, 1, 2))
    tar_train_labels = tar_label_train

    fin_train_dataset = LIDCDataset(train_imgs, tar_train_labels)
    train_loader = torch.utils.data.DataLoader(
        fin_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=6,
        prefetch_factor=3,
        persistent_workers=True,
        drop_last=False,
    )

    return train_loader


class LIDCDataset(Dataset):
    def __init__(self, images, class_labels):
        self.images = images
        self.class_labels = class_labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.expand_dims(self.images[idx], axis=0)
        class_label = self.class_labels[idx]

        return image, class_label
