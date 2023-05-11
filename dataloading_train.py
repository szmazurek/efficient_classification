import numpy as np
import os
import torch
from torch.utils.data import Dataset
from loading_helper import generateMyTrainingData
import h5py
from sklearn.model_selection import train_test_split

def load_training_data(args):
    if not os.path.isdir("my_training_data"):
        os.makedirs("my_training_data")
        print("Create my training dataset")
        generateMyTrainingData(args)

    h5f = h5py.File("my_training_data/traindata.h5", 'r')
    img_train = np.array(h5f['img'])
    tar_label_train = np.array(h5f['tar_label'])

    split_train, split_val = train_test_split(torch.arange(img_train.shape[0]), test_size=0.1,
                                              stratify=tar_label_train)
    train_imgs = np.transpose(img_train[split_train], (0, 3, 1, 2))
    tar_train_labels = tar_label_train[split_train]
    val_imgs = np.transpose(img_train[split_val], (0, 3, 1, 2))
    tar_val_labels = tar_label_train[split_val]

    fin_train_dataset = LIDCDataset(train_imgs, tar_train_labels)
    fin_val_dataset = LIDCDataset(val_imgs, tar_val_labels)

    train_loader = torch.utils.data.DataLoader(fin_train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(fin_val_dataset, batch_size=args.batch_size, shuffle=True)

    return train_loader, val_loader


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
