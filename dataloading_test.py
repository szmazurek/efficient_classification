import numpy as np
import os
import torch
from torch.utils.data import Dataset
from loading_helper import generateMyTestingData
import h5py

def load_testing_data(args):
    if not os.path.isdir("my_testing_data"):
        os.makedirs("my_testing_data")
        print("Create my testing dataset")
        generateMyTestingData(args)

    h5f = h5py.File("my_testing_data/testdata.h5", 'r')
    img_test = np.array(h5f['img'])
    scanID_test = np.array(h5f['scan'])
    nodID_test = np.array(h5f['nod'])

    test_imgs = np.transpose(img_test, (0, 3, 1, 2))
    fin_test_dataset = LIDCDataset(test_imgs, scanID_test, nodID_test)

    test_loader = torch.utils.data.DataLoader(fin_test_dataset, batch_size=args.batch_size, shuffle=True)

    return test_loader


class LIDCDataset(Dataset):
    def __init__(self, images, scanIDs, nodIDs):
        self.images = images
        self.scanIDs = scanIDs
        self.nodIDs = nodIDs
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image = np.expand_dims(self.images[idx], axis=0)
        scanID = self.scanIDs[idx]
        nodID = self.nodIDs[idx]
        return image,scanID,nodID
