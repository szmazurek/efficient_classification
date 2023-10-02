import numpy as np
import os
import h5py
import nibabel as nib
from torchio import Resample


def generateMyTrainingData(args):
    scan_counter = 0
    img_list, tar_list = [], []
    for scan_folder in os.listdir(args.training_data_path):
        scan_counter += 1
        print(
            'Creating "my_training_data" with custom preprocessed scan patches,  at scan: '
            + str(scan_counter)
            + " of "
            + str(len(os.listdir(args.training_data_path)))
        )
        scan_vol = nib.load(
            args.training_data_path + "/" + scan_folder + "/image_total.nii"
        ).get_fdata()

        for nodule_folders in os.listdir(
            args.training_data_path + "/" + scan_folder
        ):
            if os.path.isdir(
                args.training_data_path
                + "/"
                + scan_folder
                + "/"
                + nodule_folders
            ):
                nod_mal = []
                for nodule_annotation_folders in os.listdir(
                    args.training_data_path
                    + "/"
                    + scan_folder
                    + "/"
                    + nodule_folders
                ):
                    nod_anni_mal = np.loadtxt(
                        args.training_data_path
                        + "/"
                        + scan_folder
                        + "/"
                        + nodule_folders
                        + "/"
                        + nodule_annotation_folders
                        + "/mal.txt",
                        delimiter=",",
                    )
                    nod_mal.append(nod_anni_mal)
                for nodule_annotation_folders in os.listdir(
                    args.training_data_path
                    + "/"
                    + scan_folder
                    + "/"
                    + nodule_folders
                ):
                    nod_anni_centroid = np.loadtxt(
                        args.training_data_path
                        + "/"
                        + scan_folder
                        + "/"
                        + nodule_folders
                        + "/"
                        + nodule_annotation_folders
                        + "/centroid.txt",
                        delimiter=",",
                    )
                    cropout_cube_size = 64
                    cropout_cube_size_half = cropout_cube_size / 2
                    cropout_bor = np.array(
                        [
                            [0, scan_vol.shape[0]],
                            [0, scan_vol.shape[1]],
                            [0, scan_vol.shape[2]],
                        ]
                    )
                    for d in range(3):
                        if (
                            int(nod_anni_centroid[d] - cropout_cube_size_half)
                            < 0
                            or int(
                                nod_anni_centroid[d] + cropout_cube_size_half
                            )
                            > scan_vol.shape[d]
                        ):
                            if (
                                int(
                                    nod_anni_centroid[d]
                                    - cropout_cube_size_half
                                )
                                < 0
                            ):
                                cropout_bor[d, 1] = cropout_cube_size
                            else:
                                cropout_bor[d, 0] = (
                                    scan_vol.shape[d] - cropout_cube_size
                                )
                        else:
                            cropout_bor[d, 0] = int(
                                nod_anni_centroid[d] - cropout_cube_size_half
                            )
                            cropout_bor[d, 1] = int(
                                nod_anni_centroid[d] + cropout_cube_size_half
                            )
                    nodule_cropout_cube = scan_vol[
                        cropout_bor[0, 0] : cropout_bor[0, 1],
                        cropout_bor[1, 0] : cropout_bor[1, 1],
                        cropout_bor[2, 0] : cropout_bor[2, 1],
                    ]
                    if np.mean(nod_mal) < 3:
                        tar_labels = 0
                    else:
                        tar_labels = 1  # changed from one-hot to proper binary
                    if (
                        cropout_bor[2, 1] - cropout_bor[2, 0]
                    ) == cropout_cube_size:
                        img_list.append(nodule_cropout_cube)
                        tar_list.append(tar_labels)

        if ((scan_counter > 0) and (scan_counter % 20 == 0)) or (
            scan_counter == len(os.listdir(args.training_data_path))
        ):
            img_train, tar_label_train = np.asarray(img_list), np.asarray(
                tar_list
            )
            if not os.path.isfile("my_training_data/traindata.h5"):
                with h5py.File("my_training_data/traindata.h5", "w") as h5f:
                    h5f.create_dataset(
                        "img",
                        shape=(
                            0,
                            img_train.shape[1],
                            img_train.shape[2],
                            img_train.shape[3],
                        ),
                        chunks=True,
                        maxshape=(
                            None,
                            img_train.shape[1],
                            img_train.shape[2],
                            img_train.shape[3],
                        ),
                    )
                    h5f.create_dataset(
                        "tar_label",
                        shape=(0,),
                        chunks=True,
                        maxshape=(None,),
                    )
                    print(tar_label_train.shape)
            with h5py.File("my_training_data/traindata.h5", "a") as h5f:
                h5f["img"].resize(
                    (h5f["img"].shape[0] + img_train.shape[0]), axis=0
                )
                h5f["img"][-img_train.shape[0] :] = img_train
                h5f["tar_label"].resize(
                    (h5f["tar_label"].shape[0] + tar_label_train.shape[0]),
                    axis=0,
                )
                h5f["tar_label"][-tar_label_train.shape[0] :] = tar_label_train

            img_list, tar_list = [], []


def generateMyTestingData(args):
    scan_counter = 0
    img_list = []
    scanID_list = []
    nodID_list = []
    for scan_folder in os.listdir(args.testing_data_path):
        scan_counter += 1
        print(
            'Creating "my_testing_data" with custom preprocessed scan patches,  at scan: '
            + str(scan_counter)
            + " of "
            + str(len(os.listdir(args.testing_data_path)))
        )

        for nodule_file in os.listdir(
            args.testing_data_path + "/" + scan_folder
        ):
            if os.path.isfile(
                os.path.join(
                    args.testing_data_path + "/" + scan_folder, nodule_file
                )
            ):
                nodule_cropout_cube = nib.load(
                    args.testing_data_path
                    + "/"
                    + scan_folder
                    + "/"
                    + nodule_file
                ).get_fdata()
                img_list.append(nodule_cropout_cube)
                scanID_list.append(int(scan_folder.split("_")[1]))
                nodID_list.append(int(nodule_file.split(".")[0].split("_")[1]))
        if ((scan_counter > 0) and (scan_counter % 20 == 0)) or (
            scan_counter == len(os.listdir(args.testing_data_path))
        ):
            img_test, scanID_test, nodID_test = (
                np.asarray(img_list),
                np.asarray(scanID_list),
                np.asarray(nodID_list),
            )
            print("RESAMPLING IMAGES")
            img_test = Resample(0.75)(img_test)
            if not os.path.isfile("my_testing_data/testdata.h5"):
                with h5py.File("my_testing_data/testdata.h5", "w") as h5f:
                    h5f.create_dataset(
                        "img",
                        shape=(
                            0,
                            img_test.shape[1],
                            img_test.shape[2],
                            img_test.shape[3],
                        ),
                        chunks=True,
                        maxshape=(
                            None,
                            img_test.shape[1],
                            img_test.shape[2],
                            img_test.shape[3],
                        ),
                    )
                    h5f.create_dataset(
                        "scan", shape=(0,), chunks=True, maxshape=(None,)
                    )
                    h5f.create_dataset(
                        "nod", shape=(0,), chunks=True, maxshape=(None,)
                    )
            with h5py.File("my_testing_data/testdata.h5", "a") as h5f:
                h5f["img"].resize(
                    (h5f["img"].shape[0] + img_test.shape[0]), axis=0
                )
                h5f["img"][-img_test.shape[0] :] = img_test
                h5f["scan"].resize(
                    (h5f["scan"].shape[0] + scanID_test.shape[0]), axis=0
                )
                h5f["scan"][-scanID_test.shape[0] :] = scanID_test
                h5f["nod"].resize(
                    (h5f["nod"].shape[0] + nodID_test.shape[0]), axis=0
                )
                h5f["nod"][-nodID_test.shape[0] :] = nodID_test

            img_list, scanID_list, nodID_list = [], [], []
