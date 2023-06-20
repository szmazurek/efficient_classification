# Starting point for E2MIP challenge

You can use the code from this repository as starting point for your work on the [E2MIP challenge](https://e2mip.github.io/) on the LIDC-IDRI dataset.
It provides code for preprocessing and classification of 3D data.

To install the requirements `$ pip install -r requirements.txt`, the code is tested with python version 3.10.9.

### Run Classification
This repository contains a simple 3D Convolutional model and a training and testing algorithm, that can be used as starting point for the challenge.
#### Data path:
Set the following parameters to the paths of the folders, created by this [repository](https://github.com/XRad-Ulm/E2MIP_LIDCI-IDRI_data).
It provides code to create folders with the datasets in the same way as the data folders that will be used to train and test your submitted code.
* **--training_data_path**: str, e.g. "=training_data"
* **--testing_data_path**: str, e.g. "=testing_data_classification"
* **--testing_data_solution_path**: str, e.g. "=testing_data_solution_classification"

####    Train:
To train the model, additionally specify the following parameters to run the script 'main.py'
* **--train=True**: bool, default=False
* --epochs: int, default=100
* --lr: float, default=0.001 (Learning rate)
####    Test:
To test the model specify the following parameters to run the script 'main.py'
* **--test=True**: bool, default=False
* **--model_path=[path_to_model.pth]**: str, (If --train=True, this argument is being ignored and the newly trained model is being tested)



For further questions about this code, please contact luisa.gallee@uni-ulm.de
