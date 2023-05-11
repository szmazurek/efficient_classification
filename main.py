import torch
import torchvision.transforms as transforms
# from load3dData import load_data
from dataloading_train import load_training_data
from classification_train import train_loop_class
from dataloading_test import load_testing_data
from classification_test import test_model, calculateAccuracy

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--training_data_path', type=str, help="Set the path to training dataset")
    parser.add_argument('--testing_data_path', type=str, help="Set the path to testing dataset")
    parser.add_argument('--testing_data_solution_path', type=str,
                        help="Set the path to solution of testing dataset (for internal testing")
    parser.add_argument('--lr', default=0.001, type=float, help="Learning rate")
    parser.add_argument('--train', type=bool, default=False, help="Use True for training")
    parser.add_argument('--test', type=bool, default=False, help="Use True for testing")
    parser.add_argument('--model_path', type=str, help="Set the path of the model to be tested")
    args = parser.parse_args()
    print(args)

    if not args.train and not args.test:
        raise TypeError(
            "Please specify, whether you want to run the training or testing code by setting the parameter --train=True or --test=True")
    if args.train:
        if not args.training_data_path == None:
            # Preprocess training data. When first time called, data is preprocessed and saved to "my_training_data".
            # When this folder exists, data is loaded from it directly.
            train_loader, val_loader = load_training_data(args)
            print("Number of samples in datasets:")
            print(" training: " + str(len(train_loader.dataset)))
            print(" validation: " + str(len(val_loader.dataset)))
            print("Shape of data:")
            print(" image: " + str(next(iter(train_loader))[0].shape))
            print(" target malignancy label: " + str(next(iter(train_loader))[1].shape))
            # Train model and saves best performing model at model_path.
            model_path = train_loop_class(train_loader, val_loader, args)
            if args.test and not args.testing_data_path == None:
                # Preprocess testing data. When first time called, data is preprocessed and saved to "my_testing_data".
                # When this folder exists, data is loaded from it directly.
                test_loader = load_testing_data(args)
                print("Number of samples in datasets:")
                print(" testing: " + str(len(test_loader.dataset)))
                args.model_path = model_path
                # Testing data is being predicted and predictions are being saved in folder "testing_data_prediction_classification".
                test_model(test_loader, args)
                if not args.testing_data_solution_path == None:
                    # Accuracy metric is being calculated between data in folder args.testing_data_solution_path and "testing_data_prediction_classification".
                    test_acc, test_auc, test_f1s, test_precision, test_recall = calculateAccuracy(args)
                    print("Testing accuracy: " + str(test_acc))
                    print("Testing AUC: " + str(test_auc))
                    print("Testing F1: " + str(test_f1s))
                    print("Testing Precision: " + str(test_precision))
                    print("Testing Recall: " + str(test_recall))
        else:
            raise TypeError(
                "Please specify the path to the training data by setting the parameter --training_data_path=\"path_to_trainingdata\"")
    elif args.test:
        if args.model_path == None:
            raise TypeError("Please specify the path to model by setting the parameter --model_path=\"path_to_model\"")
        else:
            if not args.testing_data_path == None:
                # Preprocess testing data. When first time called, data is preprocessed and saved to "my_testing_data"; this takes a considerably amount of time.
                # When this folder exists, data is loaded from it directly.
                test_loader = load_testing_data(args)
                # Testing data is being predicted and predictions are being saved in folder "testing_data_prediction_classification".
                test_model(test_loader, args)
                if not args.testing_data_solution_path == None:
                    # Accuracy metric is being calculated between data in folder args.testing_data_solution_path and "testing_data_prediction_classification".
                    test_acc, test_auc, test_f1s, test_precision, test_recall = calculateAccuracy(args)
                    print("Testing accuracy: " + str(test_acc))
                    print("Testing AUC: " + str(test_auc))
                    print("Testing F1: " + str(test_f1s))
                    print("Testing Precision: " + str(test_precision))
                    print("Testing Recall: " + str(test_recall))
                else:
                    raise TypeError(
                        "Please specify the path to the testing solution/ground truth data by setting the parameter --testing_data_solution_path=\"path_to_testingdata_solution\"")
            else:
                raise TypeError(
                    "Please specify the path to the testing data by setting the parameter --testing_data_path=\"path_to_testingdata\"")
