from test import calculate_accuracy_lightning, test_lightning_model
from train import train_lightning
from utils.dataloading_test import load_testing_data
from utils.dataloading_train import load_training_data

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument(
        "--training_data_path",
        type=str,
        help="Set the path to training dataset",
    )
    parser.add_argument(
        "--testing_data_path", type=str, help="Set the path to testing dataset"
    )
    parser.add_argument(
        "--testing_data_solution_path",
        type=str,
        help="Set the path to solution of testing dataset (for internal testing",
    )
    parser.add_argument(
        "--lr", default=0.001, type=float, help="Learning rate"
    )
    parser.add_argument(
        "--train", action="store_true", help="Use True for training"
    )
    parser.add_argument(
        "--test", action="store_true", help="Use True for test"
    )
    parser.add_argument(
        "--model_path", type=str, help="Set the path of the model to be tested"
    )
    parser.add_argument("--exp_name", type=str, help="Experiment name")
    parser.add_argument("--num_classes", default=1, type=int)
    args = parser.parse_args()

    if not args.train and not args.test:
        raise TypeError(
            "Please specify, whether you want to run the training or testing code by setting the parameter --train=True or --test=True"
        )
    if args.train:
        if args.training_data_path is not None:
            # Preprocess training data. When first time called, data is preprocessed and saved to "my_training_data".
            # When this folder exists, data is loaded from it directly.
            train_loader = load_training_data(args)

            # Train model and saves best performing model at model_path.
            # model_path = train_loop_class(train_loader, val_loader, args)
            model_path, proc_rank = train_lightning(train_loader, args)

            if (
                args.test
                and args.testing_data_path is not None
                and proc_rank == 0
            ):
                # Preprocess testing data. When first time called, data is preprocessed and saved to "my_testing_data".
                # When this folder exists, data is loaded from it directly.
                test_loader = load_testing_data(args)
                print("Number of samples in datasets:")
                print(" testing: " + str(len(test_loader.dataset)))
                args.model_path = model_path
                # Testing data is being predicted and predictions are being saved in folder "testing_data_prediction_classification".
                test_lightning_model(test_loader, args)
                if args.testing_data_solution_path is not None:
                    # Accuracy metric is being calculated between data in folder args.testing_data_solution_path and "testing_data_prediction_classification".
                    (
                        test_acc,
                        test_auc,
                        test_f1s,
                        test_precision,
                        test_recall,
                    ) = calculate_accuracy_lightning(args)
                    print("Testing accuracy: " + str(test_acc))
                    print("Testing AUC: " + str(test_auc))
                    print("Testing F1: " + str(test_f1s))
                    print("Testing Precision: " + str(test_precision))
                    print("Testing Recall: " + str(test_recall))
        else:
            raise TypeError(
                'Please specify the path to the training data by setting the parameter --training_data_path="path_to_training data"'
            )
    elif args.test:
        if args.model_path is None:
            raise TypeError(
                'Please specify the path to model by setting the parameter --model_path="path_to_model"'
            )
        else:
            if args.testing_data_path is not None:
                # Preprocess testing data. When first time called, data is preprocessed and saved to "my_testing_data"; this takes a considerably amount of time.
                # When this folder exists, data is loaded from it directly.
                test_loader = load_testing_data(args)
                # Testing data is being predicted and predictions are being saved in folder "testing_data_prediction_classification".
                # test_model(test_loader, args)
                test_lightning_model(test_loader, args)
                if args.testing_data_solution_path is not None:
                    # Accuracy metric is being calculated between data in folder args.testing_data_solution_path and "testing_data_prediction_classification".
                    (
                        test_acc,
                        test_auc,
                        test_f1s,
                        test_precision,
                        test_recall,
                    ) = calculate_accuracy_lightning(args)
                    print("Testing accuracy: " + str(test_acc))
                    print("Testing AUC: " + str(test_auc))
                    print("Testing F1: " + str(test_f1s))
                    print("Testing Precision: " + str(test_precision))
                    print("Testing Recall: " + str(test_recall))
                else:
                    raise TypeError(
                        'Please specify the path to the testing solution/ground truth data by setting the parameter --testing_data_solution_path="path_to_testingdata_solution"'
                    )
            else:
                raise TypeError(
                    'Please specify the path to the testing data by setting the parameter --testing_data_path="path_to_testingdata"'
                )
