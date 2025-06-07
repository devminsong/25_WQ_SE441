
"""
# ===============================================
#  Created by: Kenny Davila Castellanos
#              for CSC 480
#
#  MODIFIED BY: [Min Song]
# ===============================================
"""

import sys
import numpy as np
import pickle

from auxiliary_functions import *

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# =======================================================================
#    This is an optional function that you can use to print your metrics
#    in a format that makes it easier to tabulate
#
#    This one is up to you. It won't be evaluated!
# =======================================================================
def custom_metric_print(metrics_dict):
    pass


# =======================================================================================================
#   This function runs cross-validation using the provided
#     - raw dataset: a dataset which has not been normalized,
#          represented by raw X and raw Y
#     - n_folds:   number of data splits to use during the process
#     - classifier_name: name of the classifier algorithm to use
#     - hyperparameter: configuration for the classifier to use
#
#   Keep in mind that dataset normalization should be applied independently
#   for each dataset split. That is, create the merged training set and
#   then normalize this data, and then use the same normalization parameters
#   to normalize the corresponding test split.
#
#   For the evaluation part, you should use classification_report from scikit-learn:
#      https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
#
#   The function runs the cross-validation step, computes the metrics per split, aggregates them and
#   returns a dictionary with the aggregated metrics. This dictionary should follow this format:
#     {
#         "train": sub_metrics_dict,
#         "validation": sub_metrics_dict
#     }
#
#   Where each sub_metrics_dict follows this format:
#     {
#           "class 1 name": group_metrics,
#           "class 2 name": group_metrics,
#              ...
#           "class N name": group_metrics,
#           "accuracy": [AVG accuracy value],
#           "macro avg": group_metrics,
#           "weighted avg": group_metrics
#     }
#
#   Where each group_metrics follows this format:
#     {
#           "precision": [AVG value for this metric],
#           "recall":    [AVG value for this metric],
#           "f1-score":  [AVG value for this metric],
#           "support":   [SUM value for this metric]
#     }
#
#   Note that this is a modified of the dictionary returned by classification_report
# =======================================================================================================
def cross_validation(raw_x: np.ndarray, raw_y: np.ndarray, n_folds: int, classifier_name: str, hyper_params: Dict) -> Dict:
    # TODO: 1) split the dataset
    partitions = split_dataset(raw_x, raw_y, n_folds)

    train_reports = []
    validation_reports = []

    # TODO: 2) for each split ...
    for i in range(n_folds):       
        # 2.1) prepare the split training dataset (concatenate and normalize)
        raw_train_data_Xs = []
        raw_train_data_Ys = []

        for j in range(n_folds):
            if i == j:
                continue
            
            raw_train_data_Xs.append(partitions[j][0])
            raw_train_data_Ys.append(partitions[j][1])

        concatenated_train_data_Xs = np.concatenate(raw_train_data_Xs, axis=0)
        concatenated_train_data_Ys = np.concatenate(raw_train_data_Ys, axis=0)

        normalized_train_data_X, scaler = apply_normalization(concatenated_train_data_Xs, None)
        normalized_train_data_Y = concatenated_train_data_Ys

        # 2.2) prepare the split validation dataset (normalize)
        raw_validation_data_X = partitions[i][0]
        raw_validation_data_Y = partitions[i][1]
        
        normalized_validation_data_X, _ = apply_normalization(raw_validation_data_X, scaler)
        normalized_validation_data_Y = raw_validation_data_Y

        # 2.3) train your classifier on the training split
        classifier = train_classifier(classifier_name, hyper_params, normalized_train_data_X, normalized_train_data_Y)

        if classifier is None:
            raise ValueError(f"classifier is None: {classifier_name}")

        # 2.4) evaluate your classifier on the training split (compute and print metrics)
        train_predictions = classifier.predict(normalized_train_data_X)
        train_report = classification_report(normalized_train_data_Y, train_predictions, output_dict=True, zero_division=0, target_names=['java', 'python'])

        # 2.5) evaluate your classifier on the validation split (compute and print metrics)
        validation_predictions = classifier.predict(normalized_validation_data_X)
        validation_report = classification_report(normalized_validation_data_Y, validation_predictions, output_dict=True, zero_division=0, target_names=['java', 'python'])

        # 2.6) collect your metrics
        train_reports.append(train_report)
        validation_reports.append(validation_report)
        
    # TODO: 3) compute the averaged metrics
    #          5.1) compute and print training metrics
    #          5.2) compute and print validation metrics
    final_metrics = {
        "train": {
           "java": {"precision": None, "recall": None, "f1-score": None, "support": None},
           "python": {"precision": None, "recall": None, "f1-score": None, "support": None},
           "accuracy": None,
           "macro avg": {"precision": None, "recall": None, "f1-score": None, "support": None},
           "weighted avg": {"precision": None, "recall": None, "f1-score": None, "support": None}
        },
        "validation": {
            "java": {"precision": None, "recall": None, "f1-score": None, "support": None},
            "python": {"precision": None, "recall": None, "f1-score": None, "support": None},
            "accuracy": None,
            "macro avg": {"precision": None, "recall": None, "f1-score": None, "support": None},
            "weighted avg": {"precision": None, "recall": None, "f1-score": None, "support": None}
        }
    }

    for data_type in ["train", "validation"]:
        reports = train_reports if data_type == "train" else validation_reports
        
        final_metrics[data_type]["accuracy"] = np.mean([report.get("accuracy", 0.0) for report in reports])

        # 각 클래스 및 매크로/가중 평균에 대해 precision, recall, f1-score를 평균하고 support는 합산합니다.
        for category in ["java", "python", "macro avg", "weighted avg"]:
            for metric_type in ["precision", "recall", "f1-score"]:
                values = [report.get(category, {}).get(metric_type, 0.0) for report in reports if report.get(category) is not None]
                if values:
                    final_metrics[data_type][category][metric_type] = np.mean(values)
                else:
                    final_metrics[data_type][category][metric_type] = 0.0

            if category == "macro avg" or category == "weighted avg":
                final_metrics[data_type][category]["support"] = sum(report.get(category, {}).get("support", 0) for report in reports)
            else:
                final_metrics[data_type][category]["support"] = sum(report.get(category, {}).get("support", 0) for report in reports)

    # TODO: 4) return your metrics
    return final_metrics


def main():
    if len(sys.argv) < 4:
        print("Usage:")
        print(f"\tpython {sys.argv[0]} in_config in_raw_data n_folds")
        print(f"ex: python .\part_01_cross_validation.py .\hyperparameters.json .\training_data_small.csv 5")
        return

    in_config_filename = sys.argv[1]
    in_raw_data_filename = sys.argv[2]
    try:
        n_folds = int(sys.argv[3])
        if n_folds < 2:
            print("Invalid value for n_folds. Must be an integer >= 2")
            return
    except:
        print("invalid value for n_folds. Must be an integer >= 2")
        return

    # TODO: 1) Load your (cross-validation) hyper-parameters
    classifier_name, hyper_params = load_hyperparameters(in_config_filename, "cross_validation") 
    
    print(f"Current Classifier: {classifier_name}")
    print(f"Hyperparameters: {hyper_params}")

    # TODO: 2) Load your data
    raw_x, raw_y = load_raw_dataset(in_raw_data_filename)
    print(f"raw_x: {raw_x.shape}")
    print(f"raw_y: {raw_y.shape}")

    # TODO: 3) Run cross-validation  
    print("\n=== Running Cross-Validation ===")
    final_metrics = cross_validation(raw_x, raw_y, n_folds, classifier_name, hyper_params)

    print("Cross-validation final metrics:")
    for metric_name, value in final_metrics.items():
        print(f"{metric_name}: {value}")

    # FINISHED!
    print("\n=== FINISHED ===")


if __name__ == "__main__":
    main()