
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
# def cross_validation(raw_x: np.ndarray, raw_y: np.ndarray, n_folds: int, classifier_name: str, hyper_params: Dict) -> Dict:
#     # TODO: 1) split the dataset
#     # split_dataset 함수는 List[Tuple[np.ndarray, np.ndarray]]를 반환합니다 [1].
#     partitions = split_dataset(raw_x, raw_y, n_folds)

#     all_train_metrics_per_fold = []
#     all_validation_metrics_per_fold = []

#     # TODO: 2) for each split ...
#     for i in range(n_folds):
#         # i번째 폴드를 검증 세트로 사용하고, 나머지 폴드를 훈련 세트로 사용합니다.
        
#         # 2.2) prepare the split validation dataset (normalize)
#         # partitions[i]는 (validation_X, validation_Y) 튜플입니다.
#         validation_split_X_raw = partitions[i][0] # 튜플의 첫 번째 요소 (X 데이터)
#         validation_split_Y_raw = partitions[i][1] # 튜플의 두 번째 요소 (Y 데이터)

#         # 2.1) prepare the split training dataset (concatenate and normalize)
#         train_splits_X_for_concat = []
#         train_splits_Y_for_concat = []

#         for j in range(n_folds):
#             if i == j: # 현재 i번째 폴드는 검증 세트이므로 훈련 세트에서는 제외합니다.
#                 continue
            
#             # 훈련 세트가 될 다른 폴드들의 X와 Y 데이터를 각각의 리스트에 추가합니다.
#             # 여기서 partitions[j] (X 데이터)와 partitions[j][3] (Y 데이터)를 명시적으로 사용해야 합니다.
#             train_splits_X_for_concat.append(partitions[j][0]) # X 데이터만 추가
#             train_splits_Y_for_concat.append(partitions[j][1]) # Y 데이터만 추가

#         # 이제 train_splits_X_for_concat은 NumPy 배열들의 리스트이므로, concatenate가 가능합니다.
#         train_split_X_concat = np.concatenate(train_splits_X_for_concat, axis=0)
#         train_split_Y_concat = np.concatenate(train_splits_Y_for_concat, axis=0)

#         # 2.1) prepare the split training dataset (normalize)
#         # 훈련 데이터셋을 정규화합니다. 정규화 파라미터는 해당 훈련 세트에서 학습됩니다 [4, 9].
#         # NOTE: auxiliary_functions.py의 apply_normalization 함수는 현재 TODO 상태입니다 [10, 11].
#         # 이 함수가 StandardScaler를 사용하여 훈련 데이터에 fit하고 transform한 후
#         # 학습된 scaler 객체를 반환하도록 구현되어야 합니다.
#         normalized_train_X, scaler = apply_normalization(train_split_X_concat, None)
#         normalized_train_Y = train_split_Y_concat # Y는 정규화할 필요가 없습니다.

#         # 2.2) prepare the split validation dataset (normalize)
#         # 검증 데이터셋을 정규화합니다. 이때, 훈련 세트에서 학습된 동일한 정규화 파라미터를 사용합니다 [4, 9].
#         normalized_val_X, _ = apply_normalization(validation_split_X_raw, scaler)
#         normalized_val_Y = validation_split_Y_raw # Y는 정규화할 필요가 없습니다.

#         # 2.3) train your classifier on the training split
#         # 정규화된 훈련 데이터로 분류기를 훈련합니다 [5].
#         # NOTE: auxiliary_functions.py의 train_classifier 함수는 현재 TODO 상태입니다 [12].
#         # 이 함수가 지정된 하이퍼파라미터로 분류기 객체를 생성하고 훈련한 후,
#         # 훈련된 분류기 객체를 반환하도록 구현되어야 합니다 [13].
#         trained_classifier = train_classifier(classifier_name, hyper_params, normalized_train_X, normalized_train_Y)

#         # 2.4) evaluate your classifier on the training split (compute and print metrics)
#         # 훈련된 분류기로 훈련 세트에 대한 예측을 수행하고 메트릭을 계산합니다 [5].
#         train_predictions = trained_classifier.predict(normalized_train_X)
#         # Scikit-Learn의 classification_report를 사용하여 메트릭을 딕셔너리 형태로 얻습니다 [1, 2].
#         # zero_division=0을 사용하여 0으로 나눌 때 발생하는 경고를 방지하고 메트릭을 0.0으로 설정합니다.
#         train_report = classification_report(normalized_train_Y, train_predictions, output_dict=True, zero_division=0)
#         train_metrics_per_fold.append(train_report)

#         # 2.5) evaluate your classifier on the validation split (compute and print metrics)
#         # 훈련된 분류기로 검증 세트에 대한 예측을 수행하고 메트릭을 계산합니다 [5].
#         val_predictions = trained_classifier.predict(normalized_val_X)
#         val_report = classification_report(normalized_val_Y, val_predictions, output_dict=True, zero_division=0)
#         val_metrics_per_fold.append(val_report)

#         # 2.6) collect your metrics (위에서 리스트에 추가함으로써 이미 수집되었습니다) [5]
#     # TODO: 3) compute the averaged metrics
#     #          5.1) compute and print training metrics
#     #          5.2) compute and print validation metrics
#     final_metrics = {
#         "train": {
#            "java": {"precision": None, "recall": None, "f1-score": None, "support": None},
#            "python": {"precision": None, "recall": None, "f1-score": None, "support": None},
#            "accuracy": None,
#            "macro avg": {"precision": None, "recall": None, "f1-score": None, "support": None},
#            "weighted avg": {"precision": None, "recall": None, "f1-score": None, "support": None}
#         },
#         "validation": {
#             "java": {"precision": None, "recall": None, "f1-score": None, "support": None},
#             "python": {"precision": None, "recall": None, "f1-score": None, "support": None},
#             "accuracy": None,
#             "macro avg": {"precision": None, "recall": None, "f1-score": None, "support": None},
#             "weighted avg": {"precision": None, "recall": None, "f1-score": None, "support": None}
#         }
#     }

#     # "java", "python", "macro avg", "weighted avg" 카테고리에 대해 평균 메트릭을 계산합니다.
#     metric_categories_to_average = ["java", "python", "macro avg", "weighted avg"]
#     metric_types_to_average = ["precision", "recall", "f1-score"]

#     for data_type in ["train", "validation"]:
#         metrics_list = train_metrics_per_fold if data_type == "train" else val_metrics_per_fold
        
#         # 정확도를 평균합니다.
#         final_metrics[data_type]["accuracy"] = np.mean([m.get("accuracy", 0.0) for m in metrics_list])

#         # 각 클래스 및 매크로/가중 평균에 대해 precision, recall, f1-score를 평균하고 support는 합산합니다.
#         for category in metric_categories_to_average:
#             for metric_type in metric_types_to_average:
#                 # 해당 메트릭 타입에 대한 모든 폴드의 값을 수집합니다.
#                 # `get(category, {})`를 사용하여 키가 없을 때 오류 방지
#                 # `get(metric_type, 0.0)`를 사용하여 메트릭 타입이 없을 때 기본값 0.0 설정
#                 values = [m.get(category, {}).get(metric_type, 0.0) for m in metrics_list if m.get(category) is not None]
#                 if values: # 값이 존재하면 평균 계산
#                     final_metrics[data_type][category][metric_type] = np.mean(values)
#                 else: # 값이 없으면 0.0 유지
#                     final_metrics[data_type][category][metric_type] = 0.0

#             # support는 각 폴드의 값을 합산합니다.
#             # Grid Search 보고서 테이블 [14-17]에는 support 필드가 없지만,
#             # 내부적으로는 필요할 수 있어 합산을 유지하되, 필요 없으면 제거할 수 있습니다.
#             # 여기서는 initial_metrics 템플릿에 support가 0.0으로 명시되어 있으므로,
#             # 해당 값을 각 폴드의 support 값을 합산한 것으로 업데이트합니다.
#             # `classification_report`의 `macro avg` 및 `weighted avg`의 support는 전체 샘플 수입니다.
#             if category == "macro avg" or category == "weighted avg":
#                 final_metrics[data_type][category]["support"] = sum(m.get(category, {}).get("support", 0) for m in metrics_list)
#             else: # 개별 클래스의 support는 해당 클래스의 총 인스턴스 수를 나타냅니다.
#                 final_metrics[data_type][category]["support"] = sum(m.get(category, {}).get("support", 0) for m in metrics_list)

#     # TODO: 4) return your metrics
#     return final_metrics

def cross_validation(raw_x: np.ndarray, raw_y: np.ndarray, n_folds: int, classifier_name: str, hyper_params: Dict) -> Dict:
    # TODO: 1) split the dataset
    # split_dataset 함수는 List[Tuple[np.ndarray, np.ndarray]]를 반환합니다 [1].
    partitions = split_dataset(raw_x, raw_y, n_folds)

    # 변수명 수정: 실제로 사용되는 변수명과 일치시킴
    train_metrics_per_fold = []
    val_metrics_per_fold = []

    # TODO: 2) for each split ...
    for i in range(n_folds):
        # i번째 폴드를 검증 세트로 사용하고, 나머지 폴드를 훈련 세트로 사용합니다.
        
        # 2.2) prepare the split validation dataset (normalize)
        # partitions[i]는 (validation_X, validation_Y) 튜플입니다.
        validation_split_X_raw = partitions[i][0] # 튜플의 첫 번째 요소 (X 데이터)
        validation_split_Y_raw = partitions[i][1] # 튜플의 두 번째 요소 (Y 데이터)

        # 2.1) prepare the split training dataset (concatenate and normalize)
        train_splits_X_for_concat = []
        train_splits_Y_for_concat = []

        for j in range(n_folds):
            if i == j: # 현재 i번째 폴드는 검증 세트이므로 훈련 세트에서는 제외합니다.
                continue
            
            # 훈련 세트가 될 다른 폴드들의 X와 Y 데이터를 각각의 리스트에 추가합니다.
            train_splits_X_for_concat.append(partitions[j][0]) # X 데이터만 추가
            train_splits_Y_for_concat.append(partitions[j][1]) # Y 데이터만 추가

        # 이제 train_splits_X_for_concat은 NumPy 배열들의 리스트이므로, concatenate가 가능합니다.
        train_split_X_concat = np.concatenate(train_splits_X_for_concat, axis=0)
        train_split_Y_concat = np.concatenate(train_splits_Y_for_concat, axis=0)

        # 2.1) prepare the split training dataset (normalize)
        # 훈련 데이터셋을 정규화합니다. 정규화 파라미터는 해당 훈련 세트에서 학습됩니다 [4, 9].
        # NOTE: auxiliary_functions.py의 apply_normalization 함수는 현재 TODO 상태입니다 [10, 11].
        # 이 함수가 StandardScaler를 사용하여 훈련 데이터에 fit하고 transform한 후
        # 학습된 scaler 객체를 반환하도록 구현되어야 합니다.
        normalized_train_X, scaler = apply_normalization(train_split_X_concat, None)
        normalized_train_Y = train_split_Y_concat # Y는 정규화할 필요가 없습니다.

        # 2.2) prepare the split validation dataset (normalize)
        # 검증 데이터셋을 정규화합니다. 이때, 훈련 세트에서 학습된 동일한 정규화 파라미터를 사용합니다 [4, 9].
        normalized_val_X, _ = apply_normalization(validation_split_X_raw, scaler)
        normalized_val_Y = validation_split_Y_raw # Y는 정규화할 필요가 없습니다.

        # 2.3) train your classifier on the training split
        # 정규화된 훈련 데이터로 분류기를 훈련합니다 [5].
        # NOTE: auxiliary_functions.py의 train_classifier 함수는 현재 TODO 상태입니다 [12].
        # 이 함수가 지정된 하이퍼파라미터로 분류기 객체를 생성하고 훈련한 후,
        # 훈련된 분류기 객체를 반환하도록 구현되어야 합니다 [13].
        trained_classifier = train_classifier(classifier_name, hyper_params, normalized_train_X, normalized_train_Y)

        # train_classifier가 None을 반환하는 경우를 대비한 안전장치
        if trained_classifier is None:
            raise ValueError(f"train_classifier 함수가 None을 반환했습니다. {classifier_name} 분류기 구현을 확인해주세요.")

        # 2.4) evaluate your classifier on the training split (compute and print metrics)
        # 훈련된 분류기로 훈련 세트에 대한 예측을 수행하고 메트릭을 계산합니다 [5].
        train_predictions = trained_classifier.predict(normalized_train_X)
        # Scikit-Learn의 classification_report를 사용하여 메트릭을 딕셔너리 형태로 얻습니다 [1, 2].
        # zero_division=0을 사용하여 0으로 나눌 때 발생하는 경고를 방지하고 메트릭을 0.0으로 설정합니다.
        train_report = classification_report(normalized_train_Y, train_predictions, output_dict=True, zero_division=0, target_names=['java', 'python'])
        train_metrics_per_fold.append(train_report)

        # 2.5) evaluate your classifier on the validation split (compute and print metrics)
        # 훈련된 분류기로 검증 세트에 대한 예측을 수행하고 메트릭을 계산합니다 [5].
        val_predictions = trained_classifier.predict(normalized_val_X)
        val_report = classification_report(normalized_val_Y, val_predictions, output_dict=True, zero_division=0, target_names=['java', 'python'])
        val_metrics_per_fold.append(val_report)

        # 2.6) collect your metrics (위에서 리스트에 추가함으로써 이미 수집되었습니다) [5]
        
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

    # "java", "python", "macro avg", "weighted avg" 카테고리에 대해 평균 메트릭을 계산합니다.
    metric_categories_to_average = ["java", "python", "macro avg", "weighted avg"]
    metric_types_to_average = ["precision", "recall", "f1-score"]

    for data_type in ["train", "validation"]:
        metrics_list = train_metrics_per_fold if data_type == "train" else val_metrics_per_fold
        
        # 정확도를 평균합니다.
        final_metrics[data_type]["accuracy"] = np.mean([m.get("accuracy", 0.0) for m in metrics_list])

        # 각 클래스 및 매크로/가중 평균에 대해 precision, recall, f1-score를 평균하고 support는 합산합니다.
        for category in metric_categories_to_average:
            for metric_type in metric_types_to_average:
                # 해당 메트릭 타입에 대한 모든 폴드의 값을 수집합니다.
                # `get(category, {})`를 사용하여 키가 없을 때 오류 방지
                # `get(metric_type, 0.0)`를 사용하여 메트릭 타입이 없을 때 기본값 0.0 설정
                values = [m.get(category, {}).get(metric_type, 0.0) for m in metrics_list if m.get(category) is not None]
                if values: # 값이 존재하면 평균 계산
                    final_metrics[data_type][category][metric_type] = np.mean(values)
                else: # 값이 없으면 0.0 유지
                    final_metrics[data_type][category][metric_type] = 0.0

            # support는 각 폴드의 값을 합산합니다.
            # Grid Search 보고서 테이블 [14-17]에는 support 필드가 없지만,
            # 내부적으로는 필요할 수 있어 합산을 유지하되, 필요 없으면 제거할 수 있습니다.
            # 여기서는 initial_metrics 템플릿에 support가 0.0으로 명시되어 있으므로,
            # 해당 값을 각 폴드의 support 값을 합산한 것으로 업데이트합니다.
            # `classification_report`의 `macro avg` 및 `weighted avg`의 support는 전체 샘플 수입니다.
            if category == "macro avg" or category == "weighted avg":
                final_metrics[data_type][category]["support"] = sum(m.get(category, {}).get("support", 0) for m in metrics_list)
            else: # 개별 클래스의 support는 해당 클래스의 총 인스턴스 수를 나타냅니다.
                final_metrics[data_type][category]["support"] = sum(m.get(category, {}).get("support", 0) for m in metrics_list)

    # TODO: 4) return your metrics
    return final_metrics

def main():
    if len(sys.argv) < 4:
        print("Usage:")
        print(f"\tpython {sys.argv[0]} in_config in_raw_data n_folds")
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
    # 'cross_validation' 단계의 하이퍼파라미터를 로드합니다 [1, 2].
    # load_hyperparameters 함수는 보조 함수로, `auxiliary_functions.py`에 정의되어 있습니다 [12].
    classifier_name, hyper_params = load_hyperparameters(in_config_filename, "cross_validation") 
    
    # 로드된 하이퍼파라미터를 확인하는 간단한 출력 (옵션)
    print(f"Active Classifier: {classifier_name}")
    print(f"Hyperparameters for cross_validation: {hyper_params}")

    # TODO: 2) Load your data
    raw_x, raw_y = load_raw_dataset(in_raw_data_filename)
    # 데이터 로드 후 raw_x와 raw_y의 형태를 확인하여 제대로 로드되었는지 검증할 수 있습니다.
    print(f"Loaded X shape: {raw_x.shape}")
    print(f"Loaded Y shape: {raw_y.shape}")

    # TODO: 3) Run cross-validation  
    final_metrics = cross_validation(raw_x, raw_y, n_folds, classifier_name, hyper_params)
    # 결과 확인을 위해 final_metrics 출력
    print("Cross-validation final metrics:")
    for metric_name, value in final_metrics.items():
        print(f"  {metric_name}: {value}")
    # FINISHED!


if __name__ == "__main__":
    main()
