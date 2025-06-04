
"""
# ===============================================
#  Created by: Kenny Davila Castellanos
#              for CSC 480
#
#  MODIFIED BY: [Min Song]
# ===============================================
"""

import json
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from typing import List, Tuple, Dict


# =========================================================================
#    This function takes as input a filename (a file in JSON format) and
#    loads the file into memory.
#
#    It returns a tuple of the form (classifier_name, class_config), where:
#      - classifier_name refers to the active classifier in the config
#          it is the value for the "active_classifier" field.
#      - class_config is a python dictionary with the contents of the
#          configuration file for the selected stage ("cross_validation" or
#          "training"), and the active classifier.
# =========================================================================
def load_hyperparameters(config_filename: str, stage: str) -> Tuple[str, Dict]:
    # TODO: 1) load the JSON file with the configuration ...
    try:
        with open(config_filename, 'r') as f:
            config_data = json.load(f) # JSON 파일을 메모리로 로드합니다 [1, 4-6].
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_filename}' not found.")
        return "[Invalid Classifier Name]", {}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{config_filename}'. Check file format.")
        return "[Invalid Classifier Name]", {}
    
    # TODO: 2) find the active classifier
    # 설정 파일에서 "active_classifier" 필드의 값을 가져와 활성 분류기 이름을 결정합니다 [1, 2, 7].
    active_classifier_name = config_data.get("active_classifier", "[Invalid Classifier Name]")
    if active_classifier_name == "[Invalid Classifier Name]":
        print("Warning: 'active_classifier' field not found in the configuration file.")
        return active_classifier_name, {}
    
    # TODO: 3) return the hyperparameters for the selected stage and classifier
    hyperparameters_section = config_data.get("hyperparameters", {})
    
    # 주어진 stage (예: "cross_validation")에 해당하는 하이퍼파라미터 딕셔너리를 가져옵니다 [1, 7].
    stage_hyperparameters = hyperparameters_section.get(stage, {})
    
    # 활성 분류기(예: "logistic_classifier")에 해당하는 최종 하이퍼파라미터를 가져옵니다 [2, 7].
    classifier_hyperparameters = stage_hyperparameters.get(active_classifier_name, {})

    if not classifier_hyperparameters:
        print(f"Warning: No hyperparameters found for stage '{stage}' and classifier '{active_classifier_name}'.")

    # 활성 분류기 이름과 해당 하이퍼파라미터 딕셔너리를 반환합니다 [8].
    return active_classifier_name, classifier_hyperparameters

    invalid_config = {}
    return "[Invalid Classifier Name]", invalid_config


# =========================================================================
#    This function takes as input a filename (a file in CVS format) and
#    loads the file into memory. It applies the initial codification
#    that converts categorical attributes into values (No=0.0, Yes=1.0).
#    Other numerical attributes are converted from string value to their
#    floating point values.
#
#    It returns a tuple (X, Y) with two numpy arrays X and Y.
#    Keep in mind that the input file has headers that name the columns
#    Also, keep in mind that one row represents one example
#    The last column always represents the y value for that example
# =========================================================================
def load_raw_dataset(dataset_filename: str) -> Tuple[np.ndarray, np.ndarray]:
    # TODO: there are many ways to achieve this goal, here I will provide recommendations
    #       for basic built-in functions (no libraries required)
    #       however, feel free to implement this using the CSV or Pandas Library
    
    # TODO: 1) load the raw data from file in CSV format
    #         (input: filename, output: list of strings (one per line)  loaded from file)
    X_list = [] # 특징 데이터를 저장할 리스트 [4]
    Y_list = [] # 레이블 데이터를 저장할 리스트 [4]

    with open(dataset_filename, 'r') as f:
        # 헤더 라인을 건너뜁니다. [1, 4]
        header = f.readline()

        for line in f:
            # 각 라인의 앞뒤 공백을 제거하고 쉼표(,)로 분할합니다.
            # 데이터 파일 샘플 [8]을 보면 공백으로 구분되어 있으나,
            # 'CSV format' [2, 9]이라고 명시되어 있으므로 쉼표(,)로 가정합니다.
            # 만약 실제 데이터 파일이 공백으로 구분되어 있다면 `line.strip().split()`으로 변경해야 합니다.
            parts = line.strip().split(',')

            # 빈 줄이거나 모든 부분이 비어있는 경우 건너뜁니다.
            if not parts or all(not part.strip() for part in parts):
                continue

            # 마지막 부분은 레이블(Y)입니다. [1, 4]
            label_str = parts[-1].strip().lower()
            if label_str == 'java':
                Y_list.append(0.0) # 'java'는 0.0으로 할당합니다.
            elif label_str == 'python':
                Y_list.append(1.0) # 'python'은 1.0으로 할당합니다.
            else:
                # 예상치 못한 레이블 값이 있는 경우 처리 (예: 경고 로깅)
                continue

            # 나머지 부분은 특징(X)입니다. [4]
            features_row = []
            for part in parts[:-1]: # 마지막 부분을 제외한 모든 부분에 대해 반복합니다.
                part_stripped = part.strip().lower()
                if part_stripped == 'yes':
                    features_row.append(1.0) # 'yes'는 1.0으로 변환합니다. [3, 5]
                elif part_stripped == 'no':
                    features_row.append(0.0) # 'no'는 0.0으로 변환합니다. [3, 5]
                else:
                    try:
                        # 숫자 특징은 실수(float)로 변환합니다. [3, 6]
                        features_row.append(float(part_stripped))
                    except ValueError:
                        # 'Yes'/'No' 또는 숫자가 아닌 다른 값이 특징으로 있는 경우 (예: 손상된 데이터)
                        # 현재는 해당 값을 무시하거나, 필요에 따라 오류를 발생시킬 수 있습니다.
                        # 주어진 데이터셋 [8, 10-18] [19-36] [37-80]은 'Yes'/'No' 또는 숫자만 포함합니다.
                        pass 
            
            # 특징 행이 비어있지 않은 경우에만 추가합니다.
            if features_row:
                X_list.append(features_row)
            else:
                # 특징이 없는 라인은 건너뜁니다 (예: 레이블만 있는 경우).
                # 이미 label_str에서 continue를 처리했으므로 일반적으로 여기에 도달하지 않습니다.
                continue


    # TODO: 2) convert the lines of text into a dataset using:
    #             a single list for Y
    #             a List of Lists for X
    #          every line in the file is a data row in the dataset
    #          except for the first row that has the header, which you can ignore
    #          For N columns, the first 1 to N-1 are the attribute values (x), and
    #             the last one (N) is the label value or class (y).
    #          Remember to convert everything to float. This also requires codification
    #             of some categorical values
    #         (input: list of strings (one per line of original file)
    #         (outputs: a list of lists for X, a list for Y)

    # TODO: 3) convert your lists for X and Y into numpy arrays
    # (THIS IS AN EXAMPLE, YOU MUST CHANGE THIS
    # e1_x = [0, 0, 0]
    # e1_y = ["python"]
    # e2_x = [0, 0, 1]
    # e2_y = ["java"]
    # dataset_x = np.array([e1_x, e2_x])
    # dataset_y = np.array([e1_y, e2_y])
    dataset_x = np.array(X_list, dtype=np.float32) # X의 dtype은 floating-point type이어야 합니다. [81]
    dataset_y = np.array(Y_list, dtype=np.float32) # Y도 숫자로 변환되었으므로 float32로 통일합니다.

    # TODO: 4) return the numpy arrays as a tuple: (x, y)
    return dataset_x, dataset_y


# =========================================================================
#    This function takes as input a raw dataset (a numpy array) and
#    applies normalization, to adjust the values assuming that they come from
#    a normal distribution.
#
#    The function takes as parameter an optional StandardScaler
#       if provided, it is assumed that it has been fitted before (on training data)
#          and this function should simply use it on the given dataset
#          the function will return the SAME StandardScaler object it received
#       if NOT provided, then a new StandardScaler object must be created
#          the function should fit the new scaler on the given dataset
#          the function will return the NEW StandardScaler object it created
#
#    It returns a tuple (new_X, Scaler)
#
#    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
# =========================================================================
def apply_normalization(raw_dataset: np.ndarray, scaler: StandardScaler | None) -> Tuple[np.ndarray, StandardScaler | None]:
    # # TODO: 1) use or create standard scaler to normalize data
    # # CHANGE OR REMOVE THIS
    # invalid_dataset = np.zeros((10, 20))
    # invalid_scaler = None

    # # TODO: 2) return the normalized data AND the scaler
    # return invalid_dataset, invalid_scaler

    if scaler is None:
        # 스케일러가 제공되지 않았다면, 새로 생성하고 데이터에 fit 및 transform 합니다.
        new_scaler = StandardScaler()
        normalized_dataset = new_scaler.fit_transform(raw_dataset)
        return normalized_dataset, new_scaler
    else:
        # 스케일러가 제공되었다면, 해당 스케일러를 사용하여 데이터를 transform 합니다.
        normalized_dataset = scaler.transform(raw_dataset)
        return normalized_dataset, scaler


# =========================================================================
#    This function takes as input a dataset and splits it into
#    n equal-size DISJOINT partitions for cross validation.
#    if the original dataset cannot be divided into equally sized partitions
#    the extra elements should be distributed on the first so many partitions
#        For example, if the dataset has 43 elements, and n=5
#        then the function needs to split them into partitions of sizes
#            9, 9, 9, 8, 8 (= 43)
#        No elements will be missing or repeated, and the largest partitions
#        can have at MOST ONE more element than the smallest partitions
#
#    Here the dataset is represented by two numpy arrays,
#       the first one for X values (the attributes)
#       the second one for Y values (he labels or classes)
#
#   BEFORE SPLITTING, THIS FUNCTION SHOULD SHUFFLE THE DATASET. THIS REQUIRES
#   THE SAME RE-ORDERING TO BE APPLIED TO BOTH X AND Y. FOR THIS, CREATE AND
#   USE AN ARRAY OF SHUFFLED INDEXES.
#
#   Each partition is represent by a tuple of numpy arrays representing
#        the X and Y for that partition
#   The function returns a list of tuples representing all partitions
#
#   HINT: some functions on Scikit-learn might be useful here, but this
#   functionality is very easy to implement with array slicing
#   as provided by the ndarray class of numpy
# =========================================================================
def split_dataset(dataset_X: np.ndarray, dataset_Y: np.ndarray, n: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    # TODO 1) create a copy of the dataset

    # TODO 2) shuffle the copy of the dataset
    # TODO:  2.1) create random order for all elements (Check: np.random.shuffle)
    # TODO:  2.2) apply this random order to X
    #  (Check: advanced indexing: https://numpy.org/doc/stable/user/basics.indexing.html)
    # TODO:  2.3) apply this random order to Y
    #  (Check: advanced indexing: https://numpy.org/doc/stable/user/basics.indexing.html)

    # TODO: 3) compute partition sizes

    # TODO: 4) compute the partitions using the SHUFFLED COPY
    #          also, don't forget to split both x and y

    # TODO: 5) return the partitions of the SHUFFLED COPY
    # return []
    # TODO 1) create a copy of the dataset (실제로는 셔플링된 인덱싱을 통해 새 배열이 생성됩니다)
    # 총 데이터 예제 수를 가져옵니다 [6].
    total_examples = len(dataset_X)

    # n이 유효한 값인지 확인합니다. (예: cross_validation에서는 n >= 2를 요구합니다 [7, 8])
    if not (isinstance(n, int) and n > 0):
        raise ValueError("n은 1보다 큰 정수여야 합니다.")
    if n > total_examples:
        # n이 총 예제 수보다 큰 경우, 각 예제가 자체 파티션이 되고 나머지는 비어있게 됩니다.
        # 교차 검증의 일반적인 시나리오에서는 n <= total_examples로 가정하는 것이 적절합니다.
        # 필요하다면 여기에 특정 예외 처리를 추가할 수 있습니다.
        pass

    # TODO 2) shuffle the copy of the dataset
    # TODO: 2.1) create random order for all elements (Check: np.random.shuffle)
    # 0부터 (total_examples - 1)까지의 정수 배열을 생성합니다 [3, 4].
    indices = np.arange(total_examples)
    # 이 인덱스 배열을 무작위로 섞습니다 [9].
    np.random.shuffle(indices)

    # TODO: 2.2) apply this random order to X
    # TODO: 2.3) apply this random order to Y
    # 셔플링된 인덱스를 사용하여 dataset_X와 dataset_Y의 행을 재정렬합니다 [3, 9, 10].
    # NumPy의 고급 인덱싱은 이 경우 새로운 배열(복사본)을 반환하므로, 명시적인 .copy() 호출은 필요하지 않습니다.
    shuffled_X = dataset_X[indices]
    shuffled_Y = dataset_Y[indices]

    # TODO: 3) compute partition sizes
    # 기본 파티션 크기를 계산합니다 [3, 6].
    base_size = total_examples // n
    # 나머지 예제 수를 계산합니다 [3, 6].
    remainder = total_examples % n

    partition_sizes = []
    # 각 파티션의 실제 크기를 결정합니다. 나머지 예제들을 첫 번째 파티션부터 하나씩 할당합니다 [3, 6].
    for i in range(n):
        current_size = base_size
        if i < remainder:
            current_size += 1
        partition_sizes.append(current_size)

    # TODO: 4) compute the partitions using the SHUFFLED COPY
    # also, don't forget to split both x and y
    partitions: List[Tuple[np.ndarray, np.ndarray]] = []
    current_idx = 0
    # 계산된 파티션 크기에 따라 셔플링된 데이터를 분할합니다 [4].
    for size in partition_sizes:
        split_X = shuffled_X[current_idx : current_idx + size]
        split_Y = shuffled_Y[current_idx : current_idx + size]
        partitions.append((split_X, split_Y))
        current_idx += size

    # TODO: 5) return the partitions of the SHUFFLED COPY
    return partitions


# ==================================================================================
#   This function takes the name of the classifier and the given hyperparameters
#   and creates a Classifier of the corresponding type, with those hyperparameters
#
#   Then, the function trains the new classifier with the given data
#   and returns it
#
#   Types of classifiers supported:
#              classifier_name == "decision_tree"       -> DecisionTreeClassifier
#              classifier_name == "random_forest"       -> RandomForestClassifier
#              classifier_name == "logistic_classifier" -> LogisticRegression
# ==================================================================================
def train_classifier(classifier_name: str, hyper_params: dict, train_split_X: np.ndarray, train_split_Y: np.ndarray):
        # TODO: 1) Create a new classifier [2]
    classifier = None

    if classifier_name == "decision_tree":
        # Decision Tree Classifier 생성 [1]
        # 하이퍼파라미터: max_depth, criterion [3, 4]
        max_depth = hyper_params.get("max_depth")
        criterion = hyper_params.get("criterion")
        classifier = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
        
    elif classifier_name == "random_forest":
        # Random Forest Classifier 생성 [1]
        # 하이퍼파라미터: n_trees (Scikit-learn에서는 n_estimators), max_depth [4, 5]
        n_estimators = hyper_params.get("n_trees") # 소스에서는 "n_trees"를 사용 [4, 5]
        max_depth = hyper_params.get("max_depth")
        classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        
    elif classifier_name == "logistic_classifier":
        # Logistic Regressor 생성 [2]
        # 하이퍼파라미터: penalty, C [4, 6]
        # 'saga' solver를 사용하여 다양한 페널티("l1", "l2", None)를 지원하도록 명시 [6]
        penalty = hyper_params.get("penalty")
        C_value = hyper_params.get("C")
        
        # JSON에서 null로 넘어오는 경우 Python에서는 None이 됩니다.
        # Scikit-learn의 LogisticRegression에서 penalty=None은 정규화를 적용하지 않습니다.
        # penalty='none' 문자열은 최근 버전에서 Deprecated되었으므로 None을 사용하는 것이 좋습니다.
        if penalty is None or penalty == 'none': # 'none' 문자열 또는 실제 None 값 처리
            classifier = LogisticRegression(penalty=None, solver='saga', C=C_value) # penalty가 None이면 C 값은 무시되지만 전달해도 무방합니다.
        else:
            classifier = LogisticRegression(penalty=penalty, C=C_value, solver='saga')
            
    else:
        print(f"Error: Unsupported classifier name: {classifier_name}")
        return None

    # TODO: 2) Train this classifier with the given data [2]
    if classifier is not None:
        # Scikit-learn 분류기는 fit() 메서드를 사용하여 훈련됩니다 [7, 8].
        classifier.fit(train_split_X, train_split_Y)

    # TODO: 3) Return the trained classifier [2]
    return classifier
    
    # TODO: 1) Create a new classifier
    # TODO: 2) Train this classifier with the given data
    # TODO: 3) Return the trained classifier
    return None


# def train_classifier(classifier_name: str, hyper_params: dict, train_split_X: np.ndarray, train_split_Y: np.ndarray):
#     """
#     지정된 분류기를 생성하고 훈련시키는 함수
    
#     Args:
#         classifier_name: 분류기 종류 ("decision_tree", "random_forest", "logistic_classifier")
#         hyper_params: 하이퍼파라미터 딕셔너리
#         train_split_X: 훈련용 특성 데이터
#         train_split_Y: 훈련용 레이블 데이터
    
#     Returns:
#         훈련된 분류기 객체
#     """
    
#     # TODO: 1) Create a new classifier
#     if classifier_name == "decision_tree":
#         classifier = DecisionTreeClassifier(**hyper_params)
#     elif classifier_name == "random_forest":
#         classifier = RandomForestClassifier(**hyper_params)
#     elif classifier_name == "logistic_classifier":
#         classifier = LogisticRegression(**hyper_params)
#     else:
#         raise ValueError(f"지원하지 않는 분류기입니다: {classifier_name}")
    
#     # TODO: 2) Train this classifier with the given data
#     classifier.fit(train_split_X, train_split_Y)
    
#     # TODO: 3) Return the trained classifier
#     return classifier
