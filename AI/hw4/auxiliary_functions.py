
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
            config_data = json.load(f)
    except FileNotFoundError:
        print(f"FileNotFoundError: {config_filename}")
        return "[Invalid Classifier Name]", {}
    except json.JSONDecodeError:
        print(f"json.JSONDecodeError: {config_filename}")
        return "[Invalid Classifier Name]", {}
    
    # TODO: 2) find the active classifier
    active_classifier_name = config_data.get("active_classifier", "[Invalid Classifier Name]")
    if active_classifier_name == "[Invalid Classifier Name]":
        print(f"{active_classifier_name} == [Invalid Classifier Name]")
        return active_classifier_name, {}
    
    # TODO: 3) return the hyperparameters for the selected stage and classifier
    hyperparameters_section = config_data.get("hyperparameters", None)
    if hyperparameters_section is None:
        print("hyperparameters_section is None.")
        return active_classifier_name, {}
    
    stage_hyperparameters = hyperparameters_section.get(stage, None)
    if stage_hyperparameters is None:
        print("stage_hyperparameters is None.")
        return active_classifier_name, {}
    
    classifier_hyperparameters = stage_hyperparameters.get(active_classifier_name, {})
    if not classifier_hyperparameters:
        print("not classifier_hyperparameters.")

    return active_classifier_name, classifier_hyperparameters


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

    X_list = []
    Y_list = []

    with open(dataset_filename, 'r') as f:
        header = f.readline()

        for line in f:
            parts = line.strip().split(',')

            is_empty_line = not parts
            all_fields_blank = all(part.strip() == '' for part in parts)

            if is_empty_line or all_fields_blank:
                continue
                        
            label_str = parts[-1].strip().lower()
            if label_str == 'java':
                Y_list.append(0.0)
            elif label_str == 'python':
                Y_list.append(1.0) 
            else:
                continue
            
            features_row = []
            for part in parts[:-1]:
                part_stripped = part.strip().lower()
                if part_stripped == 'yes':
                    features_row.append(1.0)
                elif part_stripped == 'no':
                    features_row.append(0.0)
                else:
                    try:                        
                        features_row.append(float(part_stripped))
                    except ValueError:
                        pass 
            
            if features_row:
                X_list.append(features_row)
            else:
                continue
   
    # TODO: 3) convert your lists for X and Y into numpy arrays
    dataset_x = np.array(X_list, dtype=np.float32) 
    dataset_y = np.array(Y_list, dtype=np.float32)

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
        new_scaler = StandardScaler()
        normalized_dataset = new_scaler.fit_transform(raw_dataset)
        return normalized_dataset, new_scaler
    else:
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

    # TODO 1) create a copy of the dataset
    # NumPy의 고급 인덱싱은 이 경우 새로운 배열(복사본)을 반환하므로, 명시적인 .copy() 호출은 필요하지 않습니다.
    total_examples = len(dataset_X)

    if not (isinstance(n, int) and n > 1):
        raise ValueError("n은 1보다 큰 정수여야 합니다.")
    if n > total_examples:
        pass

    # TODO 2) shuffle the copy of the dataset
    # TODO: 2.1) create random order for all elements (Check: np.random.shuffle)   
    indices = np.arange(total_examples)
    np.random.shuffle(indices)

    # TODO: 2.2) apply this random order to X
    # TODO: 2.3) apply this random order to Y
    shuffled_X = dataset_X[indices]
    shuffled_Y = dataset_Y[indices]

    # TODO: 3) compute partition sizes
    base_size = total_examples // n
    remainder = total_examples % n

    partition_sizes = []

    for i in range(n):
        current_size = base_size
        if i < remainder:
            current_size += 1
        partition_sizes.append(current_size)

    # TODO: 4) compute the partitions using the SHUFFLED COPY
    # also, don't forget to split both x and y
    partitions: List[Tuple[np.ndarray, np.ndarray]] = []
    current_idx = 0

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
    # TODO: 1) Create a new classifier
    classifier = None

    if classifier_name == "decision_tree":
        max_depth = hyper_params.get("max_depth")
        criterion = hyper_params.get("criterion")
        classifier = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
        
    elif classifier_name == "random_forest":
        n_estimators = hyper_params.get("n_trees")
        max_depth = hyper_params.get("max_depth")
        classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        
    elif classifier_name == "logistic_classifier":
        penalty = hyper_params.get("penalty")
        C_value = hyper_params.get("C")
        
        if penalty is None:
            classifier = LogisticRegression(penalty=None, C=C_value, solver='saga')
        else:
            classifier = LogisticRegression(penalty=penalty, C=C_value, solver='saga')   

    else:
        print(f"{classifier_name} is not supported.")
        return None

    # TODO: 2) Train this classifier with the given data
    if classifier is not None:
        classifier.fit(train_split_X, train_split_Y)

    # TODO: 3) Return the trained classifier
    return classifier