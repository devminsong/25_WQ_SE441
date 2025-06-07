
"""
# ===============================================
#  Created by: Kenny Davila Castellanos
#              for CSC 480
#
#  MODIFIED BY: [Min Song]
# ===============================================
"""

import sys
import pickle

from auxiliary_functions import *

from sklearn.metrics import classification_report

def main():
    if len(sys.argv) < 4:
        print("Usage:")
        print(f"\tpython {sys.argv[0]} in_raw_data in_normalizer in_classifier")
        return

    in_raw_data_filename = sys.argv[1]
    in_normalizer_filename = sys.argv[2]
    in_classifier_filename = sys.argv[3]

    # TODO: 1) Load your data
    # 'auxiliary_functions.py'의 load_raw_dataset 함수를 사용하여 지정된 원시 데이터셋 파일을 로드합니다. [5-10]
    # 이 함수는 원시 데이터 X와 레이블 Y를 NumPy 배열 형태로 반환합니다. [9, 10]
    print(f"Loading raw data from {in_raw_data_filename}...")
    raw_x, raw_y = load_raw_dataset(in_raw_data_filename)
    print(f"Raw X shape: {raw_x.shape}")
    print(f"Raw Y shape: {raw_y.shape}")

    # TODO: 2) Load the normalizer
    # 'part_03_training.py'에서 저장한 훈련된 StandardScaler 객체를 파일에서 로드합니다. [11, 12]
    # 이 파일은 pickle 라이브러리를 사용하여 바이너리 모드('rb')로 저장되었습니다. [11, 12]
    print(f"\nLoading trained normalizer from {in_normalizer_filename}...")
    try:
        with open(in_normalizer_filename, 'rb') as f:
            scaler = pickle.load(f)
        print("Normalizer loaded successfully.")
    except Exception as e:
        print(f"Error loading normalizer: {e}")
        return
    
    # TODO: 3) Load the classifier
    # 'part_03_training.py'에서 저장한 훈련된 분류기 모델 객체를 파일에서 로드합니다. [11]
    # 이 파일도 pickle 라이브러리를 사용하여 바이너리 모드('rb')로 저장되었습니다. [11]
    print(f"Loading trained classifier from {in_classifier_filename}...")
    try:
        with open(in_classifier_filename, 'rb') as f:
            classifier = pickle.load(f)
        print("Classifier loaded successfully.")
    except Exception as e:
        print(f"Error loading classifier: {e}")
        return
    
    # TODO: 4) normalize the data ...
    # 'auxiliary_functions.py'의 apply_normalization 함수를 사용하여 로드된 원시 데이터(raw_x)를 정규화합니다. [13, 14]
    # 이 단계에서는 새로운 StandardScaler를 피팅하는 것이 아니라, 이전에 훈련된 'scaler' 객체를 사용하여 데이터를 변환해야 합니다. [14]
    # 레이블(Y)은 정규화할 필요가 없습니다. [15]
    print(f"\nNormalizing the data using the loaded normalizer...")
    normalized_x, _ = apply_normalization(raw_x, scaler) # 반환되는 스케일러는 이미 로드된 스케일러와 동일하므로 무시합니다.
    normalized_y = raw_y # Y (레이블)는 정규화할 필요가 없습니다.
    print(f"Normalized X shape: {normalized_x.shape}")

    # TODO: 5) evaluate your classifier on the testing dataset (compute and print metrics)
    # 훈련된 분류기(classifier)를 사용하여 정규화된 데이터(normalized_x)에 대한 예측을 수행합니다. [16]
    print(f"\nEvaluating classifier on the testing dataset...")
    test_predictions = classifier.predict(normalized_x)

    # 실제 레이블(normalized_y)과 예측된 레이블(test_predictions)을 비교하여 분류 보고서를 생성합니다.
    # 'output_dict=True'는 결과를 딕셔너리 형태로 반환하도록 하여 프로그램 내에서 사용하기 용이하게 합니다. [17]
    # 'zero_division=0'은 나눗셈 오류를 방지하고 0으로 처리하도록 합니다. [17]
    # 'target_names'는 레이블 0.0과 1.0이 각각 'java'와 'python'에 매핑되도록 합니다. [17]
    test_report = classification_report(normalized_y, test_predictions,
                                        output_dict=True, zero_division=0,
                                        target_names=['java', 'python'])

    print_metrics(test_report)
    # # 테스트 데이터셋에 대한 평가 지표를 출력합니다.
    # print(f"Test Report:")
    # print(f" Accuracy: {test_report['accuracy']:.4f}")
    # print(f" Macro Avg Precision: {test_report['macro avg']['precision']:.4f}")
    # print(f" Macro Avg Recall: {test_report['macro avg']['recall']:.4f}")
    # print(f" Macro Avg F1-Score: {test_report['macro avg']['f1-score']:.4f}")
    # print(f" Weighted Avg Precision: {test_report['weighted avg']['precision']:.4f}")
    # print(f" Weighted Avg Recall: {test_report['weighted avg']['recall']:.4f}")
    # print(f" Weighted Avg F1-Score: {test_report['weighted avg']['f1-score']:.4f}")

    # FINISHED!


if __name__ == "__main__":
    main()
