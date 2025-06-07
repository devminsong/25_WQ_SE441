
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
    if len(sys.argv) < 5:
        print("Usage:")
        print(f"\tpython {sys.argv[0]} in_config in_raw_data out_normalizer out_classifier")
        return

    in_config_filename = sys.argv[1]
    in_raw_data_filename = sys.argv[2]
    out_normalizer_filename = sys.argv[3]
    out_classifier_filename = sys.argv[4]

    # TODO: 1) Load your (training) hyper-parameters
    # 'auxiliary_functions.py'의 load_hyperparameters 함수를 사용하여 설정 파일에서 훈련용 하이퍼파라미터를 로드합니다 [5-9].
    # 'Part 3. Training'에서는 그리드 서치에서 찾은 최적의 하이퍼파라미터 조합을 'training' 설정에 수동으로 설정해야 합니다 [10].
    classifier_name, hyper_params = load_hyperparameters(in_config_filename, "training")
    print(f"Current Classifier: {classifier_name}")
    print(f"Hyperparameters: {hyper_params}")

    # TODO: 2) Load your data
    # 'auxiliary_functions.py'의 load_raw_dataset 함수를 사용하여 지정된 훈련 데이터셋 파일을 로드합니다 [9, 13-17].
    # 이 함수는 원시 데이터 X와 레이블 Y를 NumPy 배열 형태로 반환합니다 [17-19].
    raw_x, raw_y = load_raw_dataset(in_raw_data_filename) # part_02_grid_search에서 유사한 변수명을 사용했습니다.
    print(f"raw_x: {raw_x.shape}")
    print(f"raw_y: {raw_y.shape}")

    # TODO: 3) normalize the data ...
    # 'auxiliary_functions.py'의 apply_normalization 함수를 사용하여 데이터를 정규화합니다 [17, 21, 22].
    # 이 단계에서는 새로운 StandardScaler를 생성하고 훈련 데이터에 맞춰 피팅(fit)한 후 데이터를 변환(transform)해야 합니다 [23].
    # 훈련된 StandardScaler 객체는 다음 단계(Part 4. Testing)에서 재사용하기 위해 저장해야 합니다 [24].
    normalized_x, scaler = apply_normalization(raw_x, None) # scaler=None으로 호출하여 새 StandardScaler 생성 및 피팅
    normalized_y = raw_y # Y(레이블)는 정규화할 필요가 없습니다.
    print(f"Normalized X shape: {normalized_x.shape}")

    # TODO: 4) train your classifier on the training split
    # 'auxiliary_functions.py'의 'train_classifier' 함수를 사용하여 정규화된 훈련 데이터로 분류기를 훈련시킵니다.
    # 이 함수는 설정 파일에서 로드된 분류기 이름과 하이퍼파라미터를 사용하여 모델을 생성하고 훈련합니다.
    print(f"\nTraining {classifier_name} classifier...")
    classifier = train_classifier(classifier_name, hyper_params, normalized_x, normalized_y)
    if classifier is None:
        print(f"Error: Failed to train classifier {classifier_name}")
        return
    print(f"Classifier trained successfully.")

    # TODO: 5) evaluate your classifier on the training dataset (compute and print metrics)
    # 훈련된 분류기(classifier)를 사용하여 정규화된 훈련 데이터(normalized_x)에 대한 예측을 수행합니다. [14]
    train_predictions = classifier.predict(normalized_x) 

    # 실제 레이블(normalized_y)과 예측된 레이블(train_predictions)을 비교하여 분류 보고서를 생성합니다.
    # 'output_dict=True'는 결과를 딕셔너리 형태로 반환하도록 하여 프로그램 내에서 사용하기 용이하게 합니다. [15]
    # 'zero_division=0'은 나눗셈 오류를 방지하고 0으로 처리하도록 합니다. [15]
    # 'target_names'는 레이블 0.0과 1.0이 각각 'java'와 'python'에 매핑되도록 합니다. [15-17]
    print(f"\nEvaluating classifier on the training dataset...")
    training_report = classification_report(normalized_y, train_predictions,
                                            output_dict=True, zero_division=0,
                                            target_names=['java', 'python']) 

    print_metrics(training_report)
    # # 훈련 데이터셋에 대한 평가 지표를 출력합니다.
    # print(f"Training Report for {classifier_name}:")
    # print(f"  Accuracy: {training_report['accuracy']:.4f}") 
    # print(f"  Macro Avg Precision: {training_report['macro avg']['precision']:.4f}")
    # print(f"  Macro Avg Recall: {training_report['macro avg']['recall']:.4f}") 
    # print(f"  Macro Avg F1-Score: {training_report['macro avg']['f1-score']:.4f}") 
    # print(f"  Weighted Avg Precision: {training_report['weighted avg']['precision']:.4f}")
    # print(f"  Weighted Avg Recall: {training_report['weighted avg']['recall']:.4f}")
    # print(f"  Weighted Avg F1-Score: {training_report['weighted avg']['f1-score']:.4f}")
    # print(f"  Java Class Metrics:")
    # print(f"    Precision: {training_report['java']['precision']:.4f}")
    # print(f"    Recall: {training_report['java']['recall']:.4f}")
    # print(f"    F1-Score: {training_report['java']['f1-score']:.4f}")
    # print(f"    Support: {training_report['java']['support']}")
    # print(f"  Python Class Metrics:")
    # print(f"    Precision: {training_report['python']['precision']:.4f}")
    # print(f"    Recall: {training_report['python']['recall']:.4f}")
    # print(f"    F1-Score: {training_report['python']['f1-score']:.4f}")
    # print(f"    Support: {training_report['python']['support']}")

    # TODO: 6) save the classifier and the standard scaler ... (pickle library is fine)
    # 훈련된 분류기(classifier)와 표준 정규화기(StandardScaler)를 파일로 저장합니다. [6]
    # 'wb' 모드(write binary)로 파일을 열어 객체를 바이너리 형태로 저장합니다.
    print(f"\nSaving trained classifier to {out_classifier_filename}...")
    try:
        with open(out_classifier_filename, 'wb') as f:
            pickle.dump(classifier, f)
        print("Classifier saved successfully.")
    except Exception as e:
        print(f"Error saving classifier: {e}")

    print(f"Saving trained normalizer to {out_normalizer_filename}...")
    try:
        with open(out_normalizer_filename, 'wb') as f:
            pickle.dump(scaler, f)
        print("Normalizer saved successfully.")
    except Exception as e:
        print(f"Error saving normalizer: {e}")

    # FINISHED!

if __name__ == "__main__":
    main()
