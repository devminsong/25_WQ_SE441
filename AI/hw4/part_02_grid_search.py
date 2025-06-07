
"""
# ===============================================
#  Created by: Kenny Davila Castellanos
#              for CSC 480
#
#  MODIFIED BY: [Min Song]
# ===============================================
"""

import sys
import itertools

from auxiliary_functions import *
from part_01_cross_validation import cross_validation, custom_metric_print

import time
import csv

def print_table(classifier_name, in_raw_data_filename, results):
       # 헤더 출력 (분류기 유형에 따라 동적으로 변경)
    # 훈련 데이터셋 이름 추출 (예: 'training_data_small.csv' -> 'small')
    filename_without_ext = in_raw_data_filename.replace('.csv', '')
    parts = filename_without_ext.split('_')
    dataset_name = '_'.join(parts[2:]).upper() # 'training_data_'를 제외한 나머지 부분을 추출

    # 각 컬럼의 헤더 이름, 너비, 정렬을 정의합니다.
    # 숫자 데이터는 오른쪽 정렬(>), 문자열 데이터는 왼쪽 정렬(<)
    header_parts = []
    data_format_parts = [] # 데이터를 최종적으로 포맷팅할 때 사용할 포맷 문자열 목록

    if classifier_name == "decision_tree":
        header_parts = [
            f"{'Dataset':<10}", f"{'Criterion':<10}", f"{'Depth':<8}",
            f"{'Train Acc.':>12}", f"{'Val. Acc.':>12}", f"{'Val. Avg Rec.':>15}",
            f"{'Val. Avg Prec.':>15}", f"{'Val. Avg F1':>12}", f"{'Time Train':>12}", f"{'Time Val.':>12}"
        ]
        data_format_parts = [
            "{:<10}", "{:<10}", "{:<8}",  # Dataset, Criterion, Depth
            "{:>12}", "{:>12}", "{:>15}", "{:>15}", "{:>12}",  # Accuracies, Recalls, Precisions, F1-scores
            "{:>12}", "{:>12}" # Times
        ]
    elif classifier_name == "random_forest":
        header_parts = [
            f"{'Dataset':<10}", f"{'N trees':<10}", f"{'Depth':<8}",
            f"{'Train Acc.':>12}", f"{'Val. Acc.':>12}", f"{'Val. Avg Rec.':>15}",
            f"{'Val. Avg Prec.':>15}", f"{'Val. Avg F1':>12}", f"{'Time Train':>12}", f"{'Time Val.':>12}"
        ]
        data_format_parts = [
            "{:<10}", "{:<10}", "{:<8}",  # Dataset, N trees, Depth
            "{:>12}", "{:>12}", "{:>15}", "{:>15}", "{:>12}",
            "{:>12}", "{:>12}"
        ]
    elif classifier_name == "logistic_classifier":
        header_parts = [
            f"{'Dataset':<10}", f"{'Penalty':<10}", f"{'C':>8}", # 'C' 값은 숫자이므로 헤더도 오른쪽 정렬
            f"{'Train Acc.':>12}", f"{'Val. Acc.':>12}", f"{'Val. Avg Rec.':>15}",
            f"{'Val. Avg Prec.':>15}", f"{'Val. Avg F1':>12}", f"{'Time Train':>12}", f"{'Time Val.':>12}"
        ]
        data_format_parts = [
            "{:<10}", "{:<10}", "{:>8}",  # Dataset, Penalty, C
            "{:>12}", "{:>12}", "{:>15}", "{:>15}", "{:>12}",
            "{:>12}", "{:>12}"
        ]
    else:
        print("Unknown classifier type.")
        return

    # 헤더 출력
    print(" ".join(header_parts))

    # 구분선 출력 (헤더 파트들의 총 너비에 맞춰 동적으로 생성)
    separator_line_length = 0
    for h_part in header_parts:
        separator_line_length += len(h_part)
    separator_line_length += len(header_parts) - 1 # 각 컬럼 사이의 공백만큼 추가
    print("-" * separator_line_length)

    # 각 결과 조합을 반복하며 테이블 형식으로 출력
    for res in results:
        combo = res["hyperparameters"]
        metrics = res["metrics"]
        total_time = res["total_run_time"] # 교차 검증에 소요된 총 시간

        # 모든 메트릭 값을 '%.4f%' 형식의 문자열로 미리 포매팅
        train_acc_str = f"{metrics['train']['accuracy']:.4f}%"
        val_acc_str = f"{metrics['validation']['accuracy']:.4f}%"
        val_avg_rec_str = f"{metrics['validation']['macro avg']['recall']:.4f}%"
        val_avg_prec_str = f"{metrics['validation']['macro avg']['precision']:.4f}%"
        val_avg_f1_str = f"{metrics['validation']['macro avg']['f1-score']:.4f}%"
        
        # 시간 값은 '%.2f s' 형식의 문자열로 포매팅 (보고서 예시에 따라 동일한 총 실행 시간을 훈련/검증 시간에 적용)
        time_str = f"{total_time:.2f} s"

        # 현재 행의 데이터를 리스트로 구성
        row_data_values = [dataset_name]

        if classifier_name == "decision_tree":
            criterion = combo.get("criterion", "N/A")
            max_depth = combo.get("max_depth", "N/A")
            # max_depth가 None일 경우 'N/A' 문자열로 변환
            max_depth_str = str(max_depth) if max_depth is not None else "N/A" 
            row_data_values.extend([criterion, max_depth_str])
        elif classifier_name == "random_forest":
            n_trees = combo.get("n_trees", "N/A")
            max_depth = combo.get("max_depth", "N/A")
            max_depth_str = str(max_depth) if max_depth is not None else "N/A"
            row_data_values.extend([n_trees, max_depth_str])
        elif classifier_name == "logistic_classifier":
            penalty = combo.get("penalty", "N/A")
            c_value = combo.get("C", "N/A")
            # C 값은 보고서 예시(Table 3)에 따라 소수점 한 자리로 포매팅
            c_value_str = f"{c_value:.1f}" if isinstance(c_value, (int, float)) else str(c_value)
            
            # penalty가 None일 수 있으므로, 명시적으로 문자열로 변환합니다.
            # 이렇게 하면 NoneType.__format__ 메서드가 호출되지 않고,
            # "None" 문자열이 포매팅됩니다.
            row_data_values.extend([str(penalty), c_value_str]) # <--- 이 라인을 수정했습니다

        # 공통 메트릭 문자열 추가
        row_data_values.extend([
            train_acc_str, val_acc_str, val_avg_rec_str, val_avg_prec_str, val_avg_f1_str,
            time_str, time_str # Time Train과 Time Val을 동일하게 사용
        ])

        # 정의된 포맷 문자열에 따라 각 데이터를 포매팅하고 공백으로 연결하여 출력
        formatted_row_output = [data_format_parts[j].format(item) for j, item in enumerate(row_data_values)]
        print(" ".join(formatted_row_output))

def save_results_to_csv(classifier_name, in_raw_data_filename, results, output_csv_filename):
    """
    그리드 서치 결과를 CSV 파일로 저장합니다.
    """
    filename_without_ext = in_raw_data_filename.replace('.csv', '')
    parts = filename_without_ext.split('_')
    dataset_name = '_'.join(parts[2:]).upper() # 'training_data_'를 제외한 나머지 부분을 추출

    # CSV 헤더 정의
    header = []
    if classifier_name == "decision_tree":
        header = ["Dataset", "Criterion", "Depth", "Train_Acc", "Val_Acc", "Val_Avg_Rec", "Val_Avg_Prec", "Val_Avg_F1", "Time_Train", "Time_Val"]
    elif classifier_name == "random_forest":
        header = ["Dataset", "N_trees", "Depth", "Train_Acc", "Val_Acc", "Val_Avg_Rec", "Val_Avg_Prec", "Val_Avg_F1", "Time_Train", "Time_Val"]
    elif classifier_name == "logistic_classifier":
        header = ["Dataset", "Penalty", "C", "Train_Acc", "Val_Acc", "Val_Avg_Rec", "Val_Avg_Prec", "Val_Avg_F1", "Time_Train", "Time_Val"]
    else:
        print("Unknown classifier type for CSV export.")
        return

    # CSV 파일 쓰기
    with open(output_csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header) # 헤더 쓰기

        for res in results:
            combo = res["hyperparameters"]
            metrics = res["metrics"]
            total_time = res["total_run_time"]

            # 메트릭 값을 문자열로 포매팅 (4자리 유효숫자, 시간은 2자리 소수점)
            train_acc_str = f"{metrics['train']['accuracy']:.4f}" # % 기호 제외
            val_acc_str = f"{metrics['validation']['accuracy']:.4f}" # % 기호 제외
            val_avg_rec_str = f"{metrics['validation']['macro avg']['recall']:.4f}"
            val_avg_prec_str = f"{metrics['validation']['macro avg']['precision']:.4f}"
            val_avg_f1_str = f"{metrics['validation']['macro avg']['f1-score']:.4f}"
            time_str = f"{total_time:.2f}"

            row_data_values = [dataset_name]

            if classifier_name == "decision_tree":
                criterion = combo.get("criterion", "N/A")
                max_depth = combo.get("max_depth", "N/A")
                max_depth_str = str(max_depth) if max_depth is not None else "N/A"
                row_data_values.extend([criterion, max_depth_str])
            elif classifier_name == "random_forest":
                n_trees = combo.get("n_trees", "N/A")
                max_depth = combo.get("max_depth", "N/A")
                max_depth_str = str(max_depth) if max_depth is not None else "N/A"
                row_data_values.extend([n_trees, max_depth_str])
            elif classifier_name == "logistic_classifier":
                penalty = combo.get("penalty", "N/A")
                c_value = combo.get("C", "N/A")
                c_value_str = f"{c_value:.1f}" if isinstance(c_value, (int, float)) else str(c_value)
                row_data_values.extend([penalty, c_value_str])

            row_data_values.extend([
                train_acc_str, val_acc_str, val_avg_rec_str, val_avg_prec_str, val_avg_f1_str,
                time_str, time_str
            ])
            csv_writer.writerow(row_data_values) # 데이터 행 쓰기
    print(f"\nResults saved to {output_csv_filename}")

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

    # TODO: 1) Load your (grid-search) hyper-parameters
    classifier_name, hyper_params = load_hyperparameters(in_config_filename, "grid_search")
    print(f"Current Classifier: {classifier_name}")
    print(f"Hyperparameters: {hyper_params}")

    # TODO: 2) Load your data
    raw_x, raw_y = load_raw_dataset(in_raw_data_filename)
    print(f"raw_x: {raw_x.shape}")
    print(f"raw_y: {raw_y.shape}")

    # TODO: 3) generate a combination of parameters (check itertools.product)
    #          https://docs.python.org/3/library/itertools.html#itertools.product
    hyper_param_keys = hyper_params.keys()
    hyper_param_values = hyper_params.values()

    temp_combos = list(itertools.product(*hyper_param_values))
    hyper_param_combinations = []

    for combo in temp_combos:
        config_dict = dict(zip(hyper_param_keys, combo))
        hyper_param_combinations.append(config_dict)

    print(f"\n{len(hyper_param_combinations)} hyperparameter combinations generated.")
    for i, combo in enumerate(hyper_param_combinations):
        print(f"  Combination {i+1}: {combo}")
    
    # TODO: 4) Use the combinations of parameters to run a grid search
    #       For each combination of parameters
    #        3.1) create a custom config for this combination
    #        3.2) run full cross-validation for this combination of parameters (re-use function from part 01)
    #        3.3) collect results
    highest_f1_score = -1.0 # 최고 F1 점수를 추적하기 위한 변수 초기화 (F1 점수는 0에서 1 사이)
    highest_config = None # 최상의 하이퍼파라미터 구성을 저장할 변수
    highest_train_results = None # 최상의 구성에 대한 훈련 결과를 저장할 변수
    highest_validation_results = None # 최상의 구성에 대한 검증 결과를 저장할 변수
    results = []

    print("\n=== Running Grid Search Cross-Validation ===")
    for i, combo in enumerate(hyper_param_combinations):
        print(f"\nEvaluating Combination {i+1}/{len(hyper_param_combinations)}: {combo}")
        
        start_time = time.time()        
        current_metrics = cross_validation(raw_x, raw_y, n_folds, classifier_name, combo)        
        end_time = time.time()
        total_run_time = end_time - start_time

        current_validation_f1 = current_metrics["validation"]["macro avg"]["f1-score"]

        print(f"  Validation Macro F1 for this combination: {current_validation_f1:.4f}")
        print(f"  Total run time for this combination: {total_run_time:.2f} seconds")

        results.append({
            "hyperparameters": combo,
            "metrics": current_metrics,
            "total_run_time": total_run_time
        })
        
        if current_validation_f1 > highest_f1_score:
            highest_f1_score = current_validation_f1
            highest_config = combo
            highest_train_results = current_metrics["train"]
            highest_validation_results = current_metrics["validation"]

    # TODO: 4) print the best parameters found (based on highest validation macro f-1 score)
    # print("\n\nBest parameters found")
    # print(f"\t-> Best configuration: ???")
    # print(f"\t-> Best configuration Results (Training):")
    # print("\t\t YOUR RESULTS HERE")
    # print(f"\t-> Best configuration Results (Validation):")
    # print("\t\t YOUR RESULTS HERE")

    print("\n\n--- Best parameters found ---")
    print(f"\t-> Best configuration: {highest_config}")    
    print(f"\t-> Best configuration Results (Training):")
    if highest_train_results:
        print(f"\t\tAccuracy: {highest_train_results['accuracy']:.4f}")
        print(f"\t\tMacro Avg Precision: {highest_train_results['macro avg']['precision']:.4f}")
        print(f"\t\tMacro Avg Recall: {highest_train_results['macro avg']['recall']:.4f}")
        print(f"\t\tMacro Avg F1-Score: {highest_train_results['macro avg']['f1-score']:.4f}")
    else:
        print("\t\tNo training results available.")
    
    print(f"\t-> Best configuration Results (Validation):")
    if highest_validation_results:
        print(f"\t\tAccuracy: {highest_validation_results['accuracy']:.4f}")
        print(f"\t\tMacro Avg Precision: {highest_validation_results['macro avg']['precision']:.4f}")
        print(f"\t\tMacro Avg Recall: {highest_validation_results['macro avg']['recall']:.4f}")
        print(f"\t\tMacro Avg F1-Score: {highest_validation_results['macro avg']['f1-score']:.4f}")
    else:
        print("\t\tNo validation results available.")

    print_table(classifier_name, in_raw_data_filename, results)
    output_csv_filename = f"result.csv"
    save_results_to_csv(classifier_name, in_raw_data_filename, results, output_csv_filename)
    
    # FINISHED!
    print("\n=== FINISHED ===")

if __name__ == "__main__":
    main()