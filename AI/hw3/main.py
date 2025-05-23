
import sys
import math

# NOTE THAT THESE TRY EXCEPTS ARE ONLY ADDED SO THAT YOU KNOW
# THAT YOU MUST INSTALL THESE LIBRARIES IF YOU DON"T ALREADY HAVE THEM

try:
    import gymnasium as gym
except:
    print("The gymnasium library is not installed!")
    print("Please install gymnasium in your python environment using:")
    print("\tpip install gymnasium")
    sys.exit(1)

try:
    import numpy as np
except:
    print("The numpy library is not installed!")
    print("Please install numpy in your python environment using:")
    print("\tpip install numpy")
    sys.exit(1)

import csv

from tutorial import generate_random_policy, run_one_experiment, display_policy

# ================================================================================
# 공통 유틸리티 함수 및 상수
# ================================================================================

def generate_env():
    """
    Gymnasium FrozenLake 8x8 환경을 생성합니다.
    과제 요구사항에 따라 8x8 맵, is_slippery=True로 설정됩니다. [1, 9]
    """
    print("Creating FrozenLake 8x8 environment...")
    # Note: render_mode is not needed for evaluation runs without display=True [1]
    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True)
    print("Environment created.")
    return env

def run_one_experiment_wrapper(env, policy, num_episodes, display=False):
    """
    run_one_experiment 함수를 호출하고 결과를 반환합니다.
    이는 evaluate_policy 내부에서 run_one_experiment를 호출하기 위한 래퍼입니다.
    run_one_experiment는 tutorial.py에 정의되어 있습니다. [10, 11]
    """
    # run_one_experiment는 tutorial.py에서 import 됩니다.
    return run_one_experiment(env, policy, num_episodes, display)


def evaluate_policy(env, policy, num_experiments, num_episodes_per_experiment):
    """
    주어진 정책에 대해 여러 번의 실험을 실행하고 성능 지표를 수집합니다.
    각 실험은 지정된 에피소드 수만큼 실행됩니다. [2, 3]

    Args:
        env: Gymnasium 환경 객체.
        policy: 평가할 정책 (1D NumPy 배열).
        num_experiments: 실행할 실험(그룹)의 수 (예: 100). [2, 3]
        num_episodes_per_experiment: 각 실험(그룹)에서 실행할 에피소드 수 (예: 10000). [2, 12]

    Returns:
        tuple:
            - experiment_goals: 각 실험에서 Goal에 도달한 횟수 리스트 (num_experiments 길이). [4, 10]
            - experiment_mean_goal_steps: 각 실험에서 Goal에 도달한 에피소드에 대한 평균 단계 수 리스트 (num_experiments 길이). [4, 13]
    """
    experiment_goals = []
    experiment_mean_goal_steps = []

    print(f"Evaluating policy over {num_experiments} experiments, each with {num_episodes_per_experiment} episodes.")

    for i in range(num_experiments):
        # display=False로 설정하여 시각화 없이 빠르게 실행 [4]
        print(f"  Running experiment {i}")
        goals, holes, total_rewards, total_goal_steps = run_one_experiment_wrapper(
            env, policy, num_episodes_per_experiment, display=False
        )

        # 각 실험에서의 평균 목표 도달 단계 계산 [4]
        # goals가 0인 경우(Goal 도달 에피소드가 없는 경우) 0.0으로 처리
        mean_steps_in_this_experiment = total_goal_steps / goals if goals > 0 else 0.0

        experiment_goals.append(goals)
        experiment_mean_goal_steps.append(mean_steps_in_this_experiment)

        # 진행 상황 출력 (선택 사항)
        if (i + 1) % 10 == 0 or (i + 1) == num_experiments:
             print(f"  Completed {i+1}/{num_experiments} experiments...")

    print("Evaluation complete.")
    return experiment_goals, experiment_mean_goal_steps

def generate_csv(data_rows, filename, header):
    """
    주어진 데이터를 CSV 파일로 저장합니다. [5, 6]

    Args:
        data_rows: CSV에 저장할 데이터 행들의 리스트 (예: [[goal1, mean_steps1], [goal2, mean_steps2], ...]).
        filename: 저장할 파일 이름 (예: "results.csv"). [5]
        header: CSV 파일의 헤더 행 리스트 (예: ['Goals', 'MeanGoalSteps']). [5]
    """
    print(f"Saving experiment results to {filename}...")
    try:
        # 'w' 모드로 파일을 열고 newline=''으로 빈 행 생성을 방지합니다. [5]
        with open(filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(header) # 헤더 쓰기 [5]
            csv_writer.writerows(data_rows) # 데이터 쓰기
        print(f"Data successfully saved to {filename}")
    except IOError as e:
        print(f"Error saving data to {filename}: {e}")


def format_policy_for_display(policy, n_states):
    """
    정책(1D 배열)을 2D 그리드 형태로 변환하고 액션 번호를 방향 화살표로 매핑합니다. [7, 8]

    Args:
        policy: 포맷팅할 정책 (1D NumPy 배열).
        n_states: 상태의 총 개수 (8x8 맵의 경우 64). [14]

    Returns:
        NumPy 배열: 방향 화살표로 표현된 2D 정책 그리드.
    """
    # display_policy 함수를 사용하여 1D 정책을 2D 배열로 변환합니다.
    # display_policy는 tutorial.py에 정의되어 있습니다. [15]
    policy_2d = display_policy(policy, n_states)

    # 액션 번호를 방향 화살표로 매핑합니다.
    # 0:Left, 1:Down, 2:Right, 3:Up [7, 8, 16]
    action_map = {0: '<', 1: 'v', 2: '>', 3: '^'}
    policy_display = np.vectorize(action_map.get)(policy_2d)

    return policy_display

# ================================================================================
# Part 1. Fixed Policies
# ================================================================================

def part_one():
    print("Running Part 1: Evaluating Fixed Policies") [1]

    # 환경 생성 [새로운 함수]
    env = generate_env()
    nS = env.observation_space.n # Number of states [1, 14]
    nA = env.action_space.n # Number of actions [1, 14]

    # 실험 파라미터 [12]
    num_policies_to_try = 10 # 10개의 다른 정책 시도 [2]
    num_experiments_per_policy = 100 # 각 정책당 100번의 실험 실행 [2, 3]
    num_episodes_per_experiment = 10000 # 각 실험당 10,000 에피소드 실행 [2, 12]

    # 정책 성능 데이터 저장을 위한 리스트
    policy_performance_data = [] # To store (mean_goals, seed, policy, experiment_results_list) [27]

    # 10개의 다른 정책 시도 (다른 시드 사용) [2, 12]
    # 과제 예시 seed=17 외의 10개 정책 (예: seeds 1부터 10까지 사용)
    start_seed = 1
    end_seed = start_seed + num_policies_to_try

    for seed in range(start_seed, end_seed):
        print(f"\n--- Testing policy with seed: {seed} ---")

        # 랜덤 정책 생성 (tutorial.py의 함수 사용) [3, 28, 29]
        policy = generate_random_policy(nA, nS, seed=seed)

        # 정책 평가 [새로운 함수]
        # 이 함수가 run_one_experiment를 100번 호출하고 결과를 수집합니다. [3, 4]
        experiment_goals, experiment_mean_goal_steps = evaluate_policy(
            env, policy, num_experiments_per_policy, num_episodes_per_experiment
        )

        # 100개 실험 결과의 평균 목표 도달 횟수 계산 [2, 4]
        mean_goals_for_policy = np.mean(experiment_goals)

        # 정책 데이터 및 결과 저장 [27]
        policy_performance_data.append({
            'seed': seed,
            'policy': policy,
            'mean_goals_across_experiments': mean_goals_for_policy,
            'experiment_goals': experiment_goals, # 100개 실험의 목표 횟수 리스트 저장 [27]
            'experiment_mean_goal_steps': experiment_mean_goal_steps # 100개 실험의 평균 단계 리스트 저장 [27]
        })

        print(f"Policy seed {seed}: Mean Goals across {num_experiments_per_policy} experiments: {mean_goals_for_policy:.2f}")

        # (옵션) 각 정책의 100개 실험 결과를 즉시 CSV로 저장
        # 과제 요구사항에 따라 상위 2개 정책만 저장해도 무방하나, 모든 정책 저장도 가능.
        # 여기서는 상위 2개 정책만 저장하는 로직을 따릅니다.

    # 10개 정책을 평균 목표 도달 횟수 기준으로 정렬 (내림차순) [30, 31]
    policy_performance_data.sort(key=lambda x: x['mean_goals_across_experiments'], reverse=True)

    print("\n--- Top 2 Policies ---")

    # 상위 2개 정책 선택 및 처리 [30, 31]
    # min(2, len(policy_performance_data))는 정책 수가 2개 미만일 경우를 처리합니다.
    for rank in range(min(2, len(policy_performance_data))):
        top_policy_info = policy_performance_data[rank]
        seed = top_policy_info['seed']
        policy = top_policy_info['policy']
        mean_goals = top_policy_info['mean_goals_across_experiments']
        experiment_goals = top_policy_info['experiment_goals']
        experiment_mean_goal_steps = top_policy_info['experiment_mean_goal_steps']

        print(f"\nRank {rank+1}: Policy with Seed {seed} (Mean Goals: {mean_goals:.2f})")

        # 정책을 2D 배열로 시각화 [7, 30] - [새로운 함수 사용]
        print("Policy (2D array):")
        policy_display = format_policy_for_display(policy, nS)
        print(policy_display)

        # 100개 실험 결과를 CSV 파일로 저장 (R 분석용) [사용자 요청, 31] - [새로운 함수 사용]
        csv_filename = f"part1_policy_seed_{seed}.csv"
        # CSV에 저장할 데이터 형식: 각 행은 한 실험의 [Goals, MeanGoalSteps] [5]
        csv_data_rows = [[experiment_goals[i], experiment_mean_goal_steps[i]] for i in range(num_experiments_per_policy)]
        csv_header = ['Goals', 'MeanGoalSteps'] # 헤더 설정 [5]
        generate_csv(csv_data_rows, csv_filename, csv_header)

        # TODO: 추가 통계 계산 및 저장은 R 스크립트에서 수행 (과제 요구사항 및 사용자 요청) [30]
        # TODO: 히스토그램 생성은 R 스크립트에서 수행 (과제 요구사항 및 사용자 요청) [30]

    env.close() # 환경 닫기 [32]
    print("\nPart 1 execution finished.")

# ================================================================================
# Part 2. Optimal Policy by Value Iteration
# ================================================================================
    
# Need a helper function to get terminal states from the environment map
def get_terminal_states(env):
    """
    FrozenLakeEnv 환경의 맵에서 터미널 상태(Holes, Goal)의 인덱스를 찾습니다.
    """
    desc = env.unwrapped.desc.tolist()
    desc_str = [''.join(bytes.decode(c) for c in row) for row in desc]
    flat_desc = ''.join(desc_str)
    # 터미널 상태는 'H' (Hole) 또는 'G' (Goal)입니다.
    terminal_states = [i for i, char in enumerate(flat_desc) if char in b'HG'.decode()] # Decode byte string 'HG' as well
    return terminal_states

def value_iteration(env, gamma=1.0, theta=1e-8):
    """
    FrozenLake 환경에서 Value Iteration 알고리즘을 수행하여 최적 상태 가치 함수와 최적 정책을 찾습니다.

    Args:
        env: Gymnasium FrozenLake 환경 객체. 환경의 P 속성을 통해 동역학을 얻습니다.
        gamma: 감가율 (discount rate). 과제 요구사항에 따라 기본값은 1.0입니다 [3].
        theta: 가치 함수의 수렴을 판단하는 작은 임계값 [의사코드].

    Returns:
        tuple:
            - V: 수렴된 최적 상태 가치 함수 (NumPy 배열, shape=[nS]).
            - policy: 추출된 최적 결정론적 정책 (NumPy 배열, shape=[nS]),
                      각 상태에 대해 최적 행동(0-3)을 나타냅니다.
    """
    print("\nRunning Value Iteration...")

    nS = env.observation_space.n # 상태의 개수 (8x8 = 64) [4]
    nA = env.action_space.n # 행동의 개수 (좌,하,우,상 = 4) [4]
    P = env.P # 환경의 전이 확률 및 보상 정보 딕셔너리 [2]

    # 상태 가치 함수 V(s)를 0으로 초기화합니다 [의사코드].
    # 터미널 상태(Hole, Goal)의 가치는 항상 0으로 유지됩니다.
    V = np.zeros(nS)

    # 터미널 상태 인덱스를 미리 식별하여 업데이트 루프에서 제외합니다.
    # Value Iteration은 비-터미널 상태의 가치만 업데이트합니다.
    terminal_states = get_terminal_states(env)
    # print(f"Terminal states indices: {terminal_states}") # 디버깅용

    # 가치 반복 메인 루프: V(s)가 수렴할 때까지 반복합니다 [의사코드].
    while True:
        delta = 0 # 이번 반복에서 V 함수의 최대 변화량을 추적합니다 [의사코드].

        # 모든 상태 s에 대해 반복합니다 [의사코드].
        for s in range(nS):
            # 터미널 상태는 가치가 0으로 고정되므로 업데이트하지 않고 건너뜀.
            if s in terminal_states:
                continue

            v_old = V[s] # 현재 상태 s의 이전 가치 [의사코드].

            # 상태 s에서 취할 수 있는 각 행동 a에 대한 기대 가치(Q 값)를 계산합니다.
            # V(s)의 새로운 값은 이 기대 가치들 중 최대값이 됩니다 [의사코드].
            q_values = np.zeros(nA)
            for a in range(nA):
                # P[s][a]는 [(prob, next_state, reward, is_terminal), ...] 형태의 리스트입니다 [2].
                # 이 리스트를 순회하며 기대 보상을 계산합니다.
                expected_value_for_action_a = 0
                for prob, next_s, reward, is_terminal in P[s][a]:
                    # 벨만 방정식의 핵심: 전이 확률 * (즉시 보상 + 감마 * 다음 상태의 가치)
                    # 감마는 1.0이므로 할인하지 않습니다 [3].
                    # V[next_s]는 다음 상태 next_s의 현재 추정 가치입니다.
                    expected_value_for_action_a += prob * (reward + gamma * V[next_s])

                q_values[a] = expected_value_for_action_a

            # 벨만 최적 방정식: V(s)는 해당 상태에서 가능한 모든 행동의 기대 가치 중 최대값입니다 [의사코드].
            V[s] = np.max(q_values)

            # 이번 반복에서의 최대 가치 변화량(delta)을 업데이트합니다 [의사코드].
            delta = max(delta, abs(V[s] - v_old))

        # 수렴 여부 확인: 최대 가치 변화량(delta)이 임계값(theta)보다 작으면 반복을 멈춥니다 [의사코드].
        print(f" Value Iteration Loop finished, delta: {delta:.10f}, theta: {theta}") # 디버깅용 출력
        if delta < theta:
            print(" Value Iteration converged.")
            break

    # V(s)가 수렴한 후, 최적 정책 π*(s)를 추출합니다 [1, 2].
    # 각 상태 s에서 Q 값을 계산하고, 최대 Q 값을 주는 행동이 최적 행동이 됩니다 [의사코드].
    policy = np.zeros(nS, dtype=int)
    for s in range(nS):
        # 터미널 상태는 정책이 정의되지 않거나 의미가 없습니다.
        # 편의상 특정 행동(예: 0/Left)을 할당하거나 나중에 표시에서 제외할 수 있습니다.
        # 여기서는 일단 0으로 설정합니다.
        if s in terminal_states:
             policy[s] = 0 # 터미널 상태는 0 (Left)으로 설정 (표시 목적)
             continue

        # 수렴된 V 값을 사용하여 각 행동에 대한 기대 가치를 다시 계산합니다.
        q_values = np.zeros(nA)
        for a in range(nA):
             expected_value_for_action_a = 0
             for prob, next_s, reward, is_terminal in P[s][a]:
                  expected_value_for_action_a += prob * (reward + gamma * V[next_s])
             q_values[a] = expected_value_for_action_a

        # 최적 정책: 해당 상태 s에서 최대 기대 가치를 주는 행동 a [의사코드].
        policy[s] = np.argmax(q_values)

    print("Optimal policy extracted.")
    return V, policy

def part_two():
    print("Running Part 2: Optimal Policy by Value Iteration")

    # 환경 생성 [새로운 함수], Value Iteration을 위해 .unwrapped 필요 [17, 32]
    env = generate_env()
    unwrapped_env = env.unwrapped # Value Iteration은 unwrapped 환경 객체를 사용합니다. [17, 32]
    nS = unwrapped_env.observation_space.n # 상태의 개수 [14]

    # Value Iteration 실행 [19, 20, 32]
    # gamma=1.0, theta=1e-8 사용 (과제 요구사항) [21, 22]
    optimal_V, optimal_policy = value_iteration(unwrapped_env, gamma=1.0, theta=1e-8)

    # 수렴된 V(s) 테이블을 2D 배열로 출력 [32, 33]
    print("\n--- Converged Optimal State-Value Function (V*) ---")
    # 8x8 맵이므로 크기는 8 [8]
    size = int(math.sqrt(nS))
    V_2d = optimal_V.reshape((size, size))
    # V 값 출력 시 소수점 포맷팅 [8]
    print(np.array2string(V_2d, formatter={'float_kind':lambda x: "%.4f" % x}))

    print("\n--- Extracted Optimal Policy ---")
    # 최적 정책을 2D 배열로 시각화 [8, 33] - [새로운 함수 사용]
    policy_display = format_policy_for_display(optimal_policy, nS)
    print(policy_display)

    # TODO: Evaluate the optimal policy and generate histogram (similar to Part 1) [6, 33]
    # 최적 정책 평가 [새로운 함수 사용]
    # Part 1과 동일한 평가 설정 사용 가정 (100 실험, 각 10,000 에피소드) [2]
    num_experiments_for_optimal = 100
    num_episodes_per_experiment = 10000
    optimal_policy_goals, optimal_policy_mean_goal_steps = evaluate_policy(
        env, optimal_policy, num_experiments_for_optimal, num_episodes_per_experiment
    )

    # 최적 정책의 100개 실험 결과를 CSV 파일로 저장 (R 분석용) [6] - [새로운 함수 사용]
    csv_filename = "part2_optimal_policy.csv"
    # CSV에 저장할 데이터 형식: 각 행은 한 실험의 [Goals, MeanGoalSteps] [6]
    csv_data_rows = [[optimal_policy_goals[i], optimal_policy_mean_goal_steps[i]] for i in range(num_experiments_for_optimal)]
    csv_header = ['Goals', 'MeanGoalSteps'] # 헤더 설정
    generate_csv(csv_data_rows, csv_filename, csv_header)

    # TODO: 추가 통계 계산 및 저장은 R 스크립트에서 수행 (과제 요구사항 및 사용자 요청) [33]
    # TODO: 히스토그램 생성은 R 스크립트에서 수행 (과제 요구사항 및 사용자 요청) [33]

    env.close() # 환경 닫기 [6]
    print("\nPart 2 execution finished.")

# ================================================================================
# Main
# ================================================================================
#   
def main():
    # TODO: feel free to change this as required
    # TODO: also, check tutorial.py for some hints on how to implement your experiments
    # parst_one()
    part_two()

if __name__ == "__main__":
    main()