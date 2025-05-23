
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

def part_one():
    print("Running Part 1: Evaluating Fixed Policies")

    # Create a FrozenLake 8x8 environment using Gymnasium
    # Note: render_mode is not needed for evaluation runs without display=True
    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True)
    nS = env.observation_space.n # Number of states [13]
    nA = env.action_space.n # Number of actions [13]

    # Experiment parameters [1]
    num_policies_to_try = 10
    num_experiments_per_policy = 100
    num_episodes_per_experiment = 10000 # Each experiment runs 10,000 episodes

    policy_performance_data = [] # To store (mean_goals, seed, policy, experiment_results_list)

    # Try 10 other policies using different seed numbers [1]
    # The example policy in HW is seed=17 [2]. Let's use seeds 18 through 27 for "10 other policies".
    start_seed = 1
    end_seed = start_seed + num_policies_to_try

    for seed in range(start_seed, end_seed):
        print(f"Testing policy with seed: {seed}")
        policy = generate_random_policy(nA, nS, seed=seed)

        experiment_goals = [] # List to store goals from each of the 100 experiments
        experiment_mean_goal_steps = [] # List to store mean steps from each of the 100 experiments

        # Run the experiment 100 times (each time is 10,000 episodes) [1]
        for i in range(num_experiments_per_policy):
            print(f"  Running experiment {i}/{num_experiments_per_policy} for seed {seed}...")
            goals, holes, total_rewards, total_goal_steps = run_one_experiment(
                env, policy, num_episodes_per_experiment, display=False
            )

            # Compute mean goal steps for this specific experiment run [2, 12]
            # This is total steps in successful episodes divided by the number of successful episodes
            mean_steps_in_this_experiment = total_goal_steps / goals if goals > 0 else 0.0 # Handle division by zero [11]

            experiment_goals.append(goals)
            experiment_mean_goal_steps.append(mean_steps_in_this_experiment)

        # Compute mean goals over the 100 experiments for this policy [2]
        mean_goals_for_policy = np.mean(experiment_goals)

        # Store policy data and results [2]
        policy_performance_data.append({
            'seed': seed,
            'policy': policy,
            'mean_goals_across_experiments': mean_goals_for_policy,
            'experiment_goals': experiment_goals, # Store list of 100 goal counts
            'experiment_mean_goal_steps': experiment_mean_goal_steps # Store list of 100 mean step values
        })

        print(f"  Policy seed {seed}: Mean Goals across {num_experiments_per_policy} experiments: {mean_goals_for_policy:.2f}")

    # Sort policies by mean goals in descending order to find TOP TWO [2]
    policy_performance_data.sort(key=lambda x: x['mean_goals_across_experiments'], reverse=True)

    print("\n--- Top 2 Policies ---")

    # Select and process the TOP TWO policies [2]  
    for rank in range(min(2, len(policy_performance_data))):
        top_policy_info = policy_performance_data[rank]
        seed = top_policy_info['seed']
        policy = top_policy_info['policy']
        mean_goals = top_policy_info['mean_goals_across_experiments']
        experiment_goals = top_policy_info['experiment_goals']
        experiment_mean_goal_steps = top_policy_info['experiment_mean_goal_steps']

        print(f"\nRank {rank+1}: Policy with Seed {seed} (Mean Goals: {mean_goals:.2f})")

        # Display the policy in a 2D array [2]
        policy_2d = display_policy(policy, nS)
        print("Policy (2D array):")
        # Convert policy actions (0, 1, 2, 3) to directional arrows or letters for better visualization
        # 0:Left, 1:Down, 2:Right, 3:Up [13]
        action_map = {0: '<', 1: 'v', 2: '>', 3: '^'}
        policy_display = np.vectorize(action_map.get)(policy_2d)
        print(policy_display)

        # Save the results of the 100 experiments to a CSV file for R analysis [사용자 요청]
        csv_filename = f"part1_policy_{seed}.csv"
        print(f"Saving experiment results for R analysis to {csv_filename}")
        with open(csv_filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            # Write header row, matching potential R script expectations [8]
            csv_writer.writerow(['Goals', 'MeanGoalSteps'])
            # Write data rows (one row per experiment run)
            for i in range(num_experiments_per_policy):
                 # Saving the goals count and the mean steps per successful episode for each 10k-episode run
                csv_writer.writerow([experiment_goals[i], experiment_mean_goal_steps[i]])

        print(f"Data for Policy Seed {seed} saved to {csv_filename}")

    env.close() # Close environment

# 이제 part_two 함수 내에서 위 value_iteration 함수를 호출하고
# 반환된 V와 policy를 사용하여 요구되는 출력 (2D 배열, 히스토그램)을 생성하면 됩니다.
# 예:
def part_two():
    print("Running Part 2: Optimal Policy by Value Iteration")
    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True).unwrapped

    # Value Iteration 실행
    optimal_V, optimal_policy = value_iteration(env, gamma=1.0, theta=1e-8)

    # 결과 출력 (2D 배열)
    print("\n--- Converged Optimal State-Value Function (V*) ---")
    size = int(math.sqrt(env.observation_space.n)) # 8x8 맵이므로 크기는 8
    V_2d = optimal_V.reshape((size, size))
    # V 값 출력 시 소수점 포맷팅
    print(np.array2string(V_2d, formatter={'float_kind':lambda x: "%.4f" % x}))

    print("\n--- Extracted Optimal Policy ---")
    # display_policy 함수를 사용하여 정책을 2D 배열로 출력 (main.txt에 있음)
    # 액션 매핑 0:Left, 1:Down, 2:Right, 3:Up [5]
    action_map = {0: '<', 1: 'v', 2: '>', 3: '^'}
    policy_display = np.vectorize(action_map.get)(optimal_policy.reshape((size, size)))
    print(policy_display)

    # TODO: Evaluate the optimal policy and generate histogram (similar to Part 1) [3]
    # You'll need to run_one_experiment multiple times using optimal_policy
    # Collect goal counts and mean steps for each run
    # Save to a CSV file
    # Generate histogram from the CSV data (likely using R or Python plotting library as you did in Part 1)

    env.close()
    print("\nPart 2 execution finished.")

# def part_two():
#     # TODO: your code here ...
#     pass

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

def main():
    # TODO: feel free to change this as required
    # TODO: also, check tutorial.py for some hints on how to implement your experiments
    # parst_one()
    part_two()


if __name__ == "__main__":
    main()

