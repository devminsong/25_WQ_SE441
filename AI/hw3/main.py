
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
# Common utility functions
# ================================================================================

def generate_env(id='FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True):
    """
    Creates a Gymnasium FrozenLake 8x8 environment.
    Sets the map to 8x8 and is_slippery=True as per the assignment requirements.
    """
    print("Generating environment...")
    env = gym.make(id=id, desc=desc, map_name=map_name, is_slippery=is_slippery)
    print("Environment created.")
    return env

def evaluate_policy(env, policy, num_experiments, num_episodes_per_experiment):
    """
    Executes multiple experiments for a given policy and collects performance metrics.
    Each experiment runs for a specified number of episodes.

    Args:
        env: Gymnasium environment object.
        policy: The policy to evaluate.
        num_experiments: The number of experiments to run (e.g., 100).
        num_episodes_per_experiment: The number of episodes to run in each experiment (e.g., 10000).

    Returns:
        tuple:
            - goals_list: A list of the number of times the Goal was reached in each experiment.
            - mean_total_goal_steps_list: A list of the average number of steps to reach the Goal for successful episodes in each experiment.
    """
    goals_list = []
    mean_total_goal_steps_list = []

    print(f"Evaluating policy over {num_experiments} experiments, each with {num_episodes_per_experiment} episodes.")

    for i in range(num_experiments):
        print(f"Running experiment {i}")
        goals, holes, total_rewards, total_goal_steps = run_one_experiment(
            env, policy, num_episodes_per_experiment, display=False
        )

        goals_list.append(goals)
        mean_total_goal_steps = total_goal_steps / goals if goals > 0 else 0.0
        mean_total_goal_steps_list.append(mean_total_goal_steps)

        if (i + 1) % 10 == 0 or (i + 1) == num_experiments:
             print(f"Completed {i+1}/{num_experiments} experiments...")

    print("Evaluation complete.")
    return goals_list, mean_total_goal_steps_list

def generate_csv(data_rows, filename, header):
    """
    Saves the given data to a CSV file.

    Args:
        data_rows: A list of data rows to save to the CSV (e.g., [[goal1, mean_steps1], [goal2, mean_steps2], ...]).
        filename: The name of the file to save to (e.g., "results.csv").
        header: A list of headers for the CSV file (e.g., ['Goals', 'MeanGoalSteps']).
    """
    print(f"Saving experiment results to {filename}...")
    try:
        with open(filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(header)
            csv_writer.writerows(data_rows)
        print(f"Data successfully saved to {filename}")
    except IOError as e:
        print(f"Error saving data to {filename}: {e}")

def display_formatted_policy(target, square_size):
    side_length = int(math.sqrt(square_size))
    reshaped = target.reshape((side_length, side_length))
    print(np.array2string(reshaped, formatter={'float_kind':lambda x: "%.4f" % x}))

num_policies_to_try = 10 
num_experiments_per_policy = 100 
num_episodes_per_experiment = 10000 

# ================================================================================
# Part 1. Fixed Policies
# ================================================================================

def part_one():
    print("Running Part 1: Evaluating Fixed Policies")

    env = generate_env('FrozenLake-v1', None, "8x8", True)
    nS = env.observation_space.n
    nA = env.action_space.n

    # List to store policy performance data
    policy_performance_data = [] 

    for seed in num_policies_to_try:
        print(f"\n--- Testing policy with seed: {seed} ---")

        policy = generate_random_policy(nA, nS, seed=seed)

        goals_list, mean_total_goal_steps_list = evaluate_policy(
            env, policy, num_experiments_per_policy, num_episodes_per_experiment
        )

        mean_goals = np.mean(goals_list)

        policy_performance_data.append({
            'seed': seed,
            'policy': policy,
            'mean_goals': mean_goals,
            'goals_list': goals_list,
            'mean_total_goal_steps_list': mean_total_goal_steps_list
        })

        print(f"Policy seed {seed}: Mean Goals across {num_experiments_per_policy} experiments: {mean_goals:.2f}")

    # Sort the 10 policies based on the average number of goals reached (descending)
    policy_performance_data.sort(key=lambda x: x['mean_goals'], reverse=True)

    print("\n--- Top 2 Policies ---")
    # Select and process the top 2 policies
    for rank in range(len(policy_performance_data)):
        top_policy_info = policy_performance_data[rank]
        seed = top_policy_info['seed']
        policy = top_policy_info['policy']
        mean_goals = top_policy_info['mean_goals']
        goals_list = top_policy_info['goals_list']
        mean_total_goal_steps_list = top_policy_info['mean_total_goal_steps_list']

        print(f"\nRank {rank+1}: Policy with Seed {seed} (Mean Goals: {mean_goals:.2f})")

        print("Policy (2D array):")
        policy_display = display_policy(policy, nS)
        print(policy_display)

        # Save the results to a CSV file
        csv_filename = f"part1_policy_seed_{seed}.csv"
        csv_data_rows = [[goals_list[i], mean_total_goal_steps_list[i]] for i in range(num_experiments_per_policy)]
        csv_header = ['Goals', 'MeanGoalSteps']
        generate_csv(csv_data_rows, csv_filename, csv_header)

    env.close()
    print("\nPart 1 execution finished.")

# ================================================================================
# Part 2. Optimal Policy by Value Iteration
# ================================================================================
    
def get_terminal_states(env):
    terminal_states = []
    desc = env.desc
    for i in range(desc.shape[0]):
        for j in range(desc.shape[1]):
            tile = desc[i][j].decode('utf-8')  
            if tile in ['H', 'G']:  # If Hole or Goal
                state = i * desc.shape[1] + j
                terminal_states.append(state)

    print(f'terminal {terminal_states}')
    return terminal_states

def value_iteration(env, gamma=1.0, theta=1e-8):
    """
    Performs the Value Iteration algorithm on the FrozenLake environment
    to find the optimal state-value function and the optimal policy.
    (Referenced pseudocode from Sutton & Barto, 2nd ed, chapter 4.4)

    Args:
        env: Gymnasium FrozenLake environment object.
        gamma: The discount rate. Defaults to 1.0 as per the assignment requirements.
        theta: A small threshold value to determine convergence of the value function.

    Returns:
        tuple:
            - V: The converged optimal state-value function.
            - policy: The extracted optimal deterministic policy.
    """
    print("\nRunning Value Iteration...")

    nS = env.observation_space.n
    nA = env.action_space.n 
    P = env.P

    # Initialize V(s), V(terminal)=0
    V = np.zeros(nS)

    # To ensure Value Iteration only updates the values of non-terminal states.
    terminal_states = get_terminal_states(env)

    # Loop
    while True:
        delta = 0 # Δ←0

        # Loop for each s∈S:
        for s in range(nS):
            if s in terminal_states:
                continue

            v_old = V[s] # v←V(s)
            q_values = np.zeros(nA)

            # V(s) ← max_a ∑_{s', r} p(s', r | s, a) * [r + γ * V(s')]           
            # for all actions
            for a in range(nA):
                q_value = 0
                # ∑_{s', r}
                # Q(s, a) += p(s', r | s, a) * [r + γ * V(s')]
                for probability, next_state, reward, is_terminal in P[s][a]:
                    q_value += probability * (reward + gamma * V[next_state])
                q_values[a] = q_value            
            # V(s) ← max_a Q(s, a) 
            V[s] = np.max(q_values)
            
            # Δ←max(Δ,∣v−V(s)∣)
            delta = max(delta, abs(v_old - V[s]))

        # until Δ<θ
        print(f"Value Iteration Loop finished, delta: {delta:.10f}, theta: {theta}")
        if delta < theta:
            print("Value Iteration converged.")
            break

    # After V(s) has converged, extract the optimal policy π*(s)
    # For all states s, select the optimal action a and store it in policy[s].
    # Output a deterministic policy, π(s) = argmax_a ∑_{s′,r} p(s′,r | s,a)[r + γ * V(s′)]
    policy = np.zeros(nS)
    for s in range(nS):
        if s in terminal_states:
             policy[s] = 0
             continue

        q_values = np.zeros(nA)
        for a in range(nA):
             q_value = 0
             for probability, next_state, reward, is_terminal in P[s][a]:
                  q_value += probability * (reward + gamma * V[next_state])
             q_values[a] = q_value

        # Determine the optimal action for a state as the action with the highest expected reward
        policy[s] = np.argmax(q_values)

    print("Optimal policy extracted.")
    return V, policy

def part_two():
    print("Running Part 2: Optimal Policy by Value Iteration")

    env = generate_env('FrozenLake-v1', None, "8x8", True)
    unwrapped_env = env.unwrapped 
    nS = unwrapped_env.observation_space.n

    optimal_V, optimal_policy = value_iteration(unwrapped_env, gamma=1.0, theta=1e-8)

    print("\n--- The Policy in a 2D array ---")
    policy_display = display_policy(optimal_policy, nS)
    print(policy_display)

    print("\n--- The converged V(s) table in a 2D array ---")  
    display_formatted_policy(optimal_V, nS)

    # Assuming the same evaluation setup as Part 1 (100 experiments, 10,000 episodes each)
    goals_list, mean_total_goal_steps_list = evaluate_policy(
        env, optimal_policy, num_experiments_per_policy, num_episodes_per_experiment
    )

    # Save the results to a CSV file
    csv_filename = "part2_optimal_policy.csv"
    csv_data_rows = [[goals_list[i], mean_total_goal_steps_list[i]] for i in range(num_experiments_per_policy)]
    csv_header = ['Goals', 'MeanGoalSteps']
    generate_csv(csv_data_rows, csv_filename, csv_header)

    env.close()
    print("\nPart 2 execution finished.")

# ================================================================================
# Main
# ================================================================================

def main():
    # part_one()
    part_two()

if __name__ == "__main__":
    main()