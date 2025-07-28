import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os

sys.path.append(os.path.abspath(".."))
from attack import UCBRecommender, get_real_reward

def beta(N, sigma, n_arms, delta):
    N = max(N, 1)
    log_argument = max((math.pi**2 * N**2 * n_arms) / (3 * delta), 1)
    return math.sqrt((2 * sigma**2 / N) * math.log(log_argument))

def get_reward_from_matrix(reward_matrix, arm):
    return np.mean(reward_matrix[:, arm])

def least_injection_attack_real(n_arms, target_arm, rho, T, reward_matrix, sigma=1, delta0=0.05, bounded=False):
    recommender = UCBRecommender(n_arms, rho)
    target_pulls = 0
    estimated_rewards = np.zeros(n_arms)
    arm_pulls = np.zeros(n_arms)
    target_pull_ratios = []
    attacked_list = []
    attack_cost = 0.0

    for t in range(1, T + n_arms + 1):
        arm = recommender.play()
        reward = get_reward_from_matrix(reward_matrix, arm)
        arm_pulls[arm] += 1
        estimated_rewards[arm] = (estimated_rewards[arm] * (arm_pulls[arm] - 1) + reward) / arm_pulls[arm]

        if t > n_arms:
            if arm not in attacked_list and arm != target_arm and arm_pulls[arm] >= math.log(T / (delta0**2)):
                attacked_list.append(arm)

                N_target = arm_pulls[target_arm]
                beta_val = beta(N_target, sigma, n_arms, delta0)
                mu_target = estimated_rewards[target_arm]
                mu_arm = estimated_rewards[arm]
                pulls_arm = recommender.pulls[arm]

                desired_avg = mu_target - 2 * beta_val - 3 * sigma
                fake_reward = desired_avg * (pulls_arm + 1) - mu_arm * pulls_arm

                if bounded:
                    fake_reward = np.clip(fake_reward, 0.0, 1.0)

                attack_cost += abs(fake_reward - reward)
                recommender.update(arm, fake_reward)

            if arm == target_arm:
                target_pulls += 1
            target_pull_ratio = target_pulls / t
            target_pull_ratios.append(target_pull_ratio)

        recommender.update(arm, reward)

    return target_pulls, target_pull_ratios, attack_cost



def least_injection_UCB_attack(n_arms, target_arm, rho, T, means, std_devs, sigma=1, delta0=0.05, bounded=False):
    arm_counts = np.zeros((n_arms, T + n_arms))
    recommender = UCBRecommender(n_arms, rho)
    target_pull_counter = 0
    cumulative_non_target_pull_counter = 0
    non_target_pull_list = []
    estimated_reward_distribution = np.zeros(n_arms)
    arm_pulls = np.zeros(n_arms)
    attack_trials_list = []
    attacked_list = []
    target_pull_ratios = []

    for round_num in range(1, T + n_arms):
        real_arm = recommender.play()
        arm_counts[real_arm, round_num] += 1
        if real_arm == target_arm:
            target_pull_counter += 1
        else:
            cumulative_non_target_pull_counter += 1
        non_target_pull_list.append(cumulative_non_target_pull_counter)

        real_reward = get_real_reward(means, std_devs, real_arm)
        arm_pulls[real_arm] += 1
        estimated_reward_distribution[real_arm] = (estimated_reward_distribution[real_arm] * (arm_pulls[real_arm] - 1) + real_reward) / arm_pulls[real_arm]

        if round_num > n_arms:
            if real_arm not in attacked_list and real_arm != target_arm and arm_pulls[real_arm] >= math.log(T / (delta0**2)):
                attacked_list.append(real_arm)
                N_target = arm_pulls[target_arm]
                attack_beta = beta(N_target, sigma, n_arms, delta0)
                mu_target = estimated_reward_distribution[target_arm]
                mu_arm = estimated_reward_distribution[real_arm]
                pulls_arm = recommender.pulls[real_arm]

                desired_avg = mu_target - 2 * attack_beta - 3 * sigma
                fake_reward = desired_avg * (pulls_arm + 1) - mu_arm * pulls_arm

                if bounded:
                    fake_reward = np.clip(fake_reward, 0.0, 1.0)

                recommender.update(real_arm, fake_reward)
                attack_trials_list.append(round_num)

        target_pull_ratio = target_pull_counter / round_num
        target_pull_ratios.append(target_pull_ratio)
        recommender.update(real_arm, real_reward)

    return arm_counts, attack_trials_list, target_pull_counter, non_target_pull_list, target_pull_ratios


def experiment_synthetic_least_injection(T=10000, n_arms=10, rho=1.0, sigma=1.0, bounded=False, trials=10):
    all_ratios = []

    for _ in range(trials):
        means = np.random.rand(n_arms)
        std_devs = np.full(n_arms, sigma)
        target_arm = np.argmin(means)

        _, _, _, _, target_pull_ratios = least_injection_UCB_attack(
            n_arms, target_arm, rho, T, means, std_devs, bounded=bounded)
        all_ratios.append(target_pull_ratios)

    # Calculate average ratios over trials
    avg_ratios = np.mean(all_ratios, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, T + n_arms), avg_ratios, label="Average Target Arm Selection Ratio")
    plt.xlabel("Rounds")
    plt.ylabel("Target Arm Selection Ratio")
    plt.title(f"Average Target Arm Selection Ratio Over Time (Single Injection Attack) - {'Bounded' if bounded else 'Unbounded'}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Completed {trials} trials.")

def experiment_real_least_injection(T=int(1e4), n_arms=10, rho=1.0, sigma=1.0, delta0=0.05, trials=10):
    all_ratios = []

    for _ in range(trials):
        reduced_matrix = np.load(os.path.join("..", "dataset", "movielens.npy"))
        selected_movie_indices = np.random.choice(reduced_matrix.shape[1], size=n_arms, replace=False)
        reduced_matrix = reduced_matrix[:, selected_movie_indices]
        movie_interactions = np.sum(reduced_matrix, axis=0)
        least_interacted_movie = np.argmin(movie_interactions)
        target_arm = least_interacted_movie

        target_pulls, target_pull_ratios, attack_cost = least_injection_attack_real(
            n_arms, target_arm, rho, T, reduced_matrix, sigma=sigma, delta0=delta0,
        )
        all_ratios.append(target_pull_ratios)

    avg_ratios = np.mean(all_ratios, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, T + 1), avg_ratios, label="Average Target Arm Selection Ratio")
    plt.xlabel("Rounds")
    plt.ylabel("Target Arm Selection Ratio")
    plt.title(f"Average Target Arm Selection Ratio Over Time (Least Injection Attack for Real Dataset) ")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_attack_cost_real(n_arms=10, rho=1.0, sigma=1.0, delta0=0.2, trials=10):
    avg_costs = []
    T_values = np.logspace(1, 4, num=10, dtype=int)
    for T in T_values:
        trial_costs = []
        for _ in range(trials):
            reduced_matrix = np.load(os.path.join("..", "dataset", "movielens.npy"))
            selected_movie_indices = np.random.choice(reduced_matrix.shape[1], size=n_arms, replace=False)
            reduced_matrix = reduced_matrix[:, selected_movie_indices]
            
            movie_interactions = np.sum(reduced_matrix, axis=0)
            target_arm = np.argmin(movie_interactions)

            target_pulls, target_pull_ratios, attack_cost = least_injection_attack_real(
                n_arms, target_arm, rho, T, reduced_matrix, sigma=sigma, delta0=delta0,
            )
            trial_costs.append(attack_cost)
        
        avg_costs.append(np.mean(trial_costs))

    plt.figure(figsize=(8, 5))
    plt.plot(T_values, avg_costs, marker='o')
    plt.xlabel("T (Rounds)")
    plt.ylabel("Average Total Attack Cost")
    plt.title(f"Total Attack Cost vs. T for Least Injection Attack (MovieLens Dataset)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    experiment_synthetic_least_injection(bounded=False, trials=10)
