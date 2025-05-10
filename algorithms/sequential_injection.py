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

def sequential_injection_attack(n_arms, target_arm, rho, T, means, std_devs, n_tilde, delta0=0.05, bounded = False):
    recommender = UCBRecommender(n_arms, rho)
    target_pulls = 0
    attack_trials = []
    sigma = std_devs[0]
    estimated_rewards = np.zeros(n_arms)
    arm_pulls = np.zeros(n_arms)
    target_pull_ratios = []

    for t in range(1, T + 1 + n_arms):
        arm = recommender.play()
        reward = get_real_reward(means, std_devs, arm)

        arm_pulls[arm] += 1
        estimated_rewards[arm] = (estimated_rewards[arm] * (arm_pulls[arm] - 1) + reward) / arm_pulls[arm]
        if t > n_arms:
            threshold = math.log(t / (delta0**2)) - n_tilde
            if arm != target_arm and arm_pulls[arm] >= threshold:
                nk = arm_pulls[target_arm]
                beta_val = beta(nk, sigma, n_arms, delta0)
                mu_target = estimated_rewards[target_arm]
                mu_arm = estimated_rewards[arm]
                rf = mu_arm - (mu_arm - mu_target + 2 * beta_val + 3 * sigma * delta0) * math.log(T / (n_tilde * delta0**2))

                if bounded:
                    rf = np.clip(rf, 0.0, 1.0)
                recommender.update(arm, rf)
                attack_trials.append(t)

            if arm == target_arm:
                target_pulls += 1

            target_pull_ratio = target_pulls / t
            target_pull_ratios.append(target_pull_ratio)

        recommender.update(arm, reward)

    return target_pulls, attack_trials, target_pull_ratios

def sequential_injection_attack_real(n_arms, target_arm, rho, T, reward_matrix, n_tilde, sigma = 1, delta0 = 0.05, bounded=False):
    recommender = UCBRecommender(n_arms, rho)
    target_pulls = 0
    attack_trials = []
    estimated_rewards = np.zeros(n_arms)
    arm_pulls = np.zeros(n_arms)
    target_pull_ratios = []

    for t in range(1, T + 1 + n_arms):
        arm = recommender.play()
        reward = get_reward_from_matrix(reward_matrix, arm)
        arm_pulls[arm] += 1
        estimated_rewards[arm] = (estimated_rewards[arm] * (arm_pulls[arm] - 1) + reward) / arm_pulls[arm]
        if t > n_arms:
            threshold = math.log(t / (delta0**2)) - n_tilde
            if arm != target_arm and arm_pulls[arm] >= threshold:
                nk = arm_pulls[target_arm]
                beta_val = beta(nk, sigma, n_arms, delta0)
                mu_target = estimated_rewards[target_arm]
                mu_arm = estimated_rewards[arm]
                rf = mu_arm - (mu_arm - mu_target + 2 * beta_val + 3 * sigma * delta0) * math.log(T / (n_tilde * delta0**2))

                if bounded:
                    rf = np.clip(rf, 0.0, 1.0)
                recommender.update(arm, rf)
                attack_trials.append(t)

            if arm == target_arm:
                target_pulls += 1

            target_pull_ratio = target_pulls / t
            target_pull_ratios.append(target_pull_ratio)

        recommender.update(arm, reward)

    return target_pulls, attack_trials, target_pull_ratios

def experiment_synthetic_sequential_injection(T=10000, n_arms=10, n_tilde=10, rho=1.0, sigma=1.0, delta0=0.05, bounded = False, trials=10):
    all_ratios = []

    for _ in range(trials):
        means = np.random.rand(n_arms)
        std_devs = np.full(n_arms, sigma)
        target_arm = np.argmin(means)

        target_pulls, attack_trials, target_pull_ratios = sequential_injection_attack(
            n_arms, target_arm, rho, T, means, std_devs, n_tilde=n_tilde, delta0=delta0, bounded=bounded
        )
        all_ratios.append(target_pull_ratios)

    avg_ratios = np.mean(all_ratios, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, T + 1), avg_ratios, label="Average Target Arm Selection Ratio")
    plt.xlabel("Rounds")
    plt.ylabel("Target Arm Selection Ratio")
    plt.title(f"Average Target Arm Selection Ratio Over Time (Sequential Injection Attack) - {'Bounded' if bounded else 'Unbounded'} ")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Completed {trials} trials.")

def experiment_real_sequential_injection(T=10000, n_arms=10, rho=1.0, n_tilde = 10, sigma=1.0, delta0=0.05, bounded = False, trials=10):
    all_ratios = []
    for _ in range(trials):
        reduced_matrix = np.load(os.path.join("..", "dataset", "movielens.npy"))
        selected_movie_indices = np.random.choice(reduced_matrix.shape[1], size=n_arms, replace=False)
        # Slice the matrix to keep only n_arms
        reduced_matrix = reduced_matrix[:, selected_movie_indices]
        # Use smallest reward arm as target
        movie_interactions = np.sum(reduced_matrix, axis=0)
        least_interacted_movie = np.argmin(movie_interactions)
        target_arm = least_interacted_movie

        target_pulls, attack_trials, target_pull_ratios = sequential_injection_attack_real(
            n_arms, target_arm, rho, T, reduced_matrix, n_tilde=n_tilde, sigma=sigma, delta0=delta0, bounded=bounded
        )
        all_ratios.append(target_pull_ratios)

    avg_ratios = np.mean(all_ratios, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, T + 1), avg_ratios, label="Average Target Arm Selection Ratio")
    plt.xlabel("Rounds")
    plt.ylabel("Target Arm Selection Ratio")
    plt.title(f"Average Target Arm Selection Ratio Over Time (Sequential Injection Attack for Real Dataset) - {'Bounded' if bounded else 'Unbounded'} ")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Completed {trials} trials.")

if __name__ == "__main__":
    experiment_real_sequential_injection(bounded = True, trials=10)
