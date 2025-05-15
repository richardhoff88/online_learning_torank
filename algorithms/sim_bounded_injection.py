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

def calculate_a_tilde(mu_k, n_arms, sigma, delta0):
    beta_one = math.sqrt((2 * sigma**2) * math.log((math.pi**2 * n_arms) / (3 * delta0)))
    a_tilde = mu_k - 3 * beta_one - 3 * sigma * delta0
    return max(a_tilde, 0.0) 

def get_reward_from_matrix(reward_matrix, arm):
    return np.mean(reward_matrix[:, arm])

def simultaneous_bounded_injection_attack(n_arms, target_arm, rho, T, means, std_devs, a_tilde, delta0=0.05):
    recommender = UCBRecommender(n_arms, rho)
    target_pulls = 0
    attack_trials = []
    sigma = std_devs[0]
    estimated_rewards = np.zeros(n_arms)
    arm_pulls = np.zeros(n_arms)
    target_pull_ratios = []
    attack_cost = 0.0

    for t in range(1, T + 1 + n_arms):
        arm = recommender.play()
        reward = get_real_reward(means, std_devs, arm)

        # Update empirical reward
        arm_pulls[arm] += 1
        estimated_rewards[arm] = (estimated_rewards[arm] * (arm_pulls[arm] - 1) + reward) / arm_pulls[arm]
        
        if t > n_arms:
            # SBI Attack Condition
            if arm != target_arm and arm_pulls[arm] >= math.log(T / (delta0**2)):
                mu_i = estimated_rewards[arm]
                mu_k = estimated_rewards[target_arm]
                l_hat = mu_k - 2 * beta(arm_pulls[target_arm], sigma, n_arms, delta0)

                # Number of injection rounds n_tilde
                a_tilde_new = min(a_tilde, mu_k - 3 * beta(arm_pulls[target_arm], sigma, n_arms, delta0) - 3 * sigma * delta0)
                n_tilde = (mu_i - l_hat) * math.log(T) / (l_hat - a_tilde_new) / delta0**2
                for _ in range(int(n_tilde)):
                    recommender.update(arm, a_tilde_new)
                    attack_cost += abs(a_tilde_new - mu_i)

                attack_trials.append(t)

            if arm == target_arm:
                target_pulls += 1

            target_pull_ratio = target_pulls / t
            target_pull_ratios.append(target_pull_ratio)

        recommender.update(arm, reward)

    return target_pulls, attack_trials, target_pull_ratios, attack_cost

def simultaneous_bounded_injection_attack_real(n_arms, target_arm, rho, T, reward_matrix, a_tilde=0, sigma = 1, delta0=0.05):
    recommender = UCBRecommender(n_arms, rho)
    target_pulls = 0
    attack_trials = []
    estimated_rewards = np.zeros(n_arms)
    arm_pulls = np.zeros(n_arms)
    target_pull_ratios = []
    attack_cost = 0.0

    for t in range(1, T + 1 + n_arms):
        arm = recommender.play()
        reward = get_reward_from_matrix(reward_matrix, arm)

        # Update empirical reward
        arm_pulls[arm] += 1
        estimated_rewards[arm] = (estimated_rewards[arm] * (arm_pulls[arm] - 1) + reward) / arm_pulls[arm]

        if t > n_arms:
            # SBI Attack Condition
            if arm != target_arm and arm_pulls[arm] >= math.log(T / (delta0**2)):
                mu_i = estimated_rewards[arm]
                mu_k = estimated_rewards[target_arm]
                l_hat = mu_k - 2 * beta(arm_pulls[target_arm], sigma, n_arms, delta0) - 3 * sigma * delta0

                # Number of injection rounds n_tilde
                a_tilde_new = min(a_tilde, mu_k - 3 * beta(arm_pulls[target_arm], sigma, n_arms, delta0) - 3 * sigma * delta0)
                n_tilde = (mu_i - l_hat) * math.log(T) / (l_hat - a_tilde_new) / delta0**2
                for _ in range(int(n_tilde)):
                    recommender.update(arm, a_tilde_new)
                    attack_cost += abs(a_tilde_new - mu_i)
                    
                attack_trials.append(t)

            if arm == target_arm:
                target_pulls += 1

            target_pull_ratio = target_pulls / t
            target_pull_ratios.append(target_pull_ratio)

        recommender.update(arm, reward)

    return target_pulls, attack_trials, target_pull_ratios, attack_cost

def experiment_simultaneous_bounded_injection(T=int(1e4), n_arms=10, rho=1.0, sigma=1.0, a_tilde=0.0, delta0=0.05, trials=10):
    all_ratios = []

    for _ in range(trials):
        means = np.random.rand(n_arms)
        std_devs = np.full(n_arms, sigma)
        target_arm = np.argmin(means)
        mu_k = means[target_arm]
        a_tilde = calculate_a_tilde(mu_k, n_arms, sigma, delta0)

        target_pulls, attack_trials, target_pull_ratios, attack_cost = simultaneous_bounded_injection_attack(
            n_arms, target_arm, rho, T, means, std_devs, a_tilde, delta0=delta0
        )
        all_ratios.append(target_pull_ratios)

    avg_ratios = np.mean(all_ratios, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, T + 1), avg_ratios, label="Average Target Arm Selection Ratio")
    plt.xlabel("Rounds")
    plt.ylabel("Target Arm Selection Ratio")
    plt.title(f"Average Target Arm Selection Ratio Over Time (SBI Attack)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Completed {trials} trials.")

def experiment_real_simultaneous_bounded_injection(T=int(1e4), n_arms=10, rho=1.0, a_tilde = 0.0, sigma=1.0, delta0=0.05, trials=10):
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

        target_pulls, attack_trials, target_pull_ratios, attack_cost = simultaneous_bounded_injection_attack_real(
            n_arms, target_arm, rho, T, reduced_matrix, a_tilde=a_tilde, sigma=sigma, delta0=delta0,
        )
        # print(attack_cost)
        all_ratios.append(target_pull_ratios)

    avg_ratios = np.mean(all_ratios, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, T + 1), avg_ratios, label="Average Target Arm Selection Ratio")
    plt.xlabel("Rounds")
    plt.ylabel("Target Arm Selection Ratio")
    plt.title(f"Average Target Arm Selection Ratio Over Time (Simultaneous Bounded Injection Attack for Real Dataset) ")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_attack_cost_real(n_arms=10, rho=1.0, a_tilde=0.0, sigma=1.0, delta0=0.2, trials=10):
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

            target_pulls, attack_trials, target_pull_ratios, attack_cost = simultaneous_bounded_injection_attack_real(
                n_arms, target_arm, rho, T, reduced_matrix, a_tilde=a_tilde, sigma=sigma, delta0=delta0,
            )
            trial_costs.append(attack_cost)
        
        avg_costs.append(np.mean(trial_costs))

    plt.figure(figsize=(8, 5))
    plt.plot(T_values, avg_costs, marker='o')
    plt.xlabel("T (Rounds)")
    plt.ylabel("Average Total Attack Cost")
    plt.title(f"Total Attack Cost vs. T for Simultaneous Injection Attack (MovieLens Dataset)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_attack_cost_vs_delta0_real(n_arms=10, rho=1.0, T=int(1e4), a_tilde=0.0, sigma=1.0, trials=10):
    avg_costs = []
    delta0_values = np.linspace(0.1, 0.5, num=20)  

    for delta0 in delta0_values:
        trial_costs = []
        for _ in range(trials):
            reduced_matrix = np.load(os.path.join("..", "dataset", "movielens.npy"))
            selected_movie_indices = np.random.choice(reduced_matrix.shape[1], size=n_arms, replace=False)
            reduced_matrix = reduced_matrix[:, selected_movie_indices]

            movie_interactions = np.sum(reduced_matrix, axis=0)
            target_arm = np.argmin(movie_interactions)

            target_pulls, attack_trials, target_pull_ratios, attack_cost = simultaneous_bounded_injection_attack_real(
            n_arms, target_arm, rho, T, reduced_matrix, a_tilde=a_tilde, sigma=sigma, delta0=delta0,
        )

            trial_costs.append(attack_cost)

        avg_costs.append(np.mean(trial_costs))

    plt.figure(figsize=(8, 5))
    plt.plot(delta0_values, avg_costs, marker='o')
    plt.xlabel("δ₀ (Confidence Parameter)")
    plt.ylabel("Average Total Attack Cost")
    plt.title(f"Total Attack Cost vs. δ₀ for Simultaneous Bounded Injection Attack (MovieLens Dataset) T = {T}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # experiment_real_simultaneous_bounded_injection(a_tilde=0.0)
    # plot_attack_cost_real()
    plot_attack_cost_vs_delta0_real()

