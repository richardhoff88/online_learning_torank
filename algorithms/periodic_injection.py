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

def periodic_injection_attack_real(n_arms, target_arm, rho, T, reward_matrix, a_tilde, R, f, sigma = 1, delta0 = 0.05, bounded=False):
    recommender = UCBRecommender(n_arms, rho)
    target_pulls = 0
    estimated_rewards = np.zeros(n_arms)
    arm_pulls = np.zeros(n_arms)
    target_pull_ratios = []
    attack_cost = 0.0
    attack_list = []
    injection_list = []
    sleep_counter = 0

    for t in range(1, T + 1 + n_arms):
        arm = recommender.play()
        reward = get_reward_from_matrix(reward_matrix, arm)
        arm_pulls[arm] += 1
        estimated_rewards[arm] = (estimated_rewards[arm] * (arm_pulls[arm] - 1) + reward) / arm_pulls[arm]

        if t > n_arms:
            print(t)
            if sleep_counter <= 0:
                for arm in attack_list[:]:
                    mu_i = estimated_rewards[arm]
                    mu_k = estimated_rewards[target_arm]
                    l_hat = mu_k - 2 * beta(arm_pulls[target_arm], sigma, n_arms, delta0) - 3 * sigma * delta0

                    # Number of injection rounds n_tilde
                    a_tilde_new = min(a_tilde, mu_k - 3 * beta(arm_pulls[target_arm], sigma, n_arms, delta0) - 3 * sigma * delta0)
                    n_tilde = (mu_i - l_hat) * math.log(T) / (l_hat - a_tilde_new)/ delta0**2
                    for _ in range(int(n_tilde)):
                        injection_list.append((arm, a_tilde_new))
                    attack_list.remove(arm)
                
                injections_to_do = injection_list[:min(f, len(injection_list))]
                for injection in injections_to_do:
                    recommender.update(injection[0], injection[1])
                injection_list = injection_list[len(injections_to_do):]
                sleep_counter = R

            if arm != target_arm and arm_pulls[arm] >= math.log(T / (delta0**2)) and arm not in attack_list:
                attack_list.append(arm)

            if arm == target_arm:
                target_pulls += 1

            target_pull_ratio = target_pulls / t
            target_pull_ratios.append(target_pull_ratio)

            if sleep_counter > 0:
                sleep_counter -= 1

        recommender.update(arm, reward)

    return target_pulls, target_pull_ratios, attack_cost

def experiment_real_periodic_bounded_injection(T=int(1e4), n_arms=10, rho=1.0, a_tilde = 0.0, sigma=1.0, delta0=0.05, R = 10, f = 100, trials=10):
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

        target_pulls, target_pull_ratios, attack_cost = periodic_injection_attack_real(
            n_arms, target_arm, rho, T, reduced_matrix, a_tilde=a_tilde, f=f, R=R, sigma=sigma, delta0=delta0)
        all_ratios.append(target_pull_ratios)

    avg_ratios = np.mean(all_ratios, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, T + 1), avg_ratios, label="Average Target Arm Selection Ratio")
    plt.xlabel("Rounds")
    plt.ylabel("Target Arm Selection Ratio")
    plt.title(f"Average Target Arm Selection Ratio Over Time (Periodic Bounded Injection Attack for Real Dataset) with f = {f}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    experiment_real_periodic_bounded_injection()