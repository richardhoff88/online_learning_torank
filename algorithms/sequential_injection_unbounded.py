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

def sequential_injection_attack(n_arms, target_arm, rho, T, means, std_devs, R, n_tilde, delta0=0.05):
    recommender = UCBRecommender(n_arms, rho)
    target_pulls = 0
    attack_trials = []
    sigma = std_devs[0]
    estimated_rewards = np.zeros(n_arms)
    arm_pulls = np.zeros(n_arms)
    target_pull_ratios = []

    for t in range(1, T + 1):
        arm = recommender.play()
        reward = get_real_reward(means, std_devs, arm)

        # Update empirical reward
        arm_pulls[arm] += 1
        estimated_rewards[arm] = (estimated_rewards[arm] * (arm_pulls[arm] - 1) + reward) / arm_pulls[arm]

        # Attack condition: (arm != target_arm and Na(t) >= log(t / delta0^2) - n_tilde)
        threshold = math.log(t / (delta0**2)) - n_tilde
        if arm != target_arm and arm_pulls[arm] >= threshold:
            nk = arm_pulls[target_arm]
            beta_val = beta(nk, sigma, n_arms, delta0)
            mu_target = estimated_rewards[target_arm]
            mu_arm = estimated_rewards[arm]

            # Calculate fake reward
            rf = mu_arm - (mu_arm - mu_target + 2 * beta_val + 3 * sigma * delta0) * math.log(T / (n_tilde * delta0**2))

            recommender.update(arm, rf)
            attack_trials.append(t)

        if arm == target_arm:
            target_pulls += 1

        # Calculate the ratio of target arm pulls
        target_pull_ratio = target_pulls / t
        target_pull_ratios.append(target_pull_ratio)

        recommender.update(arm, reward)

    return target_pulls, attack_trials, target_pull_ratios


def experiment_synthetic_sequential_injection(T=100, n_arms=10, rho=1.0, sigma=1.0, delta0=0.05):
    means = np.random.rand(n_arms)
    std_devs = np.full(n_arms, sigma)
    target_arm = np.argmin(means)
    print(means)
    print(target_arm)
    n_tilde = 10  # number of injection points

    target_pulls, attack_trials, target_pull_ratios = sequential_injection_attack(
        n_arms, target_arm, rho, T, means, std_devs, R=10, n_tilde=n_tilde, delta0=delta0
    )

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, T + 1), target_pull_ratios, label="Target Arm Selection Ratio")
    plt.xlabel("Rounds")
    plt.ylabel("Target Arm Selection Ratio")
    plt.title("Target Arm Selection Ratio Over Time (Sequential Injection Attack)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Total target arm pulls: {target_pulls}")
    print(f"Attack trials: {attack_trials}")


if __name__ == "__main__":
    experiment_synthetic_sequential_injection()
