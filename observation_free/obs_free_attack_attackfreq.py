import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os

sys.path.append(os.path.abspath(".."))
from attack import UCBRecommender, get_real_reward

def simulate_observation_attack_frequency(T, n_arms, target_arm, R, means, std_devs, rho, delta0, C1_ratio=0.3, attack_value=1.0):
    recommender = UCBRecommender(n_arms, rho)
    target_pulls = 0
    sigma = std_devs[0]

    corruption_bound = int((n_arms * np.log(T)) / (means[target_arm] ** 2))
    C1 = int(C1_ratio * corruption_bound)
    C2 = corruption_bound - C1

    for t in range(1, T + 1):
        attack_flag = (t % (R + 1) == 1)

        if t <= C1:
            corrupted_rewards = np.zeros(n_arms)
        elif t <= C1 + C2 and attack_flag:
            corrupted_rewards = np.zeros(n_arms)
            # Bound the attack value to be within [0, 1]
            bounded_value = np.clip(attack_value, 0.0, 1.0)
            corrupted_rewards[target_arm] = bounded_value
        else:
            corrupted_rewards = np.array([
                get_real_reward(means, std_devs, i) for i in range(n_arms)
            ])

        arm = recommender.play()
        reward = corrupted_rewards[arm]
        recommender.update(arm, reward)

        if arm == target_arm:
            target_pulls += 1

    return recommender.pulls, recommender.rewards, target_pulls

def experiment_observation_attack_frequency(T=50000, means=[0.9, 0.8], rho=1.0, sigma=1.0, delta0=0.05):
    means = np.array(means)
    n_arms = len(means)
    std_devs = np.full(n_arms, sigma)
    target_arm = np.argmin(means)

    R_values = [1, 5, 10, 25, 50, 100]
    results = []

    for R in R_values:
        pulls, rewards, target_count = simulate_observation_attack_frequency(
            T, n_arms, target_arm, R, means, std_devs, rho, delta0
        )
        ratio = target_count / T * 100
        results.append((R, ratio))
        print(f"R = {R} → Target arm pull ratio: {ratio:.2f}%")

    Rs, ratios = zip(*results)
    plt.figure(figsize=(10, 6))
    plt.plot(Rs, ratios, marker='o', label='Target Arm Pull %')
    plt.title("Effect of Attack Frequency (Observation-Free) on Target Arm Pull Ratio")
    plt.xlabel("Attack Interval R (attacks every R+1 rounds)")
    plt.ylabel("Target Arm Pull Ratio (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    experiment_observation_attack_frequency(
        T=1000000,
        means=np.random.rand(10),
        rho=1.0,
        sigma=1.0,
        delta0=0.05
    )