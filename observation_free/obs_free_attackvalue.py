import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os


sys.path.append(os.path.abspath(".."))
from attack import UCBRecommender, get_real_reward

def observation_free_attack(n_arms, target_arm, rho, T, means, std_devs, C1, C2, attack_value=1.0):
    recommender = UCBRecommender(n_arms, rho)
    pulls = np.zeros(n_arms)
    rewards = np.zeros(n_arms)
    target_arm_pulls = 0

    for t in range(1, T + 1):
        real_arm = recommender.play()

        if t <= C1:
            corrupted_rewards = np.zeros(n_arms)
        elif t <= C1 + C2:
            corrupted_rewards = np.zeros(n_arms)
            corrupted_rewards[target_arm] = attack_value  # <-- variable attack value
        else:
            corrupted_rewards = np.array([
                get_real_reward(means, std_devs, i) for i in range(n_arms)
            ])

        reward = corrupted_rewards[real_arm]
        recommender.update(real_arm, reward)

        pulls[real_arm] += 1
        rewards[real_arm] += reward
        if real_arm == target_arm:
            target_arm_pulls += 1

    return pulls, rewards, target_arm_pulls

def experiment_attack_value_sweep():
    T = 50000
    n_arms = 2
    rho = 1.0
    sigma = 1
    means = [0.9, 0.8]  # target is arm 1
    std_devs = np.full(n_arms, sigma)
    target_arm = 1
    C1 = 34
    C2 = 66

    # attack_values = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
    attack_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
    results = []

    for val in attack_values:
        _, _, target_count = observation_free_attack(
            n_arms, target_arm, rho, T, means, std_devs, C1, C2, attack_value=val
        )
        ratio = target_count / T * 100
        results.append((val, ratio))
        print(f"Attack value {val:.2f} → Target arm pull ratio: {ratio:.2f}%")

    vals, ratios = zip(*results)
    plt.plot(vals, ratios, marker='o')
    plt.xlabel("Attack Reward Value for Target Arm")
    plt.ylabel("Target Arm Pull Ratio (%)")
    plt.title("Effect of Attack Reward Value on Target Arm Success")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    experiment_attack_value_sweep()
