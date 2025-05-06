import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os


sys.path.append(os.path.abspath(".."))
from attack import UCBRecommender, get_real_reward 

def simulate_frequency_limited_attack(T, n_arms, target_arm, R, means, std_devs, rho, delta0):
    recommender = UCBRecommender(n_arms, rho)
    target_pulls = 0
    attack_flag = True
    sigma = std_devs[0]  

    for t in range(1, T + 1):
        if t % (R + 1) == 1:
            attack_flag = True

        arm = recommender.play()
        reward = get_real_reward(means, std_devs, arm)

        if attack_flag and arm != target_arm and t > n_arms:
            nk = recommender.pulls[target_arm]
            beta = math.sqrt((2 * sigma**2 / (nk + 1)) * math.log((math.pi**2 * (nk + 1)**2 * n_arms) / (math.e * delta0)))
            desired_avg = means[target_arm] - 2 * beta - 3 * sigma * delta0
            fake_reward = desired_avg * (recommender.pulls[arm] + 1) - recommender.avg_rewards[arm] * recommender.pulls[arm]
            recommender.update(arm, fake_reward, fake=True)
            attack_flag = False
        else:
            recommender.update(arm, reward)

        if arm == target_arm:
            target_pulls += 1

    return recommender.pulls, recommender.rewards, target_pulls


def experiment_attack_frequency(T=50000, means=[0.9, 0.8], rho=1.0, sigma=1.0, delta0=0.05):
    means = np.array(means)
    n_arms = len(means)
    std_devs = np.full(n_arms, sigma)
    target_arm = np.argmin(means)

    R_values = [1, 5, 10, 25, 50, 100]
    results = []

    for R in R_values:
        pulls, rewards, target_count = simulate_frequency_limited_attack(
            T, n_arms, target_arm, R, means, std_devs, rho, delta0
        )
        ratio = target_count / T * 100
        results.append((R, ratio))
        print(f"R = {R} → Target arm pull ratio: {ratio:.2f}%")

    # Plot
    Rs, ratios = zip(*results)
    plt.figure(figsize=(10, 6))
    plt.plot(Rs, ratios, marker='o', label='Target Arm Pull %')
    plt.title("Effect of Attack Frequency on Target Arm Pull Ratio")
    plt.xlabel("Attack Interval R (attacks every R+1 rounds)")
    plt.ylabel("Target Arm Pull Ratio (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    experiment_attack_frequency(
        T=1000000,
        means=np.random.rand(10),
        rho=1.0,
        sigma=1.0,
        delta0=0.05
    )

