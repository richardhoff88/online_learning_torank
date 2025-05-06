import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import sys
import os


sys.path.append(os.path.abspath(".."))
from attack import UCBRecommender, get_real_reward


def observation_free_attack(n_arms, target_arm, rho, T, means, std_devs, C1, C2):
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
            corrupted_rewards[target_arm] = 1
        
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

# our params
# T = 1000
# n_arms = 10
# rho = 1.0
# sigma = 1
# means = np.random.rand(n_arms)
# std_devs = np.full(n_arms, sigma)
# target_arm = np.argmin(means)  # worst arm
# C1 = 300
# C2 = 500

# paper params
T = 50000
n_arms = 2
rho = 1.0
sigma = 1
means = [0.9, 0.8]
std_devs = np.full(n_arms, sigma)
target_arm = 1  

# for UCB specifically
C1 = 34
C2 = 66




def plot_target_dominance():
    pulls_over_time = []
    recommender = UCBRecommender(n_arms, rho)
    target_count = 0
    history = []

    # plot first 100 rounds 
    for t in range(1, 101):
        arm = recommender.play()
        if t <= C1:
            reward_vec = np.zeros(n_arms)
        elif t <= C1 + C2:
            reward_vec = np.zeros(n_arms)
            reward_vec[target_arm] = 1
        else:
            reward_vec = np.array([
                get_real_reward(means, std_devs, i) for i in range(n_arms)
            ])
        recommender.update(arm, reward_vec[arm])
        if arm == target_arm:
            target_count += 1
        pulls_over_time.append(target_count / t)

    plt.plot(pulls_over_time)
    plt.suptitle("Fraction of Target Arm Chosen Over Time")
    plt.title(ratio)
    plt.xlabel("Rounds")
    plt.ylabel("Target Arm Selection Ratio")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    pulls, rewards, target_count = observation_free_attack(
    n_arms, target_arm, rho, T, means, std_devs, C1, C2
    )

    ratio = (f"Target arm pull ratio: {target_count / T * 100:.3f}%")
    plot_target_dominance()