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

def periodic_injection_attack_real(n_arms, target_arm, rho, T, R, reward_matrix, a_tilde, sigma = 1, delta0 = 0.05, bounded=False):
    recommender = UCBRecommender(n_arms, rho)
    target_pulls = 0
    attack_trials = []
    estimated_rewards = np.zeros(n_arms)
    arm_pulls = np.zeros(n_arms)
    target_pull_ratios = []
    attack_cost = 0.0
    attack_list = []

    for t in range(1, T + 1 + n_arms):
        arm = recommender.play()
        reward = get_reward_from_matrix(reward_matrix, arm)
        arm_pulls[arm] += 1
        estimated_rewards[arm] = (estimated_rewards[arm] * (arm_pulls[arm] - 1) + reward) / arm_pulls[arm]

        if t > n_arms:
            if (t - n_arms) % R == 0:
                attack_list = []
            elif arm != target_arm and arm_pulls[arm] >= math.log(T / (delta0**2)) and arm not in attack_list:
                attack_list.append(arm)

            if arm != target_arm and arm_pulls[arm] >= threshold:
                nk = arm_pulls[target_arm]
                beta_val = beta(nk, sigma, n_arms, delta0)
                mu_target = estimated_rewards[target_arm]
                mu_arm = estimated_rewards[arm]
                rf = mu_arm - (mu_arm - mu_target + 2 * beta_val + 3 * sigma * delta0) * math.log(T / (n_tilde * delta0**2))

                if bounded:
                    rf = np.clip(rf, 0.0, 1.0)
                attack_cost += abs(rf)
                recommender.update(arm, rf)
                attack_trials.append(t)

            if arm == target_arm:
                target_pulls += 1

            target_pull_ratio = target_pulls / t
            target_pull_ratios.append(target_pull_ratio)

        recommender.update(arm, reward)

    return target_pulls, attack_trials, target_pull_ratios, attack_cost