from typing import List
from matplotlib import pyplot as plt
import importlib
import numpy as np

K = 10
T = 100000
delta0 = 1.0

attacker_single = importlib.import_module("single_injection_ts").attacker(K=K, T=T, delta=0.05, delta0=1.0, sigma=0.5)
attacker_sequential = importlib.import_module("sequential_injection_ts").attacker(K=K, T=T, delta=0.05, delta0=1.0, sigma=0.5)
attacker_periodic = importlib.import_module("periodic_injection_ts").attacker(K=K, T=T, delta=0.05, delta0=1.0, sigma=0.5)

from real_reward import get_reward

class Thompson_single:
    def __init__(self, K: int, T: int, sigma: float = 0.05):
        self.K = K
        self.T = T
        self.n = [0] * K
        self.empirical_means = [0.0] * K
        self.sigma = sigma

    def update(self, k: int, reward: float):
        self.n[k] += 1
        self.empirical_means[k] = ((self.n[k] - 1) * self.empirical_means[k] + reward) / self.n[k]

    def run(self) -> float:
        ratio = [0] * self.T
        for t in range(1, self.K+1):
            k = t - 1
            reward = get_reward(k, self.sigma)
            self.update(k, reward)
            ratio[t-1] = self.n[self.K-1] / (t + 1)
        for t in range(self.K+1, self.T+1):
            k, r = attacker_single.feedback()
            self.update(k, r)
            ratio[t-1] = self.n[self.K-1] / (t + 1)
        return attacker_single.attack_cost

class Thompson_sequential:
    def __init__(self, K: int, T: int, sigma: float = 0.05):
        self.K = K
        self.T = T
        self.n = [0] * K
        self.empirical_means = [0.0] * K
        self.sigma = sigma

    def update(self, k: int, reward: float):
        self.n[k] += 1
        self.empirical_means[k] = ((self.n[k] - 1) * self.empirical_means[k] + reward) / self.n[k]

    def run(self) -> float:
        ratio = [0] * self.T
        for t in range(1, self.K+1):
            k = t - 1
            reward = get_reward(k, self.sigma)
            self.update(k, reward)
            ratio[t-1] = self.n[self.K-1] / (t + 1)
        for t in range(self.K+1, self.T+1):
            k, r = attacker_sequential.feedback()
            self.update(k, r)
            ratio[t-1] = self.n[self.K-1] / (t + 1)
        return attacker_sequential.attack_cost

class Thompson_periodic:
    def __init__(self, K: int, T: int, sigma: float = 0.05):
        self.K = K
        self.T = T
        self.n = [0] * K
        self.empirical_means = [0.0] * K
        self.sigma = sigma

    def update(self, k: int, reward: float):
        self.n[k] += 1
        self.empirical_means[k] = ((self.n[k] - 1) * self.empirical_means[k] + reward) / self.n[k]

    def run(self) -> float:
        ratio = [0] * self.T
        for t in range(1, self.K+1):
            k = t - 1
            reward = get_reward(k, self.sigma)
            self.update(k, reward)
            ratio[t-1] = self.n[self.K-1] / (t + 1)
        for t in range(self.K+1, self.T+1):
            k, r = attacker_periodic.feedback()
            self.update(k, r)
            ratio[t-1] = self.n[self.K-1] / (t + 1)
        return attacker_periodic.attack_cost


if __name__ == "__main__":
    x_axis = [i*1000 + 10000 for i in range(10)]
    len_x = len(x_axis)
    # plt.plot(range(1, T+1), ratio_periodic, label='Periodic Injection')
    # plt.xlabel("Time")
    # plt.ylabel("Ratio")
    # plt.title("Thompson Sampling")
    # plt.legend()
    # plt.savefig("thompson-ratio.png")


    # x_axis = [i*0.02 + 0.1 for i in range(21)]
    # len_x = len(x_axis)
    
    # cost_single = []
    # cost_sequential = []
    # cost_periodic = []
    
    # n_trials = 10
    # for _ in range(n_trials):
    #     for i in range(len_x):
    #         delta0 = x_axis[i]
    #         attacker_single = importlib.import_module("single_injection_ts").attacker(K=K, T=T, delta=0.05, delta0=delta0, sigma=0.5)
    #         attacker_sequential = importlib.import_module("sequential_injection_ts").attacker(K=K, T=T, delta=0.05, delta0=delta0, sigma=0.5)
    #         attacker_periodic = importlib.import_module("periodic_injection_ts").attacker(K=K, T=T, delta=0.05, delta0=delta0, sigma=0.5)

    #         thompson_single = Thompson_single(K, T)
    #         ratio_single = thompson_single.run()
    #         thompson_sequential = Thompson_sequential(K, T)
    #         ratio_sequential = thompson_sequential.run()
    #         thompson_periodic = Thompson_periodic(K, T)
    #         ratio_periodic = thompson_periodic.run()
            
    #         cost_single.append(attacker_single.attack_cost)
    #         cost_sequential.append(attacker_sequential.attack_cost)
    #         cost_periodic.append(attacker_periodic.attack_cost)
    
    # costs_single = np.array(cost_single).reshape(n_trials, len_x)
    # costs_sequential = np.array(cost_sequential).reshape(n_trials, len_x)
    # costs_periodic = np.array(cost_periodic).reshape(n_trials, len_x)
    
    # mean_costs_single = np.mean(costs_single, axis=0)
    # std_costs_single = np.std(costs_single, axis=0)
    # mean_costs_sequential = np.mean(costs_sequential, axis=0)
    # std_costs_sequential = np.std(costs_sequential, axis=0)
    # mean_costs_periodic = np.mean(costs_periodic, axis=0)
    # std_costs_periodic = np.std(costs_periodic, axis=0)
    
    # plt.figure(figsize=(10, 6))
    
    # plt.errorbar(x_axis, mean_costs_single, yerr=std_costs_single,
    #              marker='^', color='blue', label='Single Injection', capsize=5)
    x_axis = [10000, 20000, 50000, 100000, 200000, 500000, 1000000]
    len_x = len(x_axis)

    cost_single_trials = [[] for _ in range(len_x)]
    cost_sequential_trials = [[] for _ in range(len_x)]
    cost_periodic_trials = [[] for _ in range(len_x)]

    n = 10

    for _ in range(n):
        for i in range(len_x):
            T = x_axis[i]
            attacker_single = importlib.import_module("single_injection_ts").attacker(K=K, T=T, delta=0.05, delta0=1.0, sigma=0.5)
            attacker_sequential = importlib.import_module("sequential_injection_ts").attacker(K=K, T=T, delta=0.05, delta0=1.0, sigma=0.5)
            attacker_periodic = importlib.import_module("periodic_injection_ts").attacker(K=K, T=T, delta=0.05, delta0=1.0, sigma=0.5)

            thompson_single = Thompson_single(K, T)
            thompson_single.run()
            thompson_sequential = Thompson_sequential(K, T)
            thompson_sequential.run()
            thompson_periodic = Thompson_periodic(K, T)
            thompson_periodic.run()

            cost_single_trials[i].append(attacker_single.attack_cost)
            cost_sequential_trials[i].append(attacker_sequential.attack_cost)
            cost_periodic_trials[i].append(attacker_periodic.attack_cost)

    mean_cost_single = [np.mean(costs) for costs in cost_single_trials]
    std_cost_single = [np.std(costs) for costs in cost_single_trials]
    mean_cost_sequential = [np.mean(costs) for costs in cost_sequential_trials]
    std_cost_sequential = [np.std(costs) for costs in cost_sequential_trials]
    mean_cost_periodic = [np.mean(costs) for costs in cost_periodic_trials]
    std_cost_periodic = [np.std(costs) for costs in cost_periodic_trials]

    plt.figure(figsize=(12, 8))

    plt.errorbar(x_axis, mean_cost_single, yerr=std_cost_single,
                 marker='o', label='Single Injection', linestyle='dotted', color='blue', capsize=5)
    plt.errorbar(x_axis, mean_cost_sequential, yerr=std_cost_sequential,
                 marker='x', label='Sequential Injection', linestyle='--', color='green', capsize=5)
    plt.errorbar(x_axis, mean_cost_periodic, yerr=std_cost_periodic,
                 marker='s', label='Periodic Injection', linestyle='-', color='red', capsize=5)

    plt.tick_params(labelsize=27)
    plt.xlabel("T", fontsize=30)
    plt.ylabel("Average Total Attack Cost", fontsize=30)
    plt.legend(fontsize=28)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('thompson_attack_costs.png')
    plt.show()