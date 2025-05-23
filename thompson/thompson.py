from typing import List
from matplotlib import pyplot as plt
import importlib

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
    # thompson_single = Thompson_single(K, T)
    # ratio_single = thompson_single.run()
    # thompson_sequential = Thompson_sequential(K, T)
    # ratio_sequential = thompson_sequential.run()
    # thompson_periodic = Thompson_periodic(K, T)
    # ratio_periodic = thompson_periodic.run()
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(1, T+1), ratio_single, label='Single Injection')
    # plt.plot(range(1, T+1), ratio_sequential, label='Sequential Injection')
    # plt.plot(range(1, T+1), ratio_periodic, label='Periodic Injection')
    # plt.xlabel("Time")
    # plt.ylabel("Ratio")
    # plt.title("Thompson Sampling")
    # plt.legend()
    # plt.savefig("thompson-ratio.png")
    # ratio_sequential = thompson_sequential.run()
    # thompson_periodic = Thompson_periodic(K, T)
    # ratio_periodic = thompson_periodic.run()
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(1, T+1), ratio_single, label='Single Injection')
    # plt.plot(range(1, T+1), ratio_sequential, label='Sequential Injection')
    # plt.plot(range(1, T+1), ratio_periodic, label='Periodic Injection')
    # plt.xlabel("Time")
    # plt.ylabel("Ratio")
    # plt.title("Thompson Sampling")
    # plt.legend()
    # plt.savefig("thompson-ratio.png")


    x_axis = [i*0.02 + 0.1 for i in range(21)]
    len_x = len(x_axis)
    
    cost_single = []
    cost_sequential = []
    cost_periodic = []
    
    n_trials = 10
    for _ in range(n_trials):
        for i in range(len_x):
            delta0 = x_axis[i]
            attacker_single = importlib.import_module("single_injection_ts").attacker(K=K, T=T, delta=0.05, delta0=delta0, sigma=0.5)
            attacker_sequential = importlib.import_module("sequential_injection_ts").attacker(K=K, T=T, delta=0.05, delta0=delta0, sigma=0.5)
            attacker_periodic = importlib.import_module("periodic_injection_ts").attacker(K=K, T=T, delta=0.05, delta0=delta0, sigma=0.5)

            thompson_single = Thompson_single(K, T)
            ratio_single = thompson_single.run()
            thompson_sequential = Thompson_sequential(K, T)
            ratio_sequential = thompson_sequential.run()
            thompson_periodic = Thompson_periodic(K, T)
            ratio_periodic = thompson_periodic.run()
            
            cost_single.append(attacker_single.attack_cost)
            cost_sequential.append(attacker_sequential.attack_cost)
            cost_periodic.append(attacker_periodic.attack_cost)
    
    costs_single = np.array(cost_single).reshape(n_trials, len_x)
    costs_sequential = np.array(cost_sequential).reshape(n_trials, len_x)
    costs_periodic = np.array(cost_periodic).reshape(n_trials, len_x)
    
    mean_costs_single = np.mean(costs_single, axis=0)
    std_costs_single = np.std(costs_single, axis=0)
    mean_costs_sequential = np.mean(costs_sequential, axis=0)
    std_costs_sequential = np.std(costs_sequential, axis=0)
    mean_costs_periodic = np.mean(costs_periodic, axis=0)
    std_costs_periodic = np.std(costs_periodic, axis=0)
    
    plt.figure(figsize=(10, 6))
    
    plt.errorbar(x_axis, mean_costs_single, yerr=std_costs_single,
                 marker='^', color='blue', label='Single Injection', capsize=5)
    plt.errorbar(x_axis, mean_costs_sequential, yerr=std_costs_sequential,
                 marker='o', color='green', label='Sequential Injection', capsize=5)
    plt.errorbar(x_axis, mean_costs_periodic, yerr=std_costs_periodic,
                 marker='s', color='red', label='Periodic Injection', capsize=5)
    
    plt.plot(x_axis, mean_costs_single, color='blue', linestyle='-')
    plt.plot(x_axis, mean_costs_sequential, color='green', linestyle='--')
    plt.plot(x_axis, mean_costs_periodic, color='red', linestyle=':')
    
    plt.xlabel("Delta0", fontsize=14)
    plt.ylabel("Attack Cost", fontsize=14)
    plt.title("Thompson Sampling Attack Cost vs Delta0", fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("thompson-cost-delta0.png")

    # x_axis = [10000, 20000, 50000, 100000, 200000, 500000, 1000000]
    # len_x = len(x_axis)
    # cost_single = [0] * len_x
    # cost_sequential = [0] * len_x
    # cost_periodic = [0] * len_x

    # n = 10

    # for _ in range(n):
    #     for i in range(len_x):
    #         T = x_axis[i]
    #         attacker_single = importlib.import_module("single_injection_ts").attacker(K=K, T=T, delta=0.05, delta0=1.0, sigma=0.5)
    #         attacker_sequential = importlib.import_module("sequential_injection_ts").attacker(K=K, T=T, delta=0.05, delta0=1.0, sigma=0.5)
    #         attacker_periodic = importlib.import_module("periodic_injection_ts").attacker(K=K, T=T, delta=0.05, delta0=1.0, sigma=0.5)

    #         thompson_single = Thompson_single(K, T)
    #         ratio_single = thompson_single.run()
    #         thompson_sequential = Thompson_sequential(K, T)
    #         ratio_sequential = thompson_sequential.run()
    #         thompson_periodic = Thompson_periodic(K, T)
    #         ratio_periodic = thompson_periodic.run()
    #         cost_single[i] += attacker_single.attack_cost
    #         cost_sequential[i] += attacker_sequential.attack_cost
    #         cost_periodic[i] += attacker_periodic.attack_cost
    # cost_single = [x / n for x in cost_single]
    # cost_sequential = [x / n for x in cost_sequential]
    # cost_periodic = [x / n for x in cost_periodic]
    # plt.figure(figsize=(10, 6))
    # plt.plot(x_axis, cost_single, label='Single Injection')
    # plt.plot(x_axis, cost_sequential, label='Sequential Injection')
    # plt.plot(x_axis, cost_periodic, label='Periodic Injection')
    # plt.scatter(x_axis, cost_single, marker='^', color='blue')
    # plt.scatter(x_axis, cost_sequential, marker='o', color='orange')
    # plt.scatter(x_axis, cost_periodic, marker='s', color='green')
    # plt.xlabel("T")
    # plt.ylabel("Cost")
    # plt.title("Thompson Sampling")
    # plt.legend()
    # plt.savefig("thompson-cost-T.png")