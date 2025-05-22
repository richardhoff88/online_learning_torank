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

    def run(self) -> List[float]:
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
        return ratio

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

    def run(self) -> List[float]:
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
        return ratio
    

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

    def run(self) -> List[float]:
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
        return ratio


if __name__ == "__main__":
    thompson_single = Thompson_single(K, T)
    ratio_single = thompson_single.run()
    thompson_sequential = Thompson_sequential(K, T)
    ratio_sequential = thompson_sequential.run()
    thompson_periodic = Thompson_periodic(K, T)
    ratio_periodic = thompson_periodic.run()
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, T+1), ratio_single, label='Single Injection')
    plt.plot(range(1, T+1), ratio_sequential, label='Sequential Injection')
    plt.plot(range(1, T+1), ratio_periodic, label='Periodic Injection')
    plt.xlabel("Time")
    plt.ylabel("Ratio")
    plt.title("Thompson Sampling")
    plt.legend()
    plt.savefig("thompson-ratio.png")


    x_axis = [i*0.02 + 0.1 for i in range(21)]

    cost_single = []
    cost_sequential = []
    cost_periodic = []

    for delta0 in x_axis:
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

    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, cost_single, label='Single Injection')
    plt.plot(x_axis, cost_sequential, label='Sequential Injection')
    plt.plot(x_axis, cost_periodic, label='Periodic Injection')
    # 给每个点加上三角形、圆形、正方形的标记
    plt.scatter(x_axis, cost_single, marker='^', color='blue')
    plt.scatter(x_axis, cost_sequential, marker='o', color='orange')
    plt.scatter(x_axis, cost_periodic, marker='s', color='green')
    plt.xlabel("Delta0")
    plt.ylabel("Cost")
    plt.title("Thompson Sampling")
    plt.legend()
    plt.savefig("thompson-cost.png")


    cost_single = []
    cost_sequential = []
    cost_periodic = []
    for T in range(100000, 1000001, 100000):
        attacker_single = importlib.import_module("single_injection_ts").attacker(K=K, T=T, delta=0.05, delta0=1.0, sigma=0.5)
        attacker_sequential = importlib.import_module("sequential_injection_ts").attacker(K=K, T=T, delta=0.05, delta0=1.0, sigma=0.5)
        attacker_periodic = importlib.import_module("periodic_injection_ts").attacker(K=K, T=T, delta=0.05, delta0=1.0, sigma=0.5)

        thompson_single = Thompson_single(K, T)
        ratio_single = thompson_single.run()
        thompson_sequential = Thompson_sequential(K, T)
        ratio_sequential = thompson_sequential.run()
        thompson_periodic = Thompson_periodic(K, T)
        ratio_periodic = thompson_periodic.run()
        cost_single.append(attacker_single.attack_cost)
        cost_sequential.append(attacker_sequential.attack_cost)
        cost_periodic.append(attacker_periodic.attack_cost)
    plt.figure(figsize=(10, 6))
    plt.plot(range(100000, 1000001, 100000), cost_single, label='Single Injection')
    plt.plot(range(100000, 1000001, 100000), cost_sequential, label='Sequential Injection')
    plt.plot(range(100000, 1000001, 100000), cost_periodic, label='Periodic Injection')
    # 给每个点加上三角形、圆形、正方形的标记
    plt.scatter(range(100000, 1000001, 100000), cost_single, marker='^', color='blue')
    plt.scatter(range(100000, 1000001, 100000), cost_sequential, marker='o', color='orange')
    plt.scatter(range(100000, 1000001, 100000), cost_periodic, marker='s', color='green')
    plt.xlabel("T")
    plt.ylabel("Cost")
    plt.title("Thompson Sampling")
    plt.legend()
    plt.savefig("thompson-cost-T.png")
