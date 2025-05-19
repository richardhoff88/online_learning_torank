from typing import List
import numpy as np
from scipy.stats import norm

import importlib

module_map = {
    1: "single_injection_ts",
    2: "sequential_injection_ts",
    3: "periodic_injection_ts"
}

condition = 1 
module_name = module_map.get(condition, "no_attack")
attacker = importlib.import_module(module_name).attacker(K=10, T=1000, delta=0.05, sigma0=1.0)

class Thompson:

    def __init__(self, K: int, T: int, sigma: float = 0.05):
        self.K = K
        self.T = T
        self.n = [0] * K
        self.empirical_means = [0.0] * K
        self.sigma = sigma

    def get_reward(self, k: int) -> float:
        return attacker.get_reward(k)

    def update(self, k: int, reward: float):
        self.n[k] += 1
        self.empirical_means[k] = ((self.n[k] - 1) * self.empirical_means[k] + reward) / self.n[k]

    def select_arm(self) -> int:
        if attacked is True:
            return attacker.select_arm()
        else:
            sampled_means = [np.random.normal(loc=self.empirical_means[k], scale=1/np.sqrt(self.n[k])) for k in range(self.K)]
            return np.argmax(sampled_means)
    
    def run(self) -> List[float]:
        select_cnt = [0] * self.K
        for t in range(1, self.K+1):
            k = t - 1
            reward = self.get_reward(k)
            self.update(k, reward)
            select_cnt[k] += 1
        for t in range(self.K+1, self.T+1):
            k = self.select_arm()
            reward = self.get_reward(k)
            self.update(k, reward)
            select_cnt[k] += 1
        return select_cnt