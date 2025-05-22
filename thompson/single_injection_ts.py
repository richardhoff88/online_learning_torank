from typing import Tuple
from math import sqrt, log, pi

import numpy as np

from real_reward import get_reward, real_mean

class attacker:
    def __init__(self, K: int, T: int, delta: float, delta0: float, sigma: float):
        self.attack_round =  int(log(T) / (delta0 * delta0)) + 1 # 向上取整
        self.K = K
        self.T = T
        self.delta = delta
        self.delta0 = delta0
        self.sigma = sigma
        self.n = [1] * K
        self.empirical_means = [0.0] * K
        self.time_slot = K
        self.attack_cost = 0

    def update(self, k: int, reward: float):
        self.n[k] += 1
        self.empirical_means[k] = ((self.n[k] - 1) * self.empirical_means[k] + reward) / self.n[k]

    def beta(self, N: int) -> float:
        return sqrt(2 * self.sigma * self.sigma / N * log((pi ** 2) * self.K * (N ** 2) / (3 * self.delta)))

    def ell(self, i: int):
        return self.empirical_means[self.K-1] - 2 * self.beta(self.n[self.K-1]) - sqrt(8 * log((pi ** 2) * self.K / (3*self.delta))) - 4 * sqrt(self.n[i]) * self.delta0

    def condition(self, k: int) -> bool:
        """
        Here we need just one injection for each arm.
        """
        return self.n[k] == self.attack_round
    

    def fake_reward(self, i: int) -> float:
        ell = self.ell(i)
        return self.n[i] * (ell - self.empirical_means[i]) + ell

    def feedback(self) -> Tuple[int, float]:
        self.time_slot += 1
        for i in range(self.K-1):
            if self.condition(i):
                ret = self.fake_reward(i)
                self.update(i, ret)
                self.attack_cost+= real_mean(i) - ret
                return i, ret
        sampled_means = [np.random.normal(loc=self.empirical_means[k], scale=1/np.sqrt(self.n[k])) for k in range(self.K)]
        i = np.argmax(sampled_means)
        r = get_reward(i, self.sigma)
        self.update(i, r)
        return i, r