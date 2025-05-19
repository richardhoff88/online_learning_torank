from typing import List, Tuple
from math import sqrt, log

# from thompson import Thompson
import numpy as np

class attcker:
    def __init__(self, K: int, T: int, delta: float, sigma0: float):
        self.attack_round =  int(log(T) / (delta * delta)) + 1 # 向上取整
        self.K = K
        self.T = T
        self.delta = delta
        self.sigma0 = sigma0
        self.n = [0] * K
        self.empirical_means = [0.0] * K

    def update(self, k: int, reward: float):
        self.n[k] += 1
        self.empirical_means[k] = ((self.n[k] - 1) * self.empirical_means[k] + reward) / self.n[k]

    def ell(self, sigma: float):
        # TODO: calculate it
        pass

    def condition(self, k: int) -> bool:
        """
        Here we need just one injection for each arm.
        """
        return self.n[k] == self.attack_round
    

    def fake_reward(self, k: int) -> float:
        # TODO: calculate it
        pass

    def feedback(self) -> Tuple[int, float]:
        for i in range(self.K):
            if self.condition(i):
                ret = self.fake_reward(i)
                self.update(i, ret)
                return i, self.fake_reward(i)
        sampled_means = [np.random.normal(loc=self.empirical_means[k], scale=1/np.sqrt(self.n[k])) for k in range(self.K)]
        return np.argmax(sampled_means), 0.0 # TODO: return the real reward