from typing import Tuple
from math import sqrt, log, pi, exp

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
        self.next_inject = [self.T] * K
        self.gap = [0] * K
        self.cnt = [0] * K
        self.fake_num = 10 # 每次插入的fake data的数量
        self.inject_number = [0] * K
        self.lower_bound = -36.66 # tilde a
        self.time_slot = K
        self.attack_cost = 0

    def update(self, k: int, reward: float):
        self.n[k] += 1
        self.empirical_means[k] = ((self.n[k] - 1) * self.empirical_means[k] + reward) / self.n[k]

    def beta(self, N: int) -> float:
        return sqrt(2 * self.sigma * self.sigma / N * log((pi ** 2) * self.K * (N ** 2) / (3 * self.delta)))

    def ell(self, i: int):
        return self.empirical_means[self.K-1] - 2 * self.beta(self.n[self.K-1]) - sqrt(8 * log((pi ** 2) * self.K / (3*self.delta))) - 4 * sqrt(self.n[i]) * self.delta0

    def condition(self, i: int) -> bool:
        """
        Here we need just one injection for each arm.
        """
        if self.n[i] == self.attack_round:
            ell = self.ell(i)
            self.cnt[i] = int((self.empirical_means[i] - ell) / (ell-self.lower_bound) * log(self.T) / (self.delta0 ** 2)) + 1
            self.inject_number[i] = int(self.cnt[i] / self.fake_num) + 1

            mu_i_c = [(self.empirical_means[i] * self.n[i] + self.fake_num * self.lower_bound * x) / (self.n[i] + self.fake_num * x) for x in range(1, self.inject_number[i]+1)]

            self.gap[i] = min((sqrt(exp(((self.empirical_means[self.K-1]-2*self.beta(self.n[self.K-1]) - mu_i_c[x-1]) ** 2 / 8)) / ((pi**2) * self.K)) - self.time_slot) / x - self.fake_num for x in range(1, self.inject_number[i]+1))
            self.next_inject[i] = self.time_slot
            self.cnt[i] = self.fake_num

        if self.time_slot >= self.next_inject[i]:
            if self.cnt[i] <= 0:
                self.inject_number[i] -= 1
                if self.inject_number[i] > 0:
                    self.cnt[i] = self.fake_num
                    self.next_inject[i] += self.gap[i]
                else:
                    self.next_inject[i] = self.T + 1
            else:
                self.cnt[i] -= 1
                self.next_inject[i] += 1
            return True

        return False

    def fake_reward(self) -> float:
        return self.lower_bound

    def feedback(self) -> Tuple[int, float]:
        self.time_slot += 1
        for i in range(self.K-1):
            if self.condition(i):
                ret = self.fake_reward()
                self.update(i, ret)
                self.attack_cost += real_mean(i) - ret
                return i, ret
        sampled_means = [np.random.normal(loc=self.empirical_means[i], scale=1/np.sqrt(self.n[i])) for i in range(self.K)]
        i = np.argmax(sampled_means)
        r = get_reward(i, self.sigma)
        self.update(i, r)
        return i, r