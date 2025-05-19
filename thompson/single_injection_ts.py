from math import sqrt, log

# from thompson import Thompson

class attcker:
    def __init__(self, K: int, T: int, delta: float, sigma0: float):
        self.attack_round =  int(log(T) / (delta * delta)) + 1 # 向上取整
        self.K = K
        self.T = T
        self.delta = delta
        self.sigma0 = sigma0
        self.N = [0] * K

    def ell(self, sigma: float):
        # TODO: calculate it
        pass

    def condition(self, k: int) -> bool:
        """
        Here we need just one injection for each arm.
        """
        return self.N[k] == self.attack_round
    

    def fake_reward(self, k: int) -> float:
        # TODO: calculate it
        pass

    def select_arm(self) -> int:
        # TODO: return the injected arm
        pass

    def get_reward(self, k: int) -> float:
        self.N[k] += 1 # update the number of times arm k has been selected
        if self.condition(k) is True:
            return self.fake_reward(k)
        else:
            return 0x7fffffff  # TODO: return the real reward

    
