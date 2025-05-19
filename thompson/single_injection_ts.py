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
        return self.N[k] == self.attack_round
    

    def get_reward(self, k: int) -> float:
        if self.condition(k) is True:
            return -0x80000000 # TODO: calculate it, fake reward
        else:
            return 0x7fffffff  # TODO: calculate it, real reward

    
