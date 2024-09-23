import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

n_arms = 10  # number of arms
rho = 1.0 # explore-exploit value
rounds = 1000 # rounds
ratios = [1, 2, 5, 100]  # list of ratios, fake to real
frequencies = [0.1, 0.25, 0.5, 1.0]
probabilities = np.random.rand(n_arms) * 1 # probability from 0 to 1 of giving a reward of 1
print(probabilities[0])

class UCBRecommender:
    def __init__(self, n_arms, rho, Q0=np.inf):
        if not rho > 0:
            raise ValueError("`rho` must be positive")
        if not (type(rho) == float and np.isreal(rho)):
            raise TypeError("`rho` must be real float")
        if not type(Q0) == float:
            raise TypeError("`Q0` must be a float number or default value 'np.inf'")

        self.rho = rho
        self.q = np.full(n_arms, Q0)
        self.rewards = np.zeros(n_arms)
        self.avg_rewards = np.zeros(n_arms)
        self.clicks = np.zeros(n_arms)
        self.recommended_times = np.zeros(n_arms)
        self.round = 0

    def play(self, context=None):
        self.round += 1
        recommended_times = self.recommended_times + 1e-10  # Add a small constant to avoid division by zero
        self.q = self.avg_rewards + np.sqrt(self.rho * np.log(self.round) / recommended_times)
        # print(self.avg_rewards[0])
        arm = np.argmax(self.q)
        return int(arm)

    def update(self, arm, reward, fake=False):
        if not fake:
            self.recommended_times[arm] += 1
        self.clicks[arm] += 1
        self.rewards[arm] += reward
        self.avg_rewards[arm] = self.rewards[arm] / self.clicks[arm]

class AttackController:
    def __init__(self, effectiveness_threshold):
        self.effectiveness_threshold = effectiveness_threshold
        self.successful_attacks = 0
        self.total_attacks = 0
    
    def evaluate_attack_effectiveness(self, real_arm, target_arm):
        # Check if the attack was successful
        if real_arm == target_arm:
            self.successful_attacks += 1
        self.total_attacks += 1
        
        # Calculate effectiveness
        effectiveness = self.successful_attacks / self.total_attacks if self.total_attacks > 0 else 0
        return effectiveness >= self.effectiveness_threshold

def adaptive_attack(attack_controller, real_arm, target_arm, ratio, frequency, round_num):
    """
    adding more fake users and increasing frequency of attack
    """
    attack_threshold = frequency / math.log(round_num + math.e)
    
    if not attack_controller.evaluate_attack_effectiveness(real_arm, target_arm):
        attack_threshold *= 1.5
        ratio += 1
    

    if np.random.rand() < attack_threshold:
        for _ in range(ratio):
            if real_arm != target_arm:
                return 0, True 
            else:
                return 1, True 
    return None, False 

def simulate_adaptive_UCB_attack(n_arms, target_arm, rho, rounds, frequency, probabilities, ratio, attack_threshold=0.5):
    arm_counts = np.zeros((n_arms, rounds))
    real_users = UCBRecommender(n_arms, rho)
    attack_controller = AttackController(effectiveness_threshold=attack_threshold)
    
    counter = 0
    for i in range(0, rounds + n_arms):
        real_arm = real_users.play()
        
        if i >= n_arms:
            arm_counts[real_arm, i - n_arms] += 1
        if real_arm == target_arm:
            counter += 1
        
        real_reward = probabilities[real_arm]
        real_users.update(real_arm, real_reward)
        
        # Simulate adaptive attack
        fake_reward, attack_occurred = adaptive_attack(attack_controller, real_arm, target_arm, ratio, frequency, i)
        if attack_occurred:
            real_users.update(real_arm, fake_reward, fake=True)
    
    return arm_counts, counter

frequency = frequencies[3]
ratio = ratios[1]
target_arm = 0
attack_threshold = 0.5  
arm_counts, chosen_times = simulate_adaptive_UCB_attack(n_arms, target_arm, rho, rounds, frequency, probabilities, ratio, attack_threshold)

print(chosen_times)
percentage = float(chosen_times) / rounds * 100
print(percentage)

matrix = np.array(arm_counts)
plt.imshow(matrix, cmap='viridis', aspect='auto')
plt.title(f'Number of attacks per round with adaptive attack strategy (frequency: {frequency})')
plt.ylabel('Number of arms selected')
plt.xlabel('Rounds')
plt.show()

x, y = np.meshgrid(range(matrix.shape[1]), range(matrix.shape[0]))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, matrix, cmap='viridis')
ax.set_ylabel('Number of arms selected')
ax.set_xlabel('Rounds')
ax.set_zlabel('# of attacks (1 or 0)')
plt.title(f'Number of attacks per round with adaptive attack strategy (frequency: {frequency})')
plt.show()