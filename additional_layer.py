""" Simulating click model """
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


# just took from multi-armed bandits github
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



def simulate_UCB_attack(n_arms, target_arm, rho, rounds, frequency, probabilities, ratio):
    # changed this so it initializes as tuple
    arm_counts = np.zeros((n_arms, rounds))
    real_users = UCBRecommender(n_arms, rho)
    counter = 0
    for i in range(0, rounds + n_arms):
        # use play function from ucb 
        # play() returns the index of the selected arm
        real_arm = real_users.play()
        if i >= n_arms:
            arm_counts[real_arm, i-10] += 1
        if real_arm == 0:
            counter += 1
        #figure out some reward function
        real_reward = probabilities[real_arm]
        real_users.update(real_arm, real_reward)
        # fake user arm selection

        if np.random.rand() < (frequency / math.log(i + math.e, math.e)): # threshold for attack frequency (log correlation to time step); different types of attacks
            for _ in range(ratio):
                if real_arm != target_arm:
                    real_users.update(real_arm, 0, fake=True)
                else:
                    real_users.update(real_arm, 1, fake=True)
    return arm_counts, counter

# look at the bad cases and design a separate algorithm that can be more effective
# new attack based on observations from the real users can we design criterion to claim that its impossible
# or not sufficient number of fake users

def adaptive_attack(real_arm, target_arm, ratio, frequency, attack_controller, round_num):
    """
    adding more fake users and increasing frequency of attack
    """
    attack_threshold = frequency / math.log(round_num + math.e)
    
    if not attack_controller.evaluate_attack_effectiveness(threshold=0.5):
        attack_threshold *= 1.5
        ratio += 1
    
    if np.random.rand() < attack_threshold:
        for _ in range(ratio):
            if real_arm != target_arm:
                return 0, True  
            else:
                return 1, True 
    return None, False 


frequency = frequencies[3]
ratio = ratios[1]
target_arm = 0
arm_counts, chosen_times = simulate_UCB_attack(n_arms, target_arm, rho, rounds, frequency, probabilities, ratio)
print(chosen_times)
print(float(chosen_times)/rounds * 100)
#convert our 2darray to matrix
matrix = np.array(arm_counts)

percentage = float(chosen_times)/rounds * 100
plt.scatter(range(rounds), percentage[:rounds])
plt.xlabel('Rounds')
plt.ylabel('Percentage of pulling the target arm (0)')
plt.show()


#plotting 1's and 0's distribution in 2d
plt.imshow(matrix, cmap='viridis',aspect='auto')
plt.title(f'Number of attacks per round based on frequency: {frequency}')
plt.ylabel('number of arms selected')
plt.xlabel('rounds')
plt.show()

# New figure - surface plot used for 2d matrics
x, y = np.meshgrid(range(matrix.shape[1]), range(matrix.shape[0]))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, matrix, cmap='viridis')

# check this
# right now plot shows 1's and 0's on plot in 3d
ax.set_ylabel('number of arms selected')
ax.set_xlabel('rounds')
ax.set_zlabel('# of attacks (1 or 0)')
plt.title(f'Number of attacks per round based on frequency: {frequency}')
plt.show()
print(len(arm_counts[0]))

# nonlinear function of fake user attacks?
# percentage of target arm pulls (bar chart)
# identify cost; total number of users to fake users ratio
# identify correlation for when attack doesn't work
# learn real reward distribution?