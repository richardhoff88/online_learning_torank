""" Simulating click model """
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

n_arms = 10  # number of arms
rho = 1.0 # explore-exploit value
rounds = 10 # rounds
ratios = [0.1, 0.25, 0.5, 1.0, 1.50, 2.0, 5.0]  # list of ratios, fake to real
ratio = 1.5
probabilities = np.random.rand(n_arms) # probability from 0 to 1 of giving a reward of 1


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
        self.round = 0

    def play(self, context=None):
        self.round += 1
        self.q = np.where(self.clicks != 0, self.avg_rewards + np.sqrt(self.rho * np.log10(self.round) / self.clicks), self.q)
        arm = np.argmax(self.q)
        return int(arm)

    def update(self, arm, reward, context=None):
        self.clicks[arm] += 1
        self.rewards[arm] += reward
        self.avg_rewards[arm] = self.rewards[arm] / self.clicks[arm]



def simulate_UCB_attack(n_arms, rho, rounds, ratio, probabilities):
    # changed this so it initializes as tuple
    arm_counts = np.zeros((n_arms, rounds))
    real_users = UCBRecommender(n_arms, rho)

    for i in range(1, rounds):
        # use play function from ucb 
        # play() returns the index of the selected arm
        real_arm = real_users.play()
        #figure out some reward function
        real_reward = real_arm * probabilities[real_arm]
        # fake user arm selection

        if np.random.rand() < ratio:
            fake_arm = np.random.randint(n_arms)
            fake_reward = 1
            real_users.update(fake_arm, fake_reward)
            arm_counts[fake_arm, i] += fake_reward

        # real_users.update(real_arm, real_reward)
        # arm_counts[real_arm, i] += 1
    return arm_counts

# 2d subplots comparing all ratios in list
num_plots = len(ratios)
fig, axs = plt.subplots(nrows=1, ncols=num_plots, figsize=(4 * num_plots, 4), constrained_layout=True)

for i, ratio in enumerate(ratios):
    arm_counts = simulate_UCB_attack(n_arms, rho, rounds, ratio, probabilities)
    matrix = np.array(arm_counts)
    
    im = axs[i].imshow(matrix, cmap='viridis', aspect='auto')
    axs[i].set_title(f'Ratio: {ratio}')
    axs[i].set_xlabel('Rounds')
    axs[i].set_ylabel('Arms')

# 3d plots to loop through all the ratios - just to get better visual
for ratio in ratios:
    arm_counts = simulate_UCB_attack(n_arms, rho, rounds, ratio, probabilities)
    matrix = np.array(arm_counts)
    # New figure - surface plot used for 2d matrics
    x, y = np.meshgrid(range(matrix.shape[1]), range(matrix.shape[0]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, matrix, cmap='viridis')

    # check this
    # right now plot shows 1's and 0's on plot in 3d
    ax.set_xlabel('number of arms')
    ax.set_ylabel('rounds')
    ax.set_zlabel('# of times attacked (1 or 0)')
    plt.title(f'Frequency of attack based on ratio: {ratio}')
    plt.show()


