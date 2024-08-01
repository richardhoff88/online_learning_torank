""" Simulating click model """
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



n_arms = 10  # number of arms
rho = 1.0 # explore-exploit value
rounds = 100 # rounds
ratios = [1, 2, 5, 2]  # list of ratios, fake to real
frequencies = [0.1, 0.25, 0.5, 1.0]
probabilities = np.random.rand(n_arms) * 1 # probability from 0 to 1 of giving a reward of 1



# # load dataset here
# def load_data():
#     data = np.loadtxt("dataset.txt")
#     arms, rewards, contexts = data[:,0], data[:,1], data[:,2:]
#     arms = arms.astype(int)
#     rewards = rewards.astype(float)
#     contexts = contexts.astype(float)
#     n_arms = len(np.unique(arms))
#     n_events = len(contexts)
#     n_dims = int(len(contexts[0])/n_arms)
#     contexts = contexts.reshape(n_events, n_arms, n_dims)


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
        non_zero_clicks = self.clicks + 1e-10  # Add a small constant to avoid division by zero
        self.q = self.avg_rewards + np.sqrt(self.rho * np.log(self.round) / non_zero_clicks)
        arm = np.argmax(self.q)
        return int(arm)

    def update(self, arm, reward, fake=False):
        if not fake:
            self.clicks[arm] += 1
        self.rewards[arm] += reward
        self.avg_rewards[arm] = self.rewards[arm] / self.clicks[arm]



def simulate_UCB_attack(n_arms, rho, rounds, frequency, probabilities, ratio):
    # changed this so it initializes as tuple
    arm_counts = np.zeros((n_arms, rounds))
    real_users = UCBRecommender(n_arms, rho)

    for i in range(0, rounds + n_arms):
        # use play function from ucb 
        # play() returns the index of the selected arm
        real_arm = real_users.play()
        if i >= n_arms:
            arm_counts[real_arm, i-10] += 1
        print(real_arm)
        #figure out some reward function
        real_reward = probabilities[real_arm]
        real_users.update(real_arm, real_reward)
        # fake user arm selection
        if np.random.rand() < frequency:
            for _ in range(ratio):
                real_users.update(0, 1, True)


    return arm_counts

frequency = frequencies[0]
ratio = ratios[0]
arm_counts = simulate_UCB_attack(n_arms, rho, rounds, frequency, probabilities, ratio)
#convert our 2darray to matrix
matrix = np.array(arm_counts)

#plotting 1's and 0's distribution in 2d
plt.imshow(matrix, cmap='viridis',aspect='auto')
plt.title(f'Frequency of attack based on ratio: {frequency}')
plt.show()

# New figure - surface plot used for 2d matrics
x, y = np.meshgrid(range(matrix.shape[1]), range(matrix.shape[0]))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, matrix, cmap='viridis')

# check this
# right now plot shows 1's and 0's on plot in 3d
ax.set_xlabel('number of arms')
ax.set_ylabel('rounds')
ax.set_zlabel('# of attacks (1 or 0)')
plt.title(f'Frequency of attack based on ratio: {frequency}')
plt.show()
print(len(arm_counts[0]))


