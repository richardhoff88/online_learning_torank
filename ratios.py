""" Simulating click model """
import numpy as np
import matplotlib.pyplot as plt


n_arms = 10  # number of arms
rho = 1.0 # explore-exploit value
time = 10 # rounds
ratios = [0.1, 0.25, 0.5, 1.0, 1.50, 2.0, 5.0]  # list of ratios, fake to real
probabilities = np.random.rand(n_arms) # probability from 0 to 1 of giving a reward of 1


# load dataset here
def load_data():
    data = np.loadtxt("dataset.txt")
    arms, rewards, contexts = data[:,0], data[:,1], data[:,2:]
    arms = arms.astype(int)
    rewards = rewards.astype(float)
    contexts = contexts.astype(float)
    n_arms = len(np.unique(arms))
    n_events = len(contexts)
    n_dims = int(len(contexts[0])/n_arms)
    contexts = contexts.reshape(n_events, n_arms, n_dims)


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


# for making the the different reward distribution
def simulate_UCB(n_arms, rho, time, ratios, probabilities):
    fake_users = UCBRecommender(n_arms, rho)
    real_users = UCBRecommender(n_arms, rho)

    fake_user_rewards = 0
    real_user_rewards = 0

    for i in range(1, time):
        # use play function from ucb 
        real_arm = real_users.play()
        fake_arm = fake_users.play()

        #figure out some reward function
        real_reward = real_arm * probabilities[real_arm]
        real_user_rewards += real_reward

        # fake arm selection
        for j in range(len(ratios)):
            if np.random.rand() < ratios[j]:
                fake_reward = fake_arm * probabilities[fake_arm]
                fake_user_rewards += fake_reward


        # for fake user rewards check with ratios list

    real_users.update(real_arm, real_reward)
    fake_users.update(fake_arm, fake_reward)

    #average?
    fake_avg = fake_user_rewards / time
    real_avg = real_user_rewards / time
    return real_avg, fake_avg



simulate_UCB(n_arms, rho, time, ratios, probabilities)




















# # Function to choose an arm based on user profile probabilities
# def choose_arm(user_profile):
#     return np.random.choice(range(k), p=user_profile)

# # Function to simulate binary feedback (success/failure) based on true success rates
# def get_feedback(arm, true_success_rates):
#     return np.random.rand() < true_success_rates[arm]

# # Function to update user profile based on feedback
# def update_profile(user_profile, arm, feedback, learning_rate=0.1):
#     if feedback:
#         user_profile[arm] += learning_rate
#     else:
#         user_profile[arm] -= learning_rate / (k - 1)
#     # Normalize the probability distribution to sum to 1
#     user_profile = np.maximum(user_profile, 0)  # Ensure no negative probabilities
#     user_profile /= user_profile.sum()
#     return user_profile

# # True success rates for each arm (for feedback simulation)
# true_success_rates = np.random.rand(k)

# # Simulation for different ratios of fake users to real users
# for ratio in ratios:
#     n_fake = int(ratio * n_real)
#     # Initialize fake user profiles with higher probability for the target arm
#     fake_users = np.full((n_fake, k), 0.2 / (k - 1))
#     fake_users[:, target_arm] = 0.8
#     # Initialize real user profiles with equal probability for each arm
#     real_users = np.full((n_real, k), 1 / k)
    
#     target_arm_counts = np.zeros(iterations)
    
#     for it in range(iterations):
#         selected_arms = []
        
#         # Fake users select arms
#         for fake_user in fake_users:
#             arm = choose_arm(fake_user)
#             selected_arms.append(arm)
        
#         # Real users select arms
#         for real_user in real_users:
#             arm = choose_arm(real_user)
#             selected_arms.append(arm)
        
#         # Collect feedback and update profiles
#         for i, arm in enumerate(selected_arms):
#             feedback = get_feedback(arm, true_success_rates)
#             if i < n_fake:
#                 # Update fake user profiles based on feedback
#                 fake_users[i] = update_profile(fake_users[i], arm, feedback)
#             else:
#                 # Update real user profiles based on feedback
#                 real_user_idx = i - n_fake
#                 real_users[real_user_idx] = update_profile(real_users[real_user_idx], arm, feedback)
        
#         # Count how many times the target arm is chosen
#         target_arm_counts[it] = selected_arms.count(target_arm)
    
#     average_target_arm_count = np.mean(target_arm_counts)
#     results.append(average_target_arm_count)

# # Plot the results
# plt.plot(ratios, results)
# plt.xlabel('Ratio of Fake Users to Real Users')
# plt.ylabel('Average Target Arm Count')
# plt.title('Influence of Fake Users on Recommendation')
# plt.show()
