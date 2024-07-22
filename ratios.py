import numpy as np
import matplotlib.pyplot as plt


n_real = 100  # number of real users
k = 5  # number of arms
target_arm = 0  # Index of the target arm
iterations = 1000 
ratios = [0.1, 0.25, 0.5, 1.0, 1.50, 2.0, 5.0]  # list of ratios, fake to real
results = []




























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
