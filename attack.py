import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math



# generate mean through [0,1]; no need to randomize standard deviation

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
        self.pulls = np.zeros(n_arms)
        self.recommended_times = np.zeros(n_arms)
        self.round = 0

    def play(self, context=None):
        self.round += 1
        for arm in range(len(self.q)):
            if self.recommended_times[arm] == 0:
                return arm
        recommended_times = self.recommended_times  # Add a small constant to avoid division by zero
        self.q = self.avg_rewards + np.sqrt(self.rho * np.log(self.round) / recommended_times)
        # print(self.avg_rewards[0])
        arm = np.argmax(self.q)
        return int(arm)

    def update(self, arm, reward, fake=False):
        if not fake:
            self.recommended_times[arm] += 1
        self.pulls[arm] += 1
        self.rewards[arm] += reward
        self.avg_rewards[arm] = self.rewards[arm] / self.pulls[arm]

def get_real_reward(means, std_devs, arm):
    return np.random.normal(means[arm], std_devs[arm])

def beta(N, sigma, n_arms, delta):
    # Calculate the beta value for a given N
    return math.sqrt((2 * sigma**2 / N) * math.log((math.pi**2 * N**2 * n_arms) / (math.e * delta)))

def simulate_UCB_attack(n_arms, target_arm, rho, rounds, means, std_devs, real_user_count, sigma = 1, delta = 0.05):
    arm_counts = np.zeros((n_arms, rounds + n_arms))
    recommender = UCBRecommender(n_arms, rho)
    target_pull_counter = 0
    estimated_reward_distribution = np.zeros(n_arms)
    arm_pulls = np.zeros(n_arms)
    attack_flag = True
    attack_trials_list = []
    for round_num in range(0, rounds + n_arms):
        if round_num % real_user_count == 0:
            attack_flag = True
        # use play function from ucb 
        # play() returns the index of the selected arm
        real_arm = recommender.play()
        arm_counts[real_arm, round_num] += 1
        if real_arm == target_arm:
            target_pull_counter += 1
        # Figure out some reward function
        real_reward = get_real_reward(means, std_devs, real_arm)
        # Update estimated reward distribution
        arm_pulls[real_arm] += 1
        estimated_reward_distribution[real_arm] = (estimated_reward_distribution[real_arm] * (arm_pulls[real_arm] - 1) + real_reward) / arm_pulls[real_arm]
    
        recommender.update(real_arm, real_reward)
        # fake user arm selection
        if round_num > n_arms:
            if attack_flag and real_arm != target_arm:
                N_target = arm_pulls[target_arm]
                attack_beta = beta(N_target, sigma, n_arms, delta)
                fake_reward = estimated_reward_distribution[target_arm] - 2 * attack_beta - 3 * sigma
                recommender.update(real_arm, fake_reward)
                attack_trials_list.append(round_num)
    return arm_counts, attack_trials_list, target_pull_counter

def plot_attacks(trials = 1000, rounds = 1000, real_user_count = 10, n_arms = 10, rho = 1.0, sigma=1, delta = 0.05):
    target_arm = n_arms - 1
    chosen_ratios = []
    failed_attack = 0
    failed_attacks_reward = []
    for i in range(trials):
        means = np.random.rand(n_arms)
        std_devs = np.full(n_arms, sigma)
        _, _, chosen_times = simulate_UCB_attack(n_arms, target_arm, rho, rounds, means, std_devs, real_user_count, sigma=sigma, delta=delta)
        chosen_ratio = float(chosen_times)/rounds * 100
        if chosen_ratio < 90:
            failed_attack += 1
            failed_attacks_reward.append(means[0])
        chosen_ratios.append(chosen_ratio)


    print(f"We have {failed_attack} failed_attacks out of {trials} total attacks")
    print(chosen_ratios)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, trials + 1), chosen_ratios, marker='o', linestyle='-', color='blue')
    plt.axhline(y=np.mean(chosen_ratios), color='red', linestyle='--', label='Mean Ratio')
    plt.title('Chosen Ratio for Target Arm Across Simulations')
    plt.xlabel('Simulation Number')
    plt.ylabel('Chosen Ratio (%)')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_attacks()



