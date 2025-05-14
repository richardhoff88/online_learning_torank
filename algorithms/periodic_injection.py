import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os

sys.path.append(os.path.abspath(".."))
from attack import UCBRecommender, get_real_reward

def beta(N, sigma, n_arms, delta):
    N = max(N, 1)
    log_argument = max((math.pi**2 * N**2 * n_arms) / (3 * delta), 1)
    return math.sqrt((2 * sigma**2 / N) * math.log(log_argument))

def get_reward_from_matrix(reward_matrix, arm):
    return np.mean(reward_matrix[:, arm])

def periodic_injection_attack_real(n_arms, target_arm, rho, T, reward_matrix, a_tilde, f, R, delta0, sigma = 1):
    recommender = UCBRecommender(n_arms, rho)
    target_pulls = 0
    estimated_rewards = np.zeros(n_arms)
    arm_pulls = np.zeros(n_arms)
    target_pull_ratios = []
    attack_cost = 0.0
    attack_list = []
    injection_list = []
    sleep_counter = 0
    for t in range(1, T + 1 + n_arms):
        arm = recommender.play()
        reward = get_reward_from_matrix(reward_matrix, arm)
        arm_pulls[arm] += 1
        estimated_rewards[arm] = (estimated_rewards[arm] * (arm_pulls[arm] - 1) + reward) / arm_pulls[arm]

        if t > n_arms:
            # print(t)
            # print(f"Injection list: {len(injection_list)}")
            if sleep_counter <= 0:
                for arm_attack in attack_list:
                    mu_i = estimated_rewards[arm_attack]
                    mu_k = estimated_rewards[target_arm]
                    beta_k = beta(arm_pulls[target_arm], sigma, n_arms, delta0)
                    l_hat = mu_k - 2 * beta_k - 3 * sigma * delta0

                    # Number of injection rounds n_tilde
                    a_tilde_new = min(a_tilde, mu_k - 3 * beta_k - 3 * sigma * delta0)
                    n_tilde = (mu_i - l_hat) * math.log(T) / (l_hat - a_tilde_new)/ delta0**2
                    # print(f"N_tilde is {n_tilde}")
                    # print(f"f is {f}")
                    mu_tic = ((arm_pulls[arm_attack] * mu_i) + f*a_tilde) / (arm_pulls[arm_attack] + f)
                    # print(f"mu_i(t_i(c)) is {mu_tic}")
                    # print(f"N_i(t) is {arm_pulls[arm_attack]}")
                    # print((((mu_k - 2 * beta_k - mu_tic) / (3 * sigma)) ** 2))
                    exponent = (((mu_k - 2 * beta_k - mu_tic) / (3 * sigma)) ** 2) * (arm_pulls[arm_attack] + f)
                    # print(f"The term inside the exp is {exponent}")
                    r_new = min(R, (n_tilde/f) * math.exp(exponent) - t - f)
                    # print(f"R is : {r_new}")
                    for _ in range(int(n_tilde)):
                        injection_list.insert(0, (arm_attack, a_tilde_new, r_new))
                    attack_list.remove(arm_attack)
                injections_to_do = injection_list[:min(f, len(injection_list))]
                counter = 0
                r = 0
                for injection in injections_to_do:
                    counter += 1
                    r = injection[2]
                    recommender.update(injection[0], injection[1])
                    attack_cost += abs(injection[1] - estimated_rewards[injection[0]])
                injection_list = injection_list[counter:]

                sleep_counter = r
            # print(f"Arm pulled {arm}")
            # print(arm==target_arm)
            if arm != target_arm and arm_pulls[arm] >= math.log(T / (delta0**2)) and arm not in attack_list:
                attack_list.append(arm)

            if arm == target_arm:
                target_pulls += 1

            target_pull_ratio = target_pulls / t
            target_pull_ratios.append(target_pull_ratio)
            if sleep_counter > 0:
                sleep_counter -= 1
        recommender.update(arm, reward)

    return target_pulls, target_pull_ratios, attack_cost

def experiment_real_periodic_bounded_injection(T=int(1e4), n_arms=10, rho=1.0, a_tilde = 0.0, sigma=1.0, delta0=0.2, R = 30, f = 10, trials=10):
    all_ratios = []

    for _ in range(trials):
        reduced_matrix = np.load(os.path.join("..", "dataset", "movielens.npy"))
        selected_movie_indices = np.random.choice(reduced_matrix.shape[1], size=n_arms, replace=False)
        # Slice the matrix to keep only n_arms
        reduced_matrix = reduced_matrix[:, selected_movie_indices]
        # Use smallest reward arm as target
        movie_interactions = np.sum(reduced_matrix, axis=0)
        least_interacted_movie = np.argmin(movie_interactions)
        target_arm = least_interacted_movie

        target_pulls, target_pull_ratios, attack_cost = periodic_injection_attack_real(
            n_arms, target_arm, rho, T, reduced_matrix, a_tilde=a_tilde, f=f, R=R, sigma=sigma, delta0=delta0)
        all_ratios.append(target_pull_ratios)

    avg_ratios = np.mean(all_ratios, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, T + 1), avg_ratios, label="Average Target Arm Selection Ratio")
    plt.xlabel("Rounds")
    plt.ylabel("Target Arm Selection Ratio")
    plt.title(f"Average Target Arm Selection Ratio Over Time (Periodic Bounded Injection Attack for Real Dataset) with f = {f}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_attack_cost_real(n_arms=10, rho=1.0, a_tilde=0.0, sigma=1.0, delta0=0.2, R = 30, f= 10, trials=5):
    avg_costs = []
    T_values = np.logspace(1, 4, num=10, dtype=int)
    for T in T_values:
        trial_costs = []
        for _ in range(trials):
            reduced_matrix = np.load(os.path.join("..", "dataset", "movielens.npy"))
            selected_movie_indices = np.random.choice(reduced_matrix.shape[1], size=n_arms, replace=False)
            reduced_matrix = reduced_matrix[:, selected_movie_indices]
            
            movie_interactions = np.sum(reduced_matrix, axis=0)
            target_arm = np.argmin(movie_interactions)

            target_pulls, target_pull_ratios, attack_cost = periodic_injection_attack_real(
            n_arms, target_arm, rho, T, reduced_matrix, a_tilde=a_tilde, f=f, R=R, sigma=sigma, delta0=delta0)
            trial_costs.append(attack_cost)
        
        avg_costs.append(np.mean(trial_costs))

    plt.figure(figsize=(8, 5))
    plt.plot(T_values, avg_costs, marker='o')
    plt.xlabel("T (Rounds)")
    plt.ylabel("Average Attack Cost")
    plt.title(f"Attack Cost vs. T for Periodic Injection Attack (MovieLens Dataset)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_attack_cost_vs_delta0_real(n_arms=10, rho=1.0, T=int(1e4), a_tilde=0, sigma=1.0, R = 20, f= 10, trials=10):
    avg_costs = []
    delta0_values = np.linspace(0.1, 0.5, num=20) 

    for delta0 in delta0_values:
        trial_costs = []
        for _ in range(trials):
            reduced_matrix = np.load(os.path.join("..", "dataset", "movielens.npy"))
            selected_movie_indices = np.random.choice(reduced_matrix.shape[1], size=n_arms, replace=False)
            reduced_matrix = reduced_matrix[:, selected_movie_indices]

            movie_interactions = np.sum(reduced_matrix, axis=0)
            target_arm = np.argmin(movie_interactions)

            target_pulls, target_pull_ratios, attack_cost = periodic_injection_attack_real(
            n_arms, target_arm, rho, T, reduced_matrix, a_tilde=a_tilde, f=f, R=R, sigma=sigma, delta0=delta0)

            trial_costs.append(attack_cost)

        avg_costs.append(np.mean(trial_costs))

    plt.figure(figsize=(8, 5))
    plt.plot(delta0_values, avg_costs, marker='o')
    plt.xlabel("δ₀ (Confidence Parameter)")
    plt.ylabel("Average Attack Cost")
    plt.title(f"Attack Cost vs. δ₀ for Periodic Injection Attack (MovieLens Dataset) with T = {T}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    experiment_real_periodic_bounded_injection()
    # plot_attack_cost_vs_delta0_real()