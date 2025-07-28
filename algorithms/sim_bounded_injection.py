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

def calculate_a_tilde(mu_k, n_arms, sigma, delta0):
    beta_one = math.sqrt((2 * sigma**2) * math.log((math.pi**2 * n_arms) / (3 * delta0)))
    a_tilde = mu_k - 3 * beta_one - 3 * sigma * delta0
    return max(a_tilde, 0.0) 

def get_reward_from_matrix(reward_matrix, arm):
    return np.mean(reward_matrix[:, arm])

def simultaneous_bounded_injection_attack(n_arms, target_arm, rho, T, means, std_devs, a_tilde, delta0=0.05):
    recommender = UCBRecommender(n_arms, rho)
    target_pulls = 0
    attack_trials = []
    sigma = std_devs[0]
    estimated_rewards = np.zeros(n_arms)
    arm_pulls = np.zeros(n_arms)
    target_pull_ratios = []
    attack_cost = 0.0

    for t in range(1, T + 1 + n_arms):
        arm = recommender.play()
        reward = get_real_reward(means, std_devs, arm)

        # Update empirical reward
        arm_pulls[arm] += 1
        estimated_rewards[arm] = (estimated_rewards[arm] * (arm_pulls[arm] - 1) + reward) / arm_pulls[arm]
        
        if t > n_arms:
            # SBI Attack Condition
            if arm != target_arm and arm_pulls[arm] >= math.log(T / (delta0**2)):
                mu_i = estimated_rewards[arm]
                mu_k = estimated_rewards[target_arm]
                l_hat = mu_k - 2 * beta(arm_pulls[target_arm], sigma, n_arms, delta0)

                # Number of injection rounds n_tilde
                a_tilde_new = min(a_tilde, mu_k - 3 * beta(arm_pulls[target_arm], sigma, n_arms, delta0) - 3 * sigma * delta0)
                n_tilde = (mu_i - l_hat) * math.log(T) / (l_hat - a_tilde_new) / delta0**2
                for _ in range(int(n_tilde)):
                    recommender.update(arm, a_tilde_new)
                    attack_cost += abs(a_tilde_new - mu_i)

                attack_trials.append(t)

            if arm == target_arm:
                target_pulls += 1

            target_pull_ratio = target_pulls / t
            target_pull_ratios.append(target_pull_ratio)

        recommender.update(arm, reward)

    return target_pulls, attack_trials, target_pull_ratios, attack_cost

def simultaneous_bounded_injection_attack_real(n_arms, target_arm, rho, T, reward_matrix, a_tilde=0, sigma = 1, delta0=0.05):
    recommender = UCBRecommender(n_arms, rho)
    target_pulls = 0
    attack_trials = []
    estimated_rewards = np.zeros(n_arms)
    arm_pulls = np.zeros(n_arms)
    target_pull_ratios = []
    attack_cost = 0.0

    for t in range(1, T + 1 + n_arms):
        arm = recommender.play()
        reward = get_reward_from_matrix(reward_matrix, arm)

        # Update empirical reward
        arm_pulls[arm] += 1
        estimated_rewards[arm] = (estimated_rewards[arm] * (arm_pulls[arm] - 1) + reward) / arm_pulls[arm]

        if t > n_arms:
            # SBI Attack Condition
            if arm != target_arm and arm_pulls[arm] >= math.log(T / (delta0**2)):
                mu_i = estimated_rewards[arm]
                mu_k = estimated_rewards[target_arm]
                l_hat = mu_k - 2 * beta(arm_pulls[target_arm], sigma, n_arms, delta0) - 3 * sigma * delta0

                # Number of injection rounds n_tilde
                a_tilde_new = min(a_tilde, mu_k - 3 * beta(arm_pulls[target_arm], sigma, n_arms, delta0) - 3 * sigma * delta0)
                n_tilde = (mu_i - l_hat) * math.log(T) / (l_hat - a_tilde_new) / delta0**2
                for _ in range(int(n_tilde)):
                    recommender.update(arm, a_tilde_new)
                    attack_cost += abs(a_tilde_new - mu_i)
                    
                attack_trials.append(t)

            if arm == target_arm:
                target_pulls += 1

            target_pull_ratio = target_pulls / t
            target_pull_ratios.append(target_pull_ratio)

        recommender.update(arm, reward)

    return target_pulls, attack_trials, target_pull_ratios, attack_cost

def periodic_injection_attack_real(n_arms, target_arm, rho, T, reward_matrix, a_tilde, f, R, delta0, sigma = 1):
    recommender = UCBRecommender(n_arms, rho)
    target_pulls = 0
    estimated_rewards = np.zeros(n_arms)
    arm_pulls = np.zeros(n_arms)
    target_pull_ratios = []
    attack_cost = 0.0
    attack_list = []
    injection_dict = {} 
    sleep_counter = {} 
    a_tildes = {}     
    
    for t in range(1, T + 1 + n_arms):
        arm = recommender.play()
        reward = get_reward_from_matrix(reward_matrix, arm)
        arm_pulls[arm] += 1
        estimated_rewards[arm] = (estimated_rewards[arm] * (arm_pulls[arm] - 1) + reward) / arm_pulls[arm]

        if t > n_arms:
            for arm_attack in attack_list[:]:  
                mu_i = estimated_rewards[arm_attack]
                mu_k = estimated_rewards[target_arm]
                beta_k = beta(arm_pulls[target_arm], sigma, n_arms, delta0)
                l_hat = mu_k - 2 * beta_k - 3 * sigma * delta0

                if arm_attack not in a_tildes:
                    a_tilde_new = min(a_tilde, mu_k - 3 * beta_k - 3 * sigma * delta0)
                    a_tildes[arm_attack] = a_tilde_new
                else:
                    a_tilde_new = a_tildes[arm_attack]

                n_tilde = math.ceil((mu_i - l_hat) / (l_hat - a_tilde_new) * math.ceil(math.log(T) / delta0**2))
                
                # Optimize R_i 
                max_c = max(1, math.ceil(n_tilde / f))
                r_candidates = []

                for c in range(1, max_c + 1):
                    total_injected_samples = f * c
                    mu_tilde_i_c = ((arm_pulls[arm_attack] * mu_i) + total_injected_samples * a_tilde_new) / (arm_pulls[arm_attack] + total_injected_samples)
                    
                    numerator = mu_k - 2 * beta_k - mu_tilde_i_c
                    exponent = ((numerator / (3 * sigma)) ** 2) * (arm_pulls[arm_attack] + f * c)
                    
                    if exponent < 40:  # Prevent overflow
                        r_candidate = (1/c) * math.exp(exponent) - t - f
                        r_candidates.append(max(0, r_candidate))
                    else:
                        r_candidates.append(float('inf'))

                # Minimum R_i across all c values
                r_new = min(r_candidates) if r_candidates else 0
                
                #print(f"Round {t}: Arm {arm_attack} - R_i = {r_new:.2f} (candidates: {[f'{x:.2f}' for x in r_candidates[:5]]})")  # Show first 5 candidates

                injection_dict[arm_attack] = {
                    'remaining': int(n_tilde),
                    'r_value': r_new,
                    'a_tilde': a_tilde_new,
                    'next_injection': t 
                }
                
                print(f"Round {t}: Setup attack for arm {arm_attack}, n_tilde={n_tilde}, R_i={r_new}")
                attack_list.remove(arm_attack)
            
            # Periodic injections
            for key in list(injection_dict.keys()):
                injection_plan = injection_dict[key]
                
                if injection_plan['next_injection'] <= t and injection_plan['remaining'] > 0:
                    samples_to_inject = min(f, injection_plan['remaining'])
                    
                    print(f"Round {t}: Injecting {samples_to_inject} samples for arm {key}, {injection_plan['remaining']} remaining")

                    for _ in range(samples_to_inject):
                        attack_cost += abs(injection_plan['a_tilde'] - estimated_rewards[key])
                        recommender.update(key, injection_plan['a_tilde'])
                    
                    injection_plan['remaining'] -= samples_to_inject
                    injection_plan['next_injection'] = t + f + injection_plan['r_value']
                                        
                    if injection_plan['remaining'] <= 0:
                        del injection_dict[key]
            
            if arm != target_arm and arm_pulls[arm] >= math.ceil(math.log(T) / (delta0**2)) and arm not in attack_list:
                attack_list.append(arm)

            if arm == target_arm:
                target_pulls += 1

            target_pull_ratio = target_pulls / t
            target_pull_ratios.append(target_pull_ratio)
            
        recommender.update(arm, reward)

    return target_pulls, target_pull_ratios, attack_cost

def least_injection_attack_real(n_arms, target_arm, rho, T, reward_matrix, sigma=1, delta0=0.05, bounded=False):
    recommender = UCBRecommender(n_arms, rho)
    target_pulls = 0
    estimated_rewards = np.zeros(n_arms)
    arm_pulls = np.zeros(n_arms)
    target_pull_ratios = []
    attacked_list = []
    attack_cost = 0.0

    for t in range(1, T + n_arms + 1):
        arm = recommender.play()
        reward = get_reward_from_matrix(reward_matrix, arm)
        arm_pulls[arm] += 1
        estimated_rewards[arm] = (estimated_rewards[arm] * (arm_pulls[arm] - 1) + reward) / arm_pulls[arm]

        if t > n_arms:
            if arm not in attacked_list and arm != target_arm and arm_pulls[arm] >= math.log(T / (delta0**2)):
                attacked_list.append(arm)

                N_target = arm_pulls[target_arm]
                beta_val = beta(N_target, sigma, n_arms, delta0)
                mu_target = estimated_rewards[target_arm]
                mu_arm = estimated_rewards[arm]
                pulls_arm = recommender.pulls[arm]

                desired_avg = mu_target - 2 * beta_val - 3 * sigma
                fake_reward = desired_avg * (pulls_arm + 1) - mu_arm * pulls_arm

                if bounded:
                    fake_reward = np.clip(fake_reward, 0.0, 1.0)

                attack_cost += abs(fake_reward - reward)
                recommender.update(arm, fake_reward)

            if arm == target_arm:
                target_pulls += 1
            target_pull_ratio = target_pulls / t
            target_pull_ratios.append(target_pull_ratio)

        recommender.update(arm, reward)

    return target_pulls, target_pull_ratios, attack_cost

def experiment_simultaneous_bounded_injection(T=int(1e4), n_arms=10, rho=1.0, sigma=1.0, a_tilde=0.0, delta0=0.05, trials=10):
    all_ratios = []

    for _ in range(trials):
        means = np.random.rand(n_arms)
        std_devs = np.full(n_arms, sigma)
        target_arm = np.argmin(means)
        mu_k = means[target_arm]
        a_tilde = calculate_a_tilde(mu_k, n_arms, sigma, delta0)

        target_pulls, attack_trials, target_pull_ratios, attack_cost = simultaneous_bounded_injection_attack(
            n_arms, target_arm, rho, T, means, std_devs, a_tilde, delta0=delta0
        )
        all_ratios.append(target_pull_ratios)

    avg_ratios = np.mean(all_ratios, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, T + 1), avg_ratios, label="Average Target Arm Selection Ratio")
    plt.xlabel("Rounds")
    plt.ylabel("Target Arm Selection Ratio")
    plt.title(f"Average Target Arm Selection Ratio Over Time (SBI Attack)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Completed {trials} trials.")

def experiment_real_simultaneous_bounded_injection(T=int(1e4), n_arms=10, rho=1.0, a_tilde = 0.0, sigma=1.0, delta0=0.05, trials=10):
    all_ratios = []

    for _ in range(trials):
        reduced_matrix = np.load(os.path.join("..", "dataset", "movielens.npy"))
        selected_movie_indices = np.random.choice(reduced_matrix.shape[1], size=n_arms, replace=False)
        reduced_matrix = reduced_matrix[:, selected_movie_indices]
        movie_interactions = np.sum(reduced_matrix, axis=0)
        least_interacted_movie = np.argmin(movie_interactions)
        target_arm = least_interacted_movie

        target_pulls, attack_trials, target_pull_ratios, attack_cost = simultaneous_bounded_injection_attack_real(
            n_arms, target_arm, rho, T, reduced_matrix, a_tilde=a_tilde, sigma=sigma, delta0=delta0,
        )
        all_ratios.append(target_pull_ratios)

    avg_ratios = np.mean(all_ratios, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, T + 1), avg_ratios, label="Average Target Arm Selection Ratio")
    plt.xlabel("Rounds")
    plt.ylabel("Target Arm Selection Ratio")
    plt.title(f"Average Target Arm Selection Ratio Over Time (Simultaneous Bounded Injection Attack for Real Dataset) ")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_attack_cost_real(n_arms=10, rho=1.0, a_tilde=0.0, sigma=1.0, delta0=0.2, trials=10):
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

            target_pulls, attack_trials, target_pull_ratios, attack_cost = simultaneous_bounded_injection_attack_real(
                n_arms, target_arm, rho, T, reduced_matrix, a_tilde=a_tilde, sigma=sigma, delta0=delta0,
            )
            trial_costs.append(attack_cost)
        
        avg_costs.append(np.mean(trial_costs))

    plt.figure(figsize=(8, 5))
    plt.plot(T_values, avg_costs, marker='o')
    plt.xlabel("T (Rounds)")
    plt.ylabel("Average Total Attack Cost")
    plt.title(f"Total Attack Cost vs. T for Simultaneous Injection Attack (MovieLens Dataset)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_attack_cost_vs_delta0_real(n_arms=10, rho=1.0, T=int(1e4), a_tilde=0.0, sigma=1.0, trials=10):
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

            target_pulls, attack_trials, target_pull_ratios, attack_cost = simultaneous_bounded_injection_attack_real(
            n_arms, target_arm, rho, T, reduced_matrix, a_tilde=a_tilde, sigma=sigma, delta0=delta0,
        )

            trial_costs.append(attack_cost)

        avg_costs.append(np.mean(trial_costs))

    plt.figure(figsize=(8, 5))
    plt.plot(delta0_values, avg_costs, marker='o')
    plt.xlabel("δ₀ (Confidence Parameter)")
    plt.ylabel("Average Total Attack Cost")
    plt.title(f"Total Attack Cost vs. δ₀ for Simultaneous Bounded Injection Attack (MovieLens Dataset) T = {T}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def experiment_comparison_injection_real(T=int(1e5), n_arms=10, rho=1.0, sigma=1.0, delta0=0.2, trials=10):
    all_ratios_sbi = []
    all_ratios_pbi = []
    all_ratios_li = []

    for _ in range(trials):
        reduced_matrix = np.load(os.path.join("..", "dataset", "movielens.npy"))
        selected_movie_indices = np.random.choice(reduced_matrix.shape[1], size=n_arms, replace=False)
        reduced_matrix = reduced_matrix[:, selected_movie_indices]
        movie_interactions = np.sum(reduced_matrix, axis=0)
        target_arm = np.argmin(movie_interactions)

        target_pulls_sbi, _, target_pull_ratios_sbi, _ = simultaneous_bounded_injection_attack_real(
            n_arms, target_arm, rho, T, reduced_matrix, a_tilde=0.0, sigma=sigma, delta0=delta0
        )
        all_ratios_sbi.append(target_pull_ratios_sbi)

        target_pulls_pbi, target_pull_ratios_pbi, _ = periodic_injection_attack_real(
            n_arms, target_arm, rho, T, reduced_matrix, a_tilde=0.0, f=5, R=5000, sigma=sigma, delta0=delta0
        )
        all_ratios_pbi.append(target_pull_ratios_pbi)

        target_pulls_li, target_pull_ratios_li, _ = least_injection_attack_real(
            n_arms, target_arm, rho, T, reduced_matrix, sigma=sigma, delta0=delta0
        )
        all_ratios_li.append(target_pull_ratios_li)



    avg_ratios_sbi = np.mean(all_ratios_sbi, axis=0)
    avg_ratios_pbi = np.mean(all_ratios_pbi, axis=0)
    avg_ratios_li = np.mean(all_ratios_li, axis = 0)
    
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, T + 1), avg_ratios_sbi, label="Simultaneous Bounded Injection", color='blue', linestyle='dotted', marker='o', markersize=4)
    plt.plot(range(1, T + 1), avg_ratios_pbi, label="Periodic Bounded Injection", color='red', linestyle='--', marker='x', markersize=1)
    plt.plot(range(1, T + 1), avg_ratios_li, label="Least Injection", color='green', linestyle='-', marker='.', markersize=1)
    plt.tick_params(labelsize=27)
    plt.xlabel("Rounds", fontsize=30)
    plt.ylabel("Target Arm Selection Ratio", fontsize=30)
    plt.legend(fontsize=28)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_attack_cost_comparison(n_arms=10, rho=1.0, a_tilde=0.0, sigma=1.0, delta0=0.2, R=5000, f=5, trials=10):
    avg_costs_sbi = []
    avg_costs_pbi = []
    std_costs_sbi = []
    std_costs_pbi = []
    base_T_values = np.logspace(1, 7, num=10, dtype=int)
    custom_T_values = np.array([int(0.4e7), int(0.6e7), int(0.8e7)])
    T_values = np.unique(np.concatenate((base_T_values, custom_T_values)))

    for T in T_values:
        trial_costs_sbi = []
        trial_costs_pbi = []

        for _ in range(trials):
            reduced_matrix = np.load(os.path.join("..", "dataset", "movielens.npy"))
            selected_movie_indices = np.random.choice(reduced_matrix.shape[1], size=n_arms, replace=False)
            reduced_matrix = reduced_matrix[:, selected_movie_indices]

            movie_interactions = np.sum(reduced_matrix, axis=0)
            target_arm = np.argmin(movie_interactions)

            _, _, _, attack_cost_sbi = simultaneous_bounded_injection_attack_real(
                n_arms, target_arm, rho, T, reduced_matrix, a_tilde=a_tilde, sigma=sigma, delta0=delta0)
            trial_costs_sbi.append(attack_cost_sbi)

            _, _, attack_cost_pbi = periodic_injection_attack_real(
                n_arms, target_arm, rho, T, reduced_matrix, a_tilde=a_tilde, f=f, R=R, sigma=sigma, delta0=delta0)
            trial_costs_pbi.append(attack_cost_pbi)

        avg_costs_sbi.append(np.mean(trial_costs_sbi))
        avg_costs_pbi.append(np.mean(trial_costs_pbi))
        std_costs_sbi.append(np.std(trial_costs_sbi))
        std_costs_pbi.append(np.std(trial_costs_pbi))

    plt.figure(figsize=(12, 8))
    plt.errorbar(T_values, avg_costs_sbi, yerr=std_costs_sbi, marker='o', label='Simultaneous Bounded Injection', linestyle='dotted', color='blue', capsize=5)
    plt.errorbar(T_values, avg_costs_pbi, yerr=std_costs_pbi, marker='x', label='Periodic Bounded Injection', linestyle='--', color='red', capsize=5)
    plt.tick_params(labelsize=27)
    plt.xlabel("T", fontsize=30)
    plt.ylabel("Average Total Attack Cost", fontsize=30)
    plt.legend(fontsize=28)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_attack_cost_vs_delta0_comparison(n_arms=10, rho=1.0, T=int(1e6), a_tilde=0.0, sigma=1.0, R=5000, f=5, trials=10):
    avg_costs_sbi = []
    avg_costs_pbi = []
    std_costs_sbi = []
    std_costs_pbi = []
    delta0_values = np.linspace(0.1, 0.5, num=20)

    for delta0 in delta0_values:
        trial_costs_sbi = []
        trial_costs_pbi = []

        for _ in range(trials):
            reduced_matrix = np.load(os.path.join("..", "dataset", "movielens.npy"))
            selected_movie_indices = np.random.choice(reduced_matrix.shape[1], size=n_arms, replace=False)
            reduced_matrix = reduced_matrix[:, selected_movie_indices]

            movie_interactions = np.sum(reduced_matrix, axis=0)
            target_arm = np.argmin(movie_interactions)

            _, _, _, attack_cost_sbi = simultaneous_bounded_injection_attack_real(
                n_arms, target_arm, rho, T, reduced_matrix, a_tilde=a_tilde, sigma=sigma, delta0=delta0)
            trial_costs_sbi.append(attack_cost_sbi)

            _, _, attack_cost_pbi = periodic_injection_attack_real(
                n_arms, target_arm, rho, T, reduced_matrix, a_tilde=a_tilde, f=f, R=R, sigma=sigma, delta0=delta0)
            trial_costs_pbi.append(attack_cost_pbi)

        avg_costs_sbi.append(np.mean(trial_costs_sbi))
        avg_costs_pbi.append(np.mean(trial_costs_pbi))
        std_costs_sbi.append(np.std(trial_costs_sbi))
        std_costs_pbi.append(np.std(trial_costs_pbi))

    plt.figure(figsize=(12, 8))
    plt.errorbar(delta0_values, avg_costs_sbi, yerr=std_costs_sbi, marker='o', label='Simultaneous Bounded Injection', linestyle='dotted', color='blue', capsize=5)
    plt.errorbar(delta0_values, avg_costs_pbi, yerr=std_costs_pbi, marker='x', label='Periodic Bounded Injection', linestyle='--', color='red', capsize=5)
    plt.tick_params(labelsize=27)
    plt.xlabel("δ₀ (Confidence Parameter)", fontsize=30)
    plt.ylabel("Average Total Attack Cost", fontsize=30)
    plt.legend(fontsize=28)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # experiment_real_simultaneous_bounded_injection(a_tilde=0.0)
    # plot_attack_cost_real()
    # plot_attack_cost_vs_delta0_real()

    experiment_comparison_injection_real()
    # plot_attack_cost_vs_delta0_comparison()
    #plot_attack_cost_comparison()


