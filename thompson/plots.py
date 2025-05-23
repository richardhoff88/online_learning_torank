import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import math

sys.path.append(os.path.abspath(".."))
from attack import get_real_reward

def plot_attack_cost_vs_delta0(n_arms=10, rho=1.0, sigma=1.0, trials=10):
    delta0_values = np.linspace(0.1, 0.5, num=20)
    avg_costs_single = []
    avg_costs_sequential = []
    avg_costs_periodic = []
    std_costs_single = []
    std_costs_sequential = []
    std_costs_periodic = []
    
    for delta0 in delta0_values:
        trial_costs_single = []
        trial_costs_sequential = []
        trial_costs_periodic = []

        for _ in range(trials):
            means = np.random.rand(n_arms)
            std_devs = np.full(n_arms, sigma)
            target_arm = np.argmin(means)
            
            from thompson import Thompson_single
            thompson = Thompson_single(n_arms, T=10000, sigma=sigma)
            attack_cost = thompson.run()
            trial_costs_single.append(attack_cost)

            from thompson import Thompson_sequential
            thompson = Thompson_sequential(n_arms, T=10000, sigma=sigma)
            attack_cost = thompson.run()
            trial_costs_sequential.append(attack_cost)

            from thompson import Thompson_periodic
            thompson = Thompson_periodic(n_arms, T=10000, sigma=sigma)
            attack_cost = thompson.run()
            trial_costs_periodic.append(attack_cost)

        avg_costs_single.append(np.median(trial_costs_single))
        avg_costs_sequential.append(np.median(trial_costs_sequential))
        avg_costs_periodic.append(np.median(trial_costs_periodic))
        std_costs_single.append(np.std(trial_costs_single, ddof=1))
        std_costs_sequential.append(np.std(trial_costs_sequential, ddof=1))
        std_costs_periodic.append(np.std(trial_costs_periodic, ddof=1))

    plt.figure(figsize=(12, 8))
    x = np.arange(len(delta0_values))
    x_error = x[::2]
    
    plt.errorbar(delta0_values[x_error], avg_costs_single[::2], yerr=std_costs_single[::2], marker='o', 
            label='Single Injection', linestyle='-', color='blue', capsize=5)
    plt.errorbar(delta0_values[x_error], avg_costs_sequential[::2], yerr=std_costs_sequential[::2], marker='x', 
            label='Sequential Injection', linestyle='--', color='red', capsize=5)
    plt.errorbar(delta0_values[x_error], avg_costs_periodic[::2], yerr=std_costs_periodic[::2], marker='^', 
            label='Periodic Injection', linestyle='dotted', color='green', capsize=5)
    
    plt.plot(delta0_values, avg_costs_single, color='blue', linestyle='-')
    plt.plot(delta0_values, avg_costs_sequential, color='red', linestyle='--')
    plt.plot(delta0_values, avg_costs_periodic, color='green', linestyle='dotted')
    
    plt.tick_params(labelsize=27)
    plt.xlabel("Delta0", fontsize=30)
    plt.ylabel("Attack Cost", fontsize=30)
    plt.legend(fontsize=28)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_attack_cost_vs_T(n_arms=10, rho=1.0, sigma=1.0, delta0=1.0, trials=10):
    T_values = np.concatenate([
        np.logspace(1, 5, num=5, dtype=int),  
        np.linspace(400000, 1000000, num=15, dtype=int)  
    ])
    
    avg_costs_single = []
    avg_costs_sequential = []
    avg_costs_periodic = []
    std_costs_single = []
    std_costs_sequential = []
    std_costs_periodic = []
    
    for T in T_values:
        trial_costs_single = []
        trial_costs_sequential = []
        trial_costs_periodic = []

        for _ in range(trials):
            from thompson import Thompson_single
            thompson = Thompson_single(n_arms, T, sigma)
            attack_cost = thompson.run()
            trial_costs_single.append(attack_cost)

            from thompson import Thompson_sequential
            thompson = Thompson_sequential(n_arms, T, sigma)
            attack_cost = thompson.run()
            trial_costs_sequential.append(attack_cost)

            from thompson import Thompson_periodic
            thompson = Thompson_periodic(n_arms, T, sigma)
            attack_cost = thompson.run()
            trial_costs_periodic.append(attack_cost)

        avg_costs_single.append(np.mean(trial_costs_single))
        avg_costs_sequential.append(np.mean(trial_costs_sequential))
        avg_costs_periodic.append(np.mean(trial_costs_periodic))
        std_costs_single.append(np.std(trial_costs_single))
        std_costs_sequential.append(np.std(trial_costs_sequential))
        std_costs_periodic.append(np.std(trial_costs_periodic))

    plt.figure(figsize=(12, 8))
    x = np.arange(len(T_values))
    x_error = x[::2]
    
    plt.errorbar(T_values[x_error], avg_costs_single[::2], yerr=std_costs_single[::2], marker='o', 
            label='Single Injection', linestyle='-', color='blue', capsize=5)
    plt.errorbar(T_values[x_error], avg_costs_sequential[::2], yerr=std_costs_sequential[::2], marker='x', 
            label='Sequential Injection', linestyle='--', color='red', capsize=5)
    plt.errorbar(T_values[x_error], avg_costs_periodic[::2], yerr=std_costs_periodic[::2], marker='^', 
            label='Periodic Injection', linestyle='dotted', color='green', capsize=5)
    
    plt.plot(T_values, avg_costs_single, color='blue', linestyle='-')
    plt.plot(T_values, avg_costs_sequential, color='red', linestyle='--')
    plt.plot(T_values, avg_costs_periodic, color='green', linestyle='dotted')
    
    plt.tick_params(labelsize=27)
    plt.xlabel("T", fontsize=30)
    plt.ylabel("Attack Cost", fontsize=30)
    plt.legend(fontsize=28)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_target_arm_ratio_vs_T(n_arms=10, rho=1.0, sigma=1.0, delta0=0.2, trials=10):
    T_values = np.logspace(1, 6, num=10, dtype=int)
    
    avg_ratios_single = []
    avg_ratios_sequential = []
    avg_ratios_periodic = []
    
    for T in T_values:
        trial_ratios_single = []
        trial_ratios_sequential = []
        trial_ratios_periodic = []

        for _ in range(trials):
            means = np.random.rand(n_arms)
            std_devs = np.full(n_arms, sigma)
            target_arm = np.argmin(means)

            from thompson import Thompson_single
            thompson = Thompson_single(n_arms, T, sigma)
            ratios = thompson.run()
            trial_ratios_single.append(ratios[-1])

            from thompson import Thompson_sequential
            thompson = Thompson_sequential(n_arms, T, sigma)
            ratios = thompson.run()
            trial_ratios_sequential.append(ratios[-1])
            from thompson import Thompson_periodic
            thompson = Thompson_periodic(n_arms, T, sigma)
            ratios = thompson.run()
            trial_ratios_periodic.append(ratios[-1])

        avg_ratios_single.append(np.mean(trial_ratios_single))
        avg_ratios_sequential.append(np.mean(trial_ratios_sequential))
        avg_ratios_periodic.append(np.mean(trial_ratios_periodic))

    plt.figure(figsize=(12, 8))
    plt.plot(T_values, avg_ratios_single, marker='o', 
            label='Single Injection', linestyle='-', color='blue')
    plt.plot(T_values, avg_ratios_sequential, marker='x', 
            label='Sequential Injection', linestyle='--', color='red')
    plt.plot(T_values, avg_ratios_periodic, marker='^', 
            label='Periodic Injection', linestyle='dotted', color='green')
    
    plt.tick_params(labelsize=27)
    plt.xlabel("Rounds", fontsize=30)
    plt.ylabel("Target Arm Ratio", fontsize=30)
    plt.legend(fontsize=28)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_target_arm_ratio(n_arms=10, rho=1.0, sigma=1.0, delta0=0.2, T=10000, trials=10):
    all_ratios_single = []
    all_ratios_sequential = []
    all_ratios_periodic = []

    for _ in range(trials):
        means = np.random.rand(n_arms)
        std_devs = np.full(n_arms, sigma)
        target_arm = np.argmin(means)

        from thompson import Thompson_single
        thompson = Thompson_single(n_arms, T, sigma)
        ratios = thompson.run()
        all_ratios_single.append(ratios)

        from thompson import Thompson_sequential
        thompson = Thompson_sequential(n_arms, T, sigma)
        ratios = thompson.run()
        all_ratios_sequential.append(ratios)

        from thompson import Thompson_periodic
        thompson = Thompson_periodic(n_arms, T, sigma)
        ratios = thompson.run()
        all_ratios_periodic.append(ratios)

    avg_ratios_single = np.mean(all_ratios_single, axis=0)
    avg_ratios_sequential = np.mean(all_ratios_sequential, axis=0)
    avg_ratios_periodic = np.mean(all_ratios_periodic, axis=0)
    std_ratios_single = np.std(all_ratios_single, axis=0)
    std_ratios_sequential = np.std(all_ratios_sequential, axis=0)
    std_ratios_periodic = np.std(all_ratios_periodic, axis=0)
    
    plt.figure(figsize=(12, 8))
    x = np.arange(1, T + 1)
    x_error = x[::100]
    
    plt.errorbar(x_error, avg_ratios_single[::100], yerr=std_ratios_single[::100],
            label='Single Injection', color='blue', linestyle='-', marker='o', markersize=4, capsize=2)
    plt.errorbar(x_error, avg_ratios_sequential[::100], yerr=std_ratios_sequential[::100],
            label='Sequential Injection', color='red', linestyle='--', marker='x', markersize=1, capsize=2)
    plt.errorbar(x_error, avg_ratios_periodic[::100], yerr=std_ratios_periodic[::100],
            label='Periodic Injection', color='green', linestyle='dotted', marker='^', markersize=2, capsize=2)
    
    plt.plot(range(1, T + 1), avg_ratios_single, color='blue', linestyle='-')
    plt.plot(range(1, T + 1), avg_ratios_sequential, color='red', linestyle='--')
    plt.plot(range(1, T + 1), avg_ratios_periodic, color='green', linestyle='dotted')
    
    plt.tick_params(labelsize=27)
    plt.xlabel("Rounds", fontsize=30)
    plt.ylabel("Target Arm Ratio", fontsize=30)
    plt.legend(fontsize=28)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def beta(N, sigma, n_arms, delta):
    N = max(N, 1)
    log_argument = max((math.pi**2 * N**2 * n_arms) / (3 * delta), 1)
    return math.sqrt((2 * sigma**2 / N) * math.log(log_argument))



if __name__ == "__main__":
    plot_attack_cost_vs_delta0()
    #plot_attack_cost_vs_T()
    # plot_target_arm_ratio()