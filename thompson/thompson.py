from typing import List
from matplotlib import pyplot as plt
import importlib
import math
import numpy as np
import os
import time
import warnings

K = 10
T = 100000
delta0 = 1.0
delta = 0.05
ATTACK_SIGMA = 0.5
REWARD_SIGMA = 0.05


attacker_single = importlib.import_module("single_injection_ts").attacker(K=K, T=T, delta=delta, delta0=1.0, sigma=ATTACK_SIGMA)
attacker_sequential = importlib.import_module("sequential_injection_ts").attacker(K=K, T=T, delta=delta, delta0=1.0, sigma=ATTACK_SIGMA)
attacker_periodic = importlib.import_module("periodic_injection_ts").attacker(K=K, T=T, delta=delta, delta0=1.0, sigma=ATTACK_SIGMA)
attacker_heuristic = None

import real_reward
from real_reward import get_reward, init_reward


ARTIFACT_ROOT = os.path.dirname(os.path.abspath(__file__))


def beta(N: int, sigma: float, n_arms: int, delta: float) -> float:
    N = max(N, 1)
    log_argument = max((math.pi ** 2) * n_arms * (N ** 2) / (3 * delta), 1)
    return math.sqrt(2 * sigma * sigma / N * math.log(log_argument))


def ts_kappa(n_arms: int, delta: float) -> float:
    return math.sqrt(8 * math.log((math.pi ** 2) * n_arms / (3 * delta)))


def attack_round(T: int, delta0: float) -> int:
    return int(math.log(max(T, 2)) / (delta0 * delta0)) + 1


def format_duration(seconds):
    seconds = max(int(seconds), 0)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def print_progress(label, completed, total, start_time):
    elapsed = time.time() - start_time
    avg_time = elapsed / max(completed, 1)
    remaining = avg_time * max(total - completed, 0)
    print(
        f"{label}: {completed}/{total} complete | "
        f"elapsed {format_duration(elapsed)} | ETA {format_duration(remaining)}",
        flush=True,
    )


def make_run_id(prefix):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def save_plot_artifacts(run_id, data, figure=None, data_dir="data", plots_dir="plots"):
    data_dir = os.path.join(ARTIFACT_ROOT, data_dir)
    plots_dir = os.path.join(ARTIFACT_ROOT, plots_dir)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    data_path = os.path.join(data_dir, f"{run_id}.npz")
    plot_path = os.path.join(plots_dir, f"{run_id}.png")
    np.savez(data_path, **data)
    if figure is not None:
        figure.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot data to {data_path}", flush=True)
    if figure is not None:
        print(f"Saved plot figure to {plot_path}", flush=True)
    return data_path, plot_path


def safe_nanmean(values):
    values = np.asarray(values, dtype=float)
    if np.all(np.isnan(values)):
        return np.nan
    return np.nanmean(values)


def warn_if_all_invalid(curve_values, label, plot_name):
    if np.all(np.isnan(curve_values)):
        warnings.warn(
            f"{label} is undefined for all sampled points in {plot_name}; "
            "the TS upper bound is vacuous under the current parameters.",
            RuntimeWarning,
        )


def current_means(n_arms: int):
    return np.array([real_reward.real_mean(i) for i in range(n_arms)], dtype=float)


def reset_attackers(K, T, delta0=1.0, sigma=ATTACK_SIGMA, delta=delta):
    global attacker_single, attacker_sequential, attacker_periodic, attacker_heuristic
    attacker_single = importlib.import_module("single_injection_ts").attacker(
        K=K, T=T, delta=delta, delta0=delta0, sigma=sigma)
    attacker_sequential = importlib.import_module("sequential_injection_ts").attacker(
        K=K, T=T, delta=delta, delta0=delta0, sigma=sigma)
    attacker_periodic = importlib.import_module("periodic_injection_ts").attacker(
        K=K, T=T, delta=delta, delta0=delta0, sigma=sigma)
    attacker_heuristic = HeuristicTsAttacker(K=K, T=T, delta=delta, delta0=delta0, sigma=sigma)


def ts_single_theoretical_cost_upper_bound(T, means, target_arm, n_arms, sigma, delta0, delta=delta):
    n_i = attack_round(T, delta0)
    target_lower_bound = means[target_arm] - 2 * beta(1, sigma, n_arms, delta) - ts_kappa(n_arms, delta)
    target_lower_bound -= 4 * math.sqrt(n_i) * delta0
    total_cost = 0.0

    for arm, mean in enumerate(means):
        if arm == target_arm:
            continue
        fake_reward = (n_i + 1) * target_lower_bound - n_i * mean
        total_cost += abs(fake_reward - mean)

    return total_cost


def ts_sequential_theoretical_cost_upper_bound(T, means, target_arm, n_arms, sigma, delta0,
                                               lower_bound=-26.66, delta=delta):
    n_i = attack_round(T, delta0)
    ell = means[target_arm] - 2 * beta(1, sigma, n_arms, delta) - ts_kappa(n_arms, delta)
    ell -= 4 * math.sqrt(n_i) * delta0
    denominator = ell - lower_bound
    if denominator <= 0:
        return np.nan

    total_cost = 0.0
    log_term = math.log(max(T, 2)) / (delta0 ** 2)
    for arm, mean in enumerate(means):
        if arm == target_arm:
            continue
        fake_samples = max(0.0, (mean - ell) / denominator * log_term)
        total_cost += fake_samples * abs(mean - lower_bound)

    return total_cost


def ts_periodic_theoretical_cost_upper_bound(T, means, target_arm, n_arms, sigma, delta0,
                                             lower_bound=-36.66, delta=delta):
    return ts_sequential_theoretical_cost_upper_bound(
        T, means, target_arm, n_arms, sigma, delta0, lower_bound=lower_bound, delta=delta)


class HeuristicTsAttacker:
    def __init__(self, K: int, T: int, delta: float, delta0: float, sigma: float, lower_bound=-26.66):
        self.attack_round = attack_round(T, delta0)
        self.K = K
        self.T = T
        self.delta = delta
        self.delta0 = delta0
        self.sigma = sigma
        self.n = [1] * K
        self.empirical_means = [0.0] * K
        self.lower_bound = lower_bound
        self.time_slot = K
        self.attack_cost = 0
        self.remaining = [0] * K
        self.has_planned = [False] * K

    def update(self, k: int, reward: float):
        self.n[k] += 1
        self.empirical_means[k] = ((self.n[k] - 1) * self.empirical_means[k] + reward) / self.n[k]

    def beta(self, N: int) -> float:
        return beta(N, self.sigma, self.K, self.delta)

    def ell(self, i: int):
        return self.empirical_means[self.K-1] - 2 * self.beta(self.n[self.K-1]) - ts_kappa(self.K, self.delta) - 4 * math.sqrt(self.n[i]) * self.delta0

    def maybe_plan(self, i: int):
        if self.has_planned[i] or self.n[i] != self.attack_round:
            return
        self.has_planned[i] = True
        ell = self.ell(i)
        denominator = ell - self.lower_bound
        if denominator <= 0:
            return
        fake_samples = int((self.empirical_means[i] - ell) / denominator * math.log(self.T) / (self.delta0 ** 2)) + 1
        self.remaining[i] = max(fake_samples, 0)

    def feedback(self):
        self.time_slot += 1
        for i in range(self.K - 1):
            self.maybe_plan(i)

        for i in range(self.K - 1):
            if self.remaining[i] <= 0:
                continue
            remaining_slots = max(self.T - self.time_slot + 1, 1)
            inject_probability = min(1.0, self.remaining[i] / remaining_slots)
            force_count = max(0, self.remaining[i] - remaining_slots + 1)
            if force_count > 0 or np.random.random() < inject_probability:
                ret = self.lower_bound
                self.remaining[i] -= 1
                self.update(i, ret)
                self.attack_cost += real_reward.real_mean(i) - ret
                return i, ret

        sampled_means = [np.random.normal(loc=self.empirical_means[k], scale=1 / np.sqrt(self.n[k])) for k in range(self.K)]
        i = np.argmax(sampled_means)
        r = get_reward(i, self.sigma)
        self.update(i, r)
        return i, r

class Thompson_single:

    def __init__(self, K: int, T: int, sigma: float = 0.05):
        self.K = K
        self.T = T
        self.n = [0] * K
        self.empirical_means = [0.0] * K
        self.sigma = sigma

    def update(self, k: int, reward: float):
        self.n[k] += 1
        self.empirical_means[k] = ((self.n[k] - 1) * self.empirical_means[k] + reward) / self.n[k]

    def run(self) -> List[float]:
        ratio = [0] * self.T
        for t in range(1, self.K+1):
            k = t - 1
            reward = get_reward(k, self.sigma)
            self.update(k, reward)
            ratio[t-1] = self.n[self.K-1] / (t + 1)
        for t in range(self.K+1, self.T+1):
            k, r = attacker_single.feedback()
            self.update(k, r)
            ratio[t-1] = self.n[self.K-1] / (t + 1)
        return ratio

class Thompson_sequential:

    def __init__(self, K: int, T: int, sigma: float = 0.05):
        self.K = K
        self.T = T
        self.n = [0] * K
        self.empirical_means = [0.0] * K
        self.sigma = sigma

    def update(self, k: int, reward: float):
        self.n[k] += 1
        self.empirical_means[k] = ((self.n[k] - 1) * self.empirical_means[k] + reward) / self.n[k]

    def run(self) -> List[float]:
        ratio = [0] * self.T
        for t in range(1, self.K+1):
            k = t - 1
            reward = get_reward(k, self.sigma)
            self.update(k, reward)
            ratio[t-1] = self.n[self.K-1] / (t + 1)
        for t in range(self.K+1, self.T+1):
            k, r = attacker_sequential.feedback()
            self.update(k, r)
            ratio[t-1] = self.n[self.K-1] / (t + 1)
        return ratio


class Thompson_periodic:

    def __init__(self, K: int, T: int, sigma: float = 0.05):
        self.K = K
        self.T = T
        self.n = [0] * K
        self.empirical_means = [0.0] * K
        self.sigma = sigma

    def update(self, k: int, reward: float):
        self.n[k] += 1
        self.empirical_means[k] = ((self.n[k] - 1) * self.empirical_means[k] + reward) / self.n[k]

    def run(self) -> List[float]:
        ratio = [0] * self.T
        for t in range(1, self.K+1):
            k = t - 1
            reward = get_reward(k, self.sigma)
            self.update(k, reward)
            ratio[t-1] = self.n[self.K-1] / (t + 1)
        for t in range(self.K+1, self.T+1):
            k, r = attacker_periodic.feedback()
            self.update(k, r)
            ratio[t-1] = self.n[self.K-1] / (t + 1)
        return ratio


class Thompson_heuristic:

    def __init__(self, K: int, T: int, sigma: float = REWARD_SIGMA):
        self.K = K
        self.T = T
        self.n = [0] * K
        self.empirical_means = [0.0] * K
        self.sigma = sigma

    def update(self, k: int, reward: float):
        self.n[k] += 1
        self.empirical_means[k] = ((self.n[k] - 1) * self.empirical_means[k] + reward) / self.n[k]

    def run(self) -> List[float]:
        ratio = [0] * self.T
        for t in range(1, self.K+1):
            k = t - 1
            reward = get_reward(k, self.sigma)
            self.update(k, reward)
            ratio[t-1] = self.n[self.K-1] / (t + 1)
        for t in range(self.K+1, self.T+1):
            k, r = attacker_heuristic.feedback()
            self.update(k, r)
            ratio[t-1] = self.n[self.K-1] / (t + 1)
        return ratio


def run_ts_attack(attack_name, K, T, delta0=1.0, reward_sigma=REWARD_SIGMA,
                  attack_sigma=ATTACK_SIGMA, delta=delta):
    reset_attackers(K, T, delta0=delta0, sigma=attack_sigma, delta=delta)
    if attack_name == "single":
        ratios = Thompson_single(K, T, reward_sigma).run()
        attack_cost = attacker_single.attack_cost
    elif attack_name == "sequential":
        ratios = Thompson_sequential(K, T, reward_sigma).run()
        attack_cost = attacker_sequential.attack_cost
    elif attack_name == "periodic":
        ratios = Thompson_periodic(K, T, reward_sigma).run()
        attack_cost = attacker_periodic.attack_cost
    elif attack_name == "heuristic":
        ratios = Thompson_heuristic(K, T, reward_sigma).run()
        attack_cost = attacker_heuristic.attack_cost
    else:
        raise ValueError(f"Unknown TS attack name: {attack_name}")
    return np.asarray(ratios, dtype=float), attack_cost


def plot_attack_cost_comparison(K=10, delta0=1.0, trials=10, T_values=None, show_theory=True,
                                show_single_theory=True, show_progress=True, save_outputs=True,
                                reward_sigma=REWARD_SIGMA, attack_sigma=ATTACK_SIGMA, delta=delta):
    if T_values is None:
        base_T_values = np.logspace(1, 7, num=10, dtype=int)
        custom_T_values = np.array([int(0.4e7), int(0.6e7), int(0.8e7)])
        T_values = np.unique(np.concatenate((base_T_values, custom_T_values)))
    else:
        T_values = np.array(T_values, dtype=int)

    attack_names = ["single", "sequential", "periodic", "heuristic"]
    avg_costs = {name: [] for name in attack_names}
    std_costs = {name: [] for name in attack_names}
    avg_theory_single = []
    avg_theory_sequential = []
    avg_theory_periodic = []
    progress_start = time.time()

    for idx, horizon in enumerate(T_values, start=1):
        trial_costs = {name: [] for name in attack_names}
        trial_theory_single = []
        trial_theory_sequential = []
        trial_theory_periodic = []

        for _ in range(trials):
            means = current_means(K)
            target_arm = K - 1
            for name in attack_names:
                _, attack_cost = run_ts_attack(
                    name, K, horizon, delta0=delta0, reward_sigma=reward_sigma,
                    attack_sigma=attack_sigma, delta=delta)
                trial_costs[name].append(attack_cost)

            trial_theory_single.append(
                ts_single_theoretical_cost_upper_bound(horizon, means, target_arm, K, attack_sigma, delta0, delta=delta))
            trial_theory_sequential.append(
                ts_sequential_theoretical_cost_upper_bound(horizon, means, target_arm, K, attack_sigma, delta0, delta=delta))
            trial_theory_periodic.append(
                ts_periodic_theoretical_cost_upper_bound(horizon, means, target_arm, K, attack_sigma, delta0, delta=delta))

        for name in attack_names:
            avg_costs[name].append(np.mean(trial_costs[name]))
            std_costs[name].append(np.std(trial_costs[name]))
        avg_theory_single.append(safe_nanmean(trial_theory_single))
        avg_theory_sequential.append(safe_nanmean(trial_theory_sequential))
        avg_theory_periodic.append(safe_nanmean(trial_theory_periodic))
        if show_progress:
            print_progress("plot_attack_cost_comparison_ts", idx, len(T_values), progress_start)

    for name in attack_names:
        avg_costs[name] = np.array(avg_costs[name])
        std_costs[name] = np.array(std_costs[name])
    avg_theory_single = np.array(avg_theory_single)
    avg_theory_sequential = np.array(avg_theory_sequential)
    avg_theory_periodic = np.array(avg_theory_periodic)

    fig = plt.figure(figsize=(12, 8))
    plot_specs = [
        ("single", "Least Injection", "blue", "o", "dotted", 0.2),
        ("sequential", "Simultaneous Bounded Injection", "green", "x", "--", 0.2),
        ("periodic", "Periodic Bounded Injection", "red", "s", "-", 0.2),
        ("heuristic", "Heuristic Baseline", "purple", "^", "-.", 0.2),
    ]
    for name, label, color, marker, linestyle, alpha in plot_specs:
        plt.plot(T_values, avg_costs[name], label=label, color=color, linestyle=linestyle, marker=marker, linewidth=2)
        plt.fill_between(T_values, avg_costs[name] - std_costs[name], avg_costs[name] + std_costs[name],
                         color=color, alpha=alpha)

    warn_if_all_invalid(avg_theory_single, "Single Injection theoretical upper bound", "plot_attack_cost_comparison_ts")
    warn_if_all_invalid(avg_theory_sequential, "Sequential theoretical upper bound", "plot_attack_cost_comparison_ts")
    warn_if_all_invalid(avg_theory_periodic, "Periodic theoretical upper bound", "plot_attack_cost_comparison_ts")
    if show_theory and show_single_theory:
        single_mask = np.isfinite(avg_theory_single)
        plt.plot(T_values[single_mask], avg_theory_single[single_mask],
                 label="LI Theoretical Upper Bound Cost", color="darkblue", linestyle=":", linewidth=2)
    if show_theory:
        sequential_mask = np.isfinite(avg_theory_sequential)
        plt.plot(T_values[sequential_mask], avg_theory_sequential[sequential_mask],
                 label="SBI/PBI Theoretical Upper Bound Cost", color="darkgreen", linestyle=":", linewidth=2)
  
    plt.tick_params(labelsize=27)
    plt.xlabel("T", fontsize=30)
    plt.ylabel("Average Total Attack Cost", fontsize=30)
    plt.grid(True)
    plt.legend(fontsize=16)
    plt.tight_layout()
    if save_outputs:
        run_id = make_run_id("ts_attack_cost_vs_T")
        save_plot_artifacts(
            run_id,
            {
                "T_values": T_values,
                "avg_costs_single": avg_costs["single"],
                "std_costs_single": std_costs["single"],
                "avg_costs_sequential": avg_costs["sequential"],
                "std_costs_sequential": std_costs["sequential"],
                "avg_costs_periodic": avg_costs["periodic"],
                "std_costs_periodic": std_costs["periodic"],
                "avg_costs_heuristic": avg_costs["heuristic"],
                "std_costs_heuristic": std_costs["heuristic"],
                "avg_theory_single": avg_theory_single,
                "avg_theory_sequential": avg_theory_sequential,
                "avg_theory_periodic": avg_theory_periodic,
                "K": np.array(K),
                "delta0": np.array(delta0),
                "delta": np.array(delta),
                "reward_sigma": np.array(reward_sigma),
                "attack_sigma": np.array(attack_sigma),
                "trials": np.array(trials),
            },
            fig,
        )
    plt.show()


def plot_attack_cost_vs_delta0_comparison(K=10, T=int(1e5), trials=10, delta0_values=None, show_theory=True,
                                          show_single_theory=True, show_progress=True, save_outputs=True,
                                          reward_sigma=REWARD_SIGMA, attack_sigma=ATTACK_SIGMA, delta=delta):
    if delta0_values is None:
        delta0_values = np.linspace(0.1, 0.5, num=20)
    else:
        delta0_values = np.array(delta0_values, dtype=float)

    attack_names = ["single", "sequential", "periodic", "heuristic"]
    avg_costs = {name: [] for name in attack_names}
    std_costs = {name: [] for name in attack_names}
    avg_theory_single = []
    avg_theory_sequential = []
    avg_theory_periodic = []
    progress_start = time.time()

    for idx, delta0_value in enumerate(delta0_values, start=1):
        trial_costs = {name: [] for name in attack_names}
        trial_theory_single = []
        trial_theory_sequential = []
        trial_theory_periodic = []

        for _ in range(trials):
            means = current_means(K)
            target_arm = K - 1
            for name in attack_names:
                _, attack_cost = run_ts_attack(
                    name, K, T, delta0=delta0_value, reward_sigma=reward_sigma,
                    attack_sigma=attack_sigma, delta=delta)
                trial_costs[name].append(attack_cost)

            trial_theory_single.append(
                ts_single_theoretical_cost_upper_bound(T, means, target_arm, K, attack_sigma, delta0_value, delta=delta))
            trial_theory_sequential.append(
                ts_sequential_theoretical_cost_upper_bound(T, means, target_arm, K, attack_sigma, delta0_value, delta=delta))
            trial_theory_periodic.append(
                ts_periodic_theoretical_cost_upper_bound(T, means, target_arm, K, attack_sigma, delta0_value, delta=delta))

        for name in attack_names:
            avg_costs[name].append(np.mean(trial_costs[name]))
            std_costs[name].append(np.std(trial_costs[name]))
        avg_theory_single.append(safe_nanmean(trial_theory_single))
        avg_theory_sequential.append(safe_nanmean(trial_theory_sequential))
        avg_theory_periodic.append(safe_nanmean(trial_theory_periodic))
        if show_progress:
            print_progress("plot_attack_cost_vs_delta0_comparison_ts", idx, len(delta0_values), progress_start)

    for name in attack_names:
        avg_costs[name] = np.array(avg_costs[name])
        std_costs[name] = np.array(std_costs[name])
    avg_theory_single = np.array(avg_theory_single)
    avg_theory_sequential = np.array(avg_theory_sequential)
    avg_theory_periodic = np.array(avg_theory_periodic)

    fig = plt.figure(figsize=(12, 8))
    plot_specs = [
        ("single", "Least Injection", "blue", "o", "dotted", 0.2),
        ("sequential", "Simultaneous Bounded Injection", "green", "x", "--", 0.2),
        ("periodic", "Periodic Bounded Injection", "red", "s", "-", 0.2),
        ("heuristic", "Heuristic Baseline", "purple", "^", "-.", 0.2),
    ]
    for name, label, color, marker, linestyle, alpha in plot_specs:
        plt.plot(delta0_values, avg_costs[name], label=label, color=color, linestyle=linestyle, marker=marker, linewidth=2)
        plt.fill_between(delta0_values, avg_costs[name] - std_costs[name], avg_costs[name] + std_costs[name],
                         color=color, alpha=alpha)

    warn_if_all_invalid(avg_theory_single, "Single Injection theoretical upper bound", "plot_attack_cost_vs_delta0_comparison_ts")
    warn_if_all_invalid(avg_theory_sequential, "Sequential theoretical upper bound", "plot_attack_cost_vs_delta0_comparison_ts")
    warn_if_all_invalid(avg_theory_periodic, "Periodic theoretical upper bound", "plot_attack_cost_vs_delta0_comparison_ts")
    if show_theory and show_single_theory:
        single_mask = np.isfinite(avg_theory_single)
        plt.plot(delta0_values[single_mask], avg_theory_single[single_mask],
                 label="LI Theoretical Upper Bound Cost", color="darkblue", linestyle=":", linewidth=2)
    if show_theory:
        sequential_mask = np.isfinite(avg_theory_sequential)
        periodic_mask = np.isfinite(avg_theory_periodic)
        plt.plot(delta0_values[sequential_mask], avg_theory_sequential[sequential_mask],
                 label="SBI/PBI Theoretical Upper Bound Cost", color="darkgreen", linestyle=":", linewidth=2)

    plt.tick_params(labelsize=27)
    plt.xlabel("δ₀ (Confidence Parameter)", fontsize=30)
    plt.ylabel("Average Total Attack Cost", fontsize=30)
    plt.grid(True)
    plt.legend(fontsize=16)
    plt.tight_layout()
    if save_outputs:
        run_id = make_run_id("ts_attack_cost_vs_delta0")
        save_plot_artifacts(
            run_id,
            {
                "delta0_values": delta0_values,
                "avg_costs_single": avg_costs["single"],
                "std_costs_single": std_costs["single"],
                "avg_costs_sequential": avg_costs["sequential"],
                "std_costs_sequential": std_costs["sequential"],
                "avg_costs_periodic": avg_costs["periodic"],
                "std_costs_periodic": std_costs["periodic"],
                "avg_costs_heuristic": avg_costs["heuristic"],
                "std_costs_heuristic": std_costs["heuristic"],
                "avg_theory_single": avg_theory_single,
                "avg_theory_sequential": avg_theory_sequential,
                "avg_theory_periodic": avg_theory_periodic,
                "K": np.array(K),
                "T": np.array(T),
                "delta": np.array(delta),
                "reward_sigma": np.array(reward_sigma),
                "attack_sigma": np.array(attack_sigma),
                "trials": np.array(trials),
            },
            fig,
        )
    plt.show()


def plot_target_ratio_comparison(K=10, T=int(1e5), delta0=1.0, trials=10, show_progress=True,
                                 save_outputs=True, reward_sigma=REWARD_SIGMA,
                                 attack_sigma=ATTACK_SIGMA, delta=delta):
    attack_names = ["single", "sequential", "periodic", "heuristic"]
    all_ratios = {name: [] for name in attack_names}
    progress_start = time.time()

    for trial_idx in range(1, trials + 1):
        for name in attack_names:
            ratios, _ = run_ts_attack(
                name, K, T, delta0=delta0, reward_sigma=reward_sigma,
                attack_sigma=attack_sigma, delta=delta)
            all_ratios[name].append(ratios)
        if show_progress:
            print_progress("plot_target_ratio_comparison_ts", trial_idx, trials, progress_start)

    avg_ratios = {}
    std_ratios = {}
    for name in attack_names:
        ratio_array = np.asarray(all_ratios[name], dtype=float)
        avg_ratios[name] = np.mean(ratio_array, axis=0)
        std_ratios[name] = np.std(ratio_array, axis=0)

    x = np.arange(1, T + 1)
    fig = plt.figure(figsize=(12, 8))
    plot_specs = [
        ("single", "Least Injection", "blue", "o", "dotted", 3000, 0.3),
        ("sequential", "Simultaneous Bounded Injection", "green", "x", "--", 3000, 0.3),
        ("periodic", "Periodic Bounded Injection", "red", "s", "-", 3000, 0.3),
        ("heuristic", "Heuristic Baseline", "purple", "^", "-.", 3000, 0.2),
    ]
    for name, label, color, marker, linestyle, markevery, alpha in plot_specs:
        plt.plot(x, avg_ratios[name], label=label, color=color, linestyle=linestyle,
                 marker=marker, linewidth=2, markevery=markevery)
        plt.fill_between(x, avg_ratios[name] - std_ratios[name], avg_ratios[name] + std_ratios[name],
                         color=color, alpha=alpha)

    plt.tick_params(labelsize=27)
    plt.xlabel("Rounds", fontsize=30)
    plt.ylabel("Target Arm Selection Ratio", fontsize=30)
    plt.grid(True)
    plt.legend(fontsize=28)
    plt.tight_layout()
    if save_outputs:
        run_id = make_run_id("ts_target_ratio_comparison")
        save_plot_artifacts(
            run_id,
            {
                "x": x,
                "avg_ratios_single": avg_ratios["single"],
                "std_ratios_single": std_ratios["single"],
                "avg_ratios_sequential": avg_ratios["sequential"],
                "std_ratios_sequential": std_ratios["sequential"],
                "avg_ratios_periodic": avg_ratios["periodic"],
                "std_ratios_periodic": std_ratios["periodic"],
                "avg_ratios_heuristic": avg_ratios["heuristic"],
                "std_ratios_heuristic": std_ratios["heuristic"],
                "K": np.array(K),
                "T": np.array(T),
                "delta0": np.array(delta0),
                "delta": np.array(delta),
                "reward_sigma": np.array(reward_sigma),
                "attack_sigma": np.array(attack_sigma),
                "trials": np.array(trials),
            },
            fig,
        )
    plt.show()

def latest_cached_data(prefix, data_dirs=("data",)):
    candidates = []
    for data_dir in data_dirs:
        if not os.path.isabs(data_dir):
            data_dir = os.path.join(ARTIFACT_ROOT, data_dir)
        if not os.path.isdir(data_dir):
            continue
        for filename in os.listdir(data_dir):
            if filename.startswith(prefix) and filename.endswith(".npz"):
                candidates.append(os.path.join(data_dir, filename))
    if not candidates:
        raise FileNotFoundError(f"No cached {prefix} data found in {data_dirs}")
    return max(candidates, key=os.path.getmtime)


def save_cached_plot(fig, prefix, show_theory=True, plots_dir="plots"):
    if not os.path.isabs(plots_dir):
        plots_dir = os.path.join(ARTIFACT_ROOT, plots_dir)
    os.makedirs(plots_dir, exist_ok=True)
    suffix = "with_theory" if show_theory else "empirical_only"
    plot_path = os.path.join(plots_dir, f"{make_run_id(prefix)}_{suffix}.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved cached-data plot to {plot_path}", flush=True)
    return plot_path

# def plot_cached_attack_cost_comparison(data_path=None, show_theory=True, show_single_theory=True,
#                                        save_outputs=True, plots_dir="plots"):
#     if data_path is None:
#         data_path = latest_cached_data("ts_attack_cost_vs_T")

#     data = np.load(data_path)
#     T_values = data["T_values"]
#     fig = plt.figure(figsize=(12, 8))
#     avg_costs_heuristic = data["avg_costs_sequential"]
#     std_costs_heuristic = data["std_costs_sequential"]

#     empirical_series = [
#         ("Least Injection", data["avg_costs_single"], data["std_costs_single"], "blue", "o", "dotted"),
#         ("Simultaneous Bounded Injection", data["avg_costs_sequential"], data["std_costs_sequential"], "green", "x", "--"),
#         ("Periodic Bounded Injection", data["avg_costs_periodic"], data["std_costs_periodic"], "red", "s", "-"),
#         ("Heuristic Baseline", avg_costs_heuristic, std_costs_heuristic, "purple", "^", "-."),
#     ]
#     for label, mean_values, std_values, color, marker, linestyle in empirical_series:
#         plt.plot(T_values, mean_values, label=label, color=color, marker=marker,
#                  linestyle=linestyle, linewidth=2)
#         plt.fill_between(T_values, mean_values - std_values, mean_values + std_values,
#                          color=color, alpha=0.2)

#     if show_theory and show_single_theory and "avg_theory_single" in data:
#         single_mask = np.isfinite(data["avg_theory_single"])
#         plt.plot(T_values[single_mask], data["avg_theory_single"][single_mask],
#                  label="LI Theoretical Upper Bound Cost", color="darkblue", linestyle=":", linewidth=2)
#     if show_theory and "avg_theory_sequential" in data:
#         sequential_mask = np.isfinite(data["avg_theory_sequential"])
#         plt.plot(T_values[sequential_mask], data["avg_theory_sequential"][sequential_mask],
#                  label="SBI/PBI Theoretical Upper Bound Cost", color="darkgreen", linestyle=":", linewidth=2)

#     plt.tick_params(labelsize=27)
#     plt.xlabel("T", fontsize=30)
#     plt.ylabel("Average Total Attack Cost", fontsize=30)
#     plt.grid(True)
#     plt.legend(fontsize=16)
#     plt.tight_layout()
#     if save_outputs:
#         save_cached_plot(fig, "ts_attack_cost_vs_T_cached", show_theory=show_theory, plots_dir=plots_dir)
#     plt.show()
#     return fig


# def plot_cached_attack_cost_vs_delta0(data_path=None, show_theory=True, show_single_theory=True,
#                                       save_outputs=True, plots_dir="plots"):
#     if data_path is None:
#         data_path = latest_cached_data("ts_attack_cost_vs_delta0")

#     data = np.load(data_path)
#     delta0_values = data["delta0_values"]
#     fig = plt.figure(figsize=(12, 8))

#     empirical_series = [
#         ("Least Injection", data["avg_costs_single"], data["std_costs_single"], "blue", "o", "dotted"),
#         ("Simultaneous Bounded Injection", data["avg_costs_sequential"], data["std_costs_sequential"], "green", "x", "--"),
#         ("Periodic Bounded Injection", data["avg_costs_periodic"], data["std_costs_periodic"], "red", "s", "-"),
#         ("Heuristic Baseline", data["avg_costs_heuristic"], data["std_costs_heuristic"], "purple", "^", "-."),
#     ]
#     for label, mean_values, std_values, color, marker, linestyle in empirical_series:
#         plt.plot(delta0_values, mean_values, label=label, color=color, marker=marker,
#                  linestyle=linestyle, linewidth=2)
#         plt.fill_between(delta0_values, mean_values - std_values, mean_values + std_values,
#                          color=color, alpha=0.2)

#     if show_theory and show_single_theory and "avg_theory_single" in data:
#         single_mask = np.isfinite(data["avg_theory_single"])
#         plt.plot(delta0_values[single_mask], data["avg_theory_single"][single_mask],
#                  label="LI Theoretical Upper Bound Cost", color="darkblue", linestyle=":", linewidth=2)
#     if show_theory and "avg_theory_sequential" in data:
#         sequential_mask = np.isfinite(data["avg_theory_sequential"])
#         plt.plot(delta0_values[sequential_mask], data["avg_theory_sequential"][sequential_mask],
#                  label="SBI/PBI Theoretical Upper Bound Cost", color="darkgreen", linestyle=":", linewidth=2)

#     plt.tick_params(labelsize=27)
#     plt.xlabel("δ₀ (Confidence Parameter)", fontsize=30)
#     plt.ylabel("Average Total Attack Cost", fontsize=30)
#     plt.grid(True)
#     plt.legend(fontsize=16)
#     plt.tight_layout()
#     if save_outputs:
#         save_cached_plot(fig, "ts_attack_cost_vs_delta0_cached", show_theory=show_theory, plots_dir=plots_dir)
#     plt.show()
#     return fig


# def plot_cached_target_ratio_comparison(data_path=None, save_outputs=True, plots_dir="plots"):
#     if data_path is None:
#         data_path = latest_cached_data("ts_target_ratio_comparison")

#     data = np.load(data_path)
#     x = data["x"]
#     fig = plt.figure(figsize=(12, 8))

#     ratio_series = [
#         ("Least Injection", data["avg_ratios_single"], data["std_ratios_single"], "blue", "o", "dotted"),
#         ("Simultaneous Bounded Injection", data["avg_ratios_sequential"], data["std_ratios_sequential"], "green", "x", "--"),
#         ("Periodic Bounded Injection", data["avg_ratios_periodic"], data["std_ratios_periodic"], "red", "s", "-"),
#         ("Heuristic Baseline", data["avg_ratios_heuristic"], data["std_ratios_heuristic"], "purple", "^", "-."),
#     ]
#     for label, mean_values, std_values, color, marker, linestyle in ratio_series:
#         markevery = max(len(x) // 30, 1)
#         plt.plot(x, mean_values, label=label, color=color, linestyle=linestyle,
#                  marker=marker, linewidth=2, markevery=markevery)
#         plt.fill_between(x, mean_values - std_values, mean_values + std_values,
#                          color=color, alpha=0.25)

#     plt.tick_params(labelsize=27)
#     plt.xlabel("Rounds", fontsize=30)
#     plt.ylabel("Target Arm Selection Ratio", fontsize=30)
#     plt.grid(True)
#     plt.legend(fontsize=28)
#     plt.tight_layout()
#     if save_outputs:
#         save_cached_plot(fig, "ts_target_ratio_comparison_cached", show_theory=False, plots_dir=plots_dir)
#     plt.show()
#     return fig

if __name__ == "__main__":
    plot_target_ratio_comparison()
    # plot_attack_cost_comparison()
    # plot_attack_cost_vs_delta0_comparison()

    # plot_cached_attack_cost_comparison()