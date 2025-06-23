# Online Learning ToRank: Robustness and Adversarial Attacks

## Overview

This project investigates the robustness of recommender systems through **online learning** and **multi-armed bandit (MAB) algorithms**, with a particular focus on adversarial attacks and defense.

Working under **Postdoc Jinhang Zuo** along with **Eric He** and **Qirun Zheng**, this research explores how knowledge from adversarial attacks on stochastic bandits can be leveraged to develop both:

- Novel **attack algorithms** targeting recommender systems
- More robust **reinforcement learning (RL) policies** to defend against such attacks

We develop a method to attack bandits via fake data injection.

---

## Project Structure
```
online_learning_torank/
├── algorithms/ # Bandit and ranking algorithms
├── dataset/ # Dataset files and preprocessing
├── thompson/ # Thompson sampling implementations
├── observation_free/ # Observation-free attack code
├── attack.py # Core attack logic
├── attack_real.py # Real-world attack experiments
├── attack_log.py # Attack logging utilities
├── movielens.py # MovieLens dataset integration
├── params_experiment.py # Experiment configuration
├── figures/ # Plots and visualizations
└── dataset.txt # Dataset details
```

## Running Experiments
```bash
python attack_real.py
```

## Research

[arXiv 05.2025](https://arxiv.org/abs/2505.21938): Practical Adversarial Attacks on Stochastic Bandits via Fake Data Injection
