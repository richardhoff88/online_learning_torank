import numpy as np

means = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45]

K = 10

def init_reward(mu: float) -> None:
    global means
    #在 0~mu 之间均匀分布
    means = np.random.uniform(0, mu, K)
    # 从大到小
    means = np.sort(means)[::-1]


def get_reward(movie_id: int, sigma: float) -> float:
    mean_rating = means[movie_id]
    return np.random.normal(loc=mean_rating, scale=sigma)

def real_mean(movie_id: int) -> float:
    return means[movie_id]



# def load_movielens_dataset():
#     import os
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     project_root = os.path.dirname(current_dir)
#     data_file = os.path.join(project_root, 'dataset.txt')
#     data = np.loadtxt(data_file)
#     return data

# def calculate_movie_means(data):
#     min_rating = data.min()
#     max_rating = data.max()
#     normalized_data = (data - min_rating) / (max_rating - min_rating)
    
#     movie_means = np.nanmean(normalized_data, axis=0)
#     return movie_means

# class MovieLensReward:
#     def __init__(self, K=10):
#         data = load_movielens_dataset()
#         means = calculate_movie_means(data)
        
#         top_k_indices = np.argsort(means)[-K:][::-1]
#         self.means = means[top_k_indices]
        
#     def get_reward(self, movie_id: int, sigma: float) -> float:
#         mean_rating = self.means[movie_id]
#         return np.random.normal(loc=mean_rating, scale=sigma)
        
#     def get_real_mean(self, movie_id: int) -> float:
#         return self.means[movie_id]

# reward_generator = MovieLensReward()

# def get_reward(movie_id: int, sigma: float) -> float:
#     return reward_generator.get_reward(movie_id, sigma)

# def real_mean(movie_id: int) -> float:
#     return reward_generator.get_real_mean(movie_id)