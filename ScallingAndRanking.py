import numpy as np
import math
from scipy.stats import rankdata

# Ranking functions
def linear_ranking(population, fitness_values):
    ranks = rankdata(-1 * np.array(fitness_values))
    alpha = 1.5  # selection pressure
    pmin = 0.1  # minimum selection probability
    n = len(ranks)
    p = pmin + (2 * alpha - 2 * alpha * pmin) * (ranks - 1) / (n - 1)
    return p


def non_linear_ranking(population, fitness_values):
    # Define non-linear ranking parameters
    c = 2.0  # scaling parameter
    ranks = rankdata(-1 * np.array(fitness_values))
    n = len(ranks)
    exp_rank = np.exp(-1 * c * (ranks - 1))
    p = exp_rank / np.sum(exp_rank)
    return p


def exponential_ranking(population, fitness_values):
    c = 2.0  # scaling parameter
    ranks = rankdata(-1 * np.array(fitness_values))
    n = len(ranks)
    exp_rank = np.exp(-1 * c * (ranks - 1))
    p = exp_rank / np.sum(exp_rank)
    return p


# scalling Methods for fitness values
def linear_scaling(fitness_values, a=1.5):
    avg_fitness = sum(fitness_values) / len(fitness_values)
    std_dev = math.sqrt(sum([(f - avg_fitness) ** 2 for f in fitness_values]) / len(fitness_values))
    scaled_values = []
    for f in fitness_values:
        if std_dev == 0:
            scaled_values.append(f)
        else:
            scaled_value = a * ((f - avg_fitness) / std_dev) + 0.5
            scaled_values.append(scaled_value)
    return scaled_values


def min_max_scaling(fitness_values):
    min_val = min(fitness_values)
    max_val = max(fitness_values)
    if max_val == min_val:
        scaled_values = [f for f in fitness_values]
    else:
        scaled_values = [(f - min_val) / (max_val - min_val) for f in fitness_values]
    return scaled_values


def exponential_scaling(fitness_scores):
    scaled_fitness_scores = [math.sqrt(fitness_score + 1) for fitness_score in fitness_scores]
    total_scaled_fitness = sum(scaled_fitness_scores)
    if total_scaled_fitness == 0:
        return no_scaling(fitness_scores)

    normalized_scaled_fitness_scores = [scaled_fitness_score / total_scaled_fitness for scaled_fitness_score in
                                        scaled_fitness_scores]
    return normalized_scaled_fitness_scores


def no_scaling(fitness_scores):
    return fitness_scores
