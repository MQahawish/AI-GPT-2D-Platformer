import statistics
import numpy as np


# Measurments
def exploitation_ratio_calc(population, fitnesses, distance_function,dim=None):
    n = len(population)
    if n <= 1:
        return 0

    # Calculate exploitation ratio using coefficient of variation
    best_fitness = max(fitnesses)
    fitness_difference = [abs(f - best_fitness) for f in fitnesses]
    mean_fitness_difference = sum(fitness_difference) / n
    if mean_fitness_difference == 0:
        exploitation_ratio = 1
    else:
        std_fitness_difference = (sum([(f - mean_fitness_difference) ** 2 for f in fitness_difference]) / n) ** 0.5
        cv = std_fitness_difference / mean_fitness_difference
        exploitation_ratio = 1 - cv / (1 + cv)

    # Calculate average pairwise distance using distance_function
    pairwise_distances = []
    for i in range(n):
        for j in range(i + 1, n):
            distance = distance_function(population[i], population[j],dim)
            pairwise_distances.append(distance)
    avg_pairwise_distance = sum(pairwise_distances) / len(pairwise_distances)

    # Normalize average pairwise distance (0 to 1)
    max_distance = max(pairwise_distances)
    if max_distance == 0:
        return 1
    normalized_pairwise_distance = avg_pairwise_distance / max_distance
    # Calculate the combined exploitation ratio using the weights
    combined_ratio = 0.5 * exploitation_ratio + 0.5 * normalized_pairwise_distance
    return combined_ratio


def selection_pressure_calc(selection_probabilities):
    n = len(selection_probabilities)
    total_prob = sum(selection_probabilities)

    # Normalize the selection probabilities
    normalized_probabilities = [p / total_prob for p in selection_probabilities]

    max_prob = max(normalized_probabilities)
    avg_prob = sum(normalized_probabilities) / n

    selection_pressure = max_prob / avg_prob
    return selection_pressure


def taspr_calc(selection_probabilities, top_n=10):
    n = len(selection_probabilities)
    # Ensure top_n is within the range [1, n]
    top_n = max(1, min(top_n, n))
    sorted_probabilities = sorted(selection_probabilities, reverse=True)
    # Calculate the average selection probability of the top N individuals
    top_n_probabilities = sorted_probabilities[:top_n]
    average_top_n_prob = sum(top_n_probabilities) / top_n
    # Calculate the average selection probability of the entire population
    average_prob = sum(selection_probabilities) / n
    # Calculate the Top-Average Selection Probability Ratio
    ratio = average_top_n_prob / average_prob
    return ratio


def exploration_ratio_calc(population, fitnesses, distance_function, threshold=5,dim=None):
    n = len(population)
    if n <= 1:
        return 0

    # Calculate exploration ratio using distance_function and threshold
    coverage = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if distance_function(population[i], population[j],dim) < threshold:
                coverage[i] += 1
                coverage[j] += 1
    exploration_ratio = sum(coverage) / (n * (n - 1))
    return exploration_ratio


def hamming_distance(s1, s2):
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def diversification_genetic(population, distance_function, max_distance,dim=None):
    # Compute the distances between every pair of solutions in the population
    distances = []
    for i, sol1 in enumerate(population):
        for j, sol2 in enumerate(population):
            if j > i:  # only compute the distance between unique pairs of solutions
                distance = distance_function(sol1, sol2,dim)
                distances.append(distance)
    # Compute the genetic diversity of the population
    if not distances:
        return 0
    avg_distance = sum(distances) / len(distances)
    return avg_distance / max_distance
