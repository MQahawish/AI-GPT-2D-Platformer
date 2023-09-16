import random
from ScallingAndRanking import linear_ranking


# parent selection functions
def SUS(population, fitness_scores, num_parents=2):
    total_fitness = sum(fitness_scores)
    if total_fitness == 0:
        return randomz(population, fitness_scores)
    selection_interval = total_fitness / num_parents
    start_point = random.uniform(0, selection_interval)
    parents = []
    parents_indices = []  # Store the indices of the selected parents
    cumulative_fitness = 0
    i = 0
    # Calculate selection probabilities (relative fitness)
    relative_fitness = [f / total_fitness for f in fitness_scores]
    while len(parents) < num_parents:
        cumulative_fitness += fitness_scores[i]
        if cumulative_fitness >= start_point:
            parents.append(population[i])
            parents_indices.append(i)  # Add the index of the selected parent
            start_point += selection_interval
        i = (i + 1) % len(population)
    return parents, parents_indices, relative_fitness


def RWS(population, fitness_values, num_parents=2, max_attempts=100):
    total_fitness = sum(fitness_values)
    if total_fitness == 0:
        return randomz(population, fitness_values)
    relative_fitness = [f / total_fitness for f in fitness_values]
    cumulative_probability = [sum(relative_fitness[:i + 1]) for i in range(len(relative_fitness))]
    selected = []
    selected_indices = set()
    attempts = 0
    while len(selected) < num_parents and attempts < max_attempts:
        rand_num = random.uniform(0, 1)
        for i in range(len(cumulative_probability)):
            if rand_num <= cumulative_probability[i]:
                if i not in selected_indices:
                    selected.append(population[i])
                    selected_indices.add(i)
                    break
                else:
                    attempts += 1
                    break                  
    # If we haven't selected enough parents, randomly pick the rest
    while len(selected) < num_parents:
        rand_index = random.randint(0, len(population) - 1)
        if rand_index not in selected_indices:
            selected.append(population[rand_index])
            selected_indices.add(rand_index)
    
    return selected, list(selected_indices), relative_fitness



def tournament(population, fitness_values, num_parents=2, k=5):
    p = linear_ranking(population, fitness_values)
    selected_indices = random.choices(range(len(population)), weights=p, k=k)  # Select indices instead of individuals
    selected = [population[i] for i in selected_indices]  # Get the selected individuals using indices
    # Create a dictionary mapping individuals to their fitness values
    fitness_map = {individual: fitness for individual, fitness in zip(population, fitness_values)}
    # Sort the selected individuals based on their fitness values using the fitness_map
    top_parents_indices = sorted(selected_indices, key=lambda x: fitness_map[population[x]], reverse=True)[:num_parents]
    top_parents = [population[i] for i in top_parents_indices]  # Get the top parents using indices
    return top_parents, top_parents_indices, p


def randomz(population, fitnesses):
    # Calculate selection probabilities for random sampling
    relative_fitness = [1 / len(population) for _ in population]

    selected_parent_indices = random.sample(range(len(population)), 2)  # Select parent indices
    selected_parents = [population[i] for i in selected_parent_indices]  # Get the selected parents using indices

    return selected_parents, selected_parent_indices, relative_fitness

