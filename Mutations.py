import random
import statistics


# Mutation Functions

def swap_mutation(gene, probability):
    mutated_gene = list(gene)
    for i in range(len(mutated_gene)):
        if random.random() < probability:
            # Choose a random position in the string to swap with
            swap_pos = random.randint(0, len(mutated_gene) - 1)
            mutated_gene[i], mutated_gene[swap_pos] = mutated_gene[swap_pos], mutated_gene[i]
    return ''.join(mutated_gene)


def scramble_mutation(gene, probability):
    mutated_gene = list(gene)
    for i in range(len(mutated_gene)):
        if random.random() < probability:
            # Choose a random slice of the string to scramble
            start_pos = random.randint(0, len(mutated_gene) - 1)
            end_pos = random.randint(start_pos + 1, len(mutated_gene))
            scrambled_slice = list(mutated_gene[start_pos:end_pos])
            random.shuffle(scrambled_slice)
            mutated_gene[start_pos:end_pos] = scrambled_slice
    return ''.join(mutated_gene)




def inversion_mutation(gene, probability):
    gene = list(gene)
    for i in range(len(gene)):
        if random.random() < probability:
            pos1 = random.randint(0, len(gene) - 1)
            pos2 = random.randint(0, len(gene) - 1)
            if pos1 > pos2:
                pos1, pos2 = pos2, pos1
            gene[pos1:pos2 + 1] = reversed(gene[pos1:pos2 + 1])
    return ''.join(gene)


def no_mutation(gene, probability):
    return gene


# Mutation control functions
def constant(mut_p, fitness_values=None, ind_fitness=None, best_history=None, current_generation=None):
    return mut_p


def non_uniform_mutation_rate(mut_p, fitness_values, ind_fitness, best_history, current_generation, decay_factor=0.99):
    avg_fitness = sum(fitness_values) / len(fitness_values)
    stdev_fitness = statistics.stdev(fitness_values)
    if stdev_fitness == 0:
        adjusted_mutation_prob = mut_p
    else:
        adjusted_mutation_prob = mut_p * (1 + (avg_fitness / stdev_fitness))
    mutation_rate = adjusted_mutation_prob * (decay_factor ** current_generation)
    return mutation_rate


def adaptive_mutation_prob_population(mut_p, fitness_values, ind_fitness, best_history, current_generation=None):
    avg_fitness = sum(fitness_values) / len(fitness_values)
    stdev_fitness = statistics.stdev(fitness_values)
    if stdev_fitness == 0:
        adjusted_mutation_prob = mut_p
    else:
        adjusted_mutation_prob = mut_p * (1 + (avg_fitness / stdev_fitness))
    return adjusted_mutation_prob


def triggered_hyper_mutation(mut_p, fitness_values, ind_fitness, best_history, current_generation=None,
                             stagnation_threshold=10,
                             increase_factor=1.2):
    if len(best_history) < stagnation_threshold:
        return mut_p

    stagnation_count = 0

    for i in range(1, len(best_history)):
        if abs(best_history[i] - best_history[i - 1]) <= best_history[i] * 0.01:
            stagnation_count += 1
        else:
            stagnation_count = 0

        if stagnation_count >= stagnation_threshold - 1:
            new_mut_p = mut_p * increase_factor
            return min(new_mut_p, 1)  # Ensure the new probability is not greater than 1
    return mut_p


def adaptive_mutation_prob_individual(mut_p, fitness_values, ind_fitness, best_history=None, current_generation=None):
    max_fitness = max(fitness_values)
    min_fitness = min(fitness_values)

    if max_fitness == min_fitness:
        rel_fitness = 1
    else:
        rel_fitness = (ind_fitness - min_fitness) / (max_fitness - min_fitness)

    return mut_p * (1 - rel_fitness)
