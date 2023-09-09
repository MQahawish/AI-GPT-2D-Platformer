import random
import numpy as np
from copy import deepcopy

# Crossover functions
def single_point_crossover(parent1, parent2):
    if len(parent1)!=len(parent2):
        print("parents have different lenghts!")
    # Choose a random crossover point
    parent1 = list(parent1)
    parent2 = list(parent2)
    crossover_point = random.randint(1, len(parent1) - 1)

    # Combine the first part of parent1 with the second part of parent2
    child1 = parent1[:crossover_point] + parent2[crossover_point:]

    # Combine the first part of parent2 with the second part of parent1
    child2 = parent2[:crossover_point] + parent1[crossover_point:]

    return child1, child2


def two_point_crossover(parent1, parent2):
    if len(parent1)!=len(parent2):
        print("parents have different lenghts!")
    if len(parent1) <= 2:
        return single_point_crossover(parent1, parent2)
    # Choose two random crossover points
    crossover_points = sorted(random.sample(range(1, len(parent1)), 2))
    parent1 = list(parent1)
    parent2 = list(parent2)

    # Combine the first part of parent1 with the second part of parent2
    child1 = parent1[:crossover_points[0]] + parent2[crossover_points[0]:crossover_points[1]] + parent1[
                                                                                                crossover_points[1]:]

    # Combine the first part of parent2 with the second part of parent1
    child2 = parent2[:crossover_points[0]] + parent1[crossover_points[0]:crossover_points[1]] + parent2[
                                                                                                crossover_points[1]:]

    return child1, child2


def uniform_crossover(parent1, parent2):
    if len(parent1)!=len(parent2):
        print("parents have different lenghts!")
    # Create an empty child string for each child
    child1 = ""
    child2 = ""
    parent1 = list(parent1)
    parent2 = list(parent2)
    # Get the length of the shorter parent
    length = min(len(parent1), len(parent2))
    # Iterate through each bit in the parents up to the length of the shorter parent
    for i in range(length):
        if random.random() < 0.5:
            child1 += parent1[i]
            child2 += parent2[i]
        else:
            child1 += parent2[i]
            child2 += parent1[i]
    # Add any remaining bits from the longer parent to the end of each child
    child1 += ''.join(parent1[length:])
    child2 += ''.join(parent2[length:])
    return child1, child2


def PMX_crossover(parent1, parent2, num_offspring=2):
    if len(parent1)!=len(parent2):
        print("parents have different lenghts!")
    offspring_list = []
    for _ in range(num_offspring):
        crossover_points = sorted(random.sample(range(len(parent1)), 2))
        mapping = {}
        for i in range(crossover_points[0], crossover_points[1]):
            mapping[parent1[i]] = parent2[i]
            mapping[parent2[i]] = parent1[i]
        offspring = []
        for i in range(len(parent1)):
            if i < crossover_points[0] or i >= crossover_points[1]:
                offspring.append(parent1[i])
            else:
                offspring.append(mapping[parent1[i]])

        offspring_list.append(''.join(offspring))
    return offspring_list


def CX_crossover(parent1, parent2, probability=0.5):
    length = len(parent1)
    if len(parent1)!=len(parent2):
        print("parents have different lenghts!")
    child_one = ["1"] * length
    child_two = ["1"] * length

    if np.random.random() < probability:  # if pc is greater than random number
        p1_copy = list(parent1)
        p2_copy = list(parent2)
        swap = True
        count = 0
        pos = 0

        while True:
            if count > length:
                break
            for i in range(length):
                if child_one[i] == -1:
                    pos = i
                    break

            if swap:
                while True:
                    child_one[pos] = parent1[pos]
                    count += 1
                    try:
                        pos = list(parent2).index(parent1[pos])
                    except ValueError:
                        break
                    if p1_copy[pos] == -1:
                        swap = False
                        break
                    p1_copy[pos] = -1
            else:
                while True:
                    child_one[pos] = parent2[pos]
                    count += 1
                    try:
                        pos = list(parent1).index(parent2[pos])
                    except ValueError:
                        break
                    if p2_copy[pos] == -1:
                        swap = True
                        break
                    p2_copy[pos] = -1

        for i in range(length):  # for the second child
            if child_one[i] == parent1[i]:
                child_two[i] = parent2[i]
            else:
                child_two[i] = parent1[i]

        for i in range(length):  # Special mode
            if child_one[i] == -1:
                if p1_copy[i] == -1:  # it means that the ith gene from p1 has been already transfered
                    child_one[i] = parent2[i]
                else:
                    child_one[i] = parent1[i]
    else:  # if pc is less than random number then don't make any change
        child_one = deepcopy(parent1)
        child_two = deepcopy(parent2)
    return child_one, child_two


def no_crossover(parent1, parent2):
    if len(parent1)!=len(parent2):
        print("parents have different lenghts!")
    return parent1, parent2