import math
import random
import numpy as np
import time
import sys
import os
import multiprocessing
import cProfile
import pstats
import networkx as nx
import matplotlib.pyplot as plt
from scipy.special import expit
import random


from Crossover import single_point_crossover, two_point_crossover, uniform_crossover, PMX_crossover, CX_crossover
from Mutations import swap_mutation, scramble_mutation, inversion_mutation, constant, \
    non_uniform_mutation_rate, adaptive_mutation_prob_individual, triggered_hyper_mutation, \
    adaptive_mutation_prob_population
from Selection import SUS, RWS, tournament, randomz
from ScallingAndRanking import linear_scaling, min_max_scaling, exponential_scaling, no_scaling
from Measurments import exploitation_ratio_calc, selection_pressure_calc, taspr_calc, exploration_ratio_calc, \
    diversification_genetic
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import warnings
from datavisuals import visualize_results
from sklearn.exceptions import ConvergenceWarning


class SuppressConsoleOutput:
    def __enter__(self):
        self.original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()
        sys.stdout = self.original_stdout


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

target_std_fitness = 0.3


def elbow_point(k_values, scores):
    # Normalize k_values and scores
    k_norm = [(k - min(k_values)) / (max(k_values) - min(k_values)) for k in k_values]
    s_norm = [(s - min(scores)) / (max(scores) - min(scores)) for s in scores]

    # Coordinates of the line connecting the first and last points
    line_start = (k_norm[0], s_norm[0])
    line_end = (k_norm[-1], s_norm[-1])

    # Calculate the perpendicular distance from each point to the line
    max_distance = -1
    elbow_index = -1

    for i in range(len(k_values)):
        point = (k_norm[i], s_norm[i])
        distance = point_line_distance(point, line_start, line_end)

        if distance > max_distance:
            max_distance = distance
            elbow_index = i

    return k_values[elbow_index]

def point_line_distance(point, line_start, line_end):
    numerator = abs(
        (line_end[1] - line_start[1]) * point[0] - (line_end[0] - line_start[0]) * point[1] + line_end[0] * line_start[
            1] - line_end[1] * line_start[0])
    denominator = ((line_end[1] - line_start[1]) ** 2 + (line_end[0] - line_start[0]) ** 2) ** 0.5
    return numerator / denominator

def find_optimal_k(population, max_k, method='silhouette'):
    data_points = np.vstack([np.array([float(value) for value in individual.values]) for individual in population])
    k_values = range(2, max_k + 1)
    scores = []
    score = 0
    optimal_k = 0

    for k in k_values:
        clustering_model = KMeans(n_clusters=k, n_init=10)
        clustering_model.fit(data_points)

        if method == 'silhouette':
            if len(np.unique(clustering_model.labels_)) > 1:
                score = silhouette_score(data_points, clustering_model.labels_)
            else:
                score = -1  # Assign a low score if there's only one cluster
        elif method == 'elbow':
            score = -clustering_model.inertia_  # SSE is stored in the 'inertia_' attribute, use negative value to
            # find the elbow point

        scores.append(score)

    if method == 'silhouette':
        optimal_k = k_values[scores.index(max(scores))]
    elif method == 'elbow':
        optimal_k = elbow_point(k_values, scores)

    return optimal_k

def speciate(population, k, method='kmeans'):
    # Extract the genetic material of the individuals as data points
    data_points = np.vstack([np.array([float(value) for value in individual.values]) for individual in population])

    if method == 'kmeans':
        clustering_model = KMeans(n_clusters=k)
    elif method == 'kmedoids':
        clustering_model = AgglomerativeClustering(n_clusters=k)

    # Perform clustering
    clustering_model.fit(data_points)

    # Assign individuals to species based on cluster labels
    species = [[] for _ in range(k)]
    for individual, label in zip(population, clustering_model.labels_):
        species[label].append(individual)
    filtered_species = [spec for spec in species if spec]
    return filtered_species

def boltzmann_replacement_probability(delta_fitness, temperature):
    return expit(-delta_fitness / temperature)

def shared_fitness(individual_index, population, fitnesses, r, dist_func,dim=None):
    individual_fitness = fitnesses[individual_index]
    sharing_factor = sum([1 - (dist_func(population[individual_index], other,dim) / r) ** 2 if dist_func(
        population[individual_index], other,dim) < r else 0 for other in population])
    return individual_fitness / sharing_factor

def calculate_shared_fitnesses(population, fitnesses, r, dist_func,dim=None):
    shared_fitnesses = [shared_fitness(i, population, fitnesses, r, dist_func,dim) for i in range(len(population))]
    return shared_fitnesses

def hamm_dist(individual1, individual2,dim=None):
    distance = 0
    for i in range(len(individual1.values)):
        if individual1.values[i] != individual2.values[i]:
            distance += 1
    return distance

def similarity(individual1, individual2, dist_func, max_distance,dim=None):
    distance = dist_func(individual1, individual2,dim)
    return 1 - distance / max_distance

def is_valid_layout(encoded_rows):
    s_count = 0
    t_count = 0
    # Check for 'S' in the first column of any row
    for row in encoded_rows:
        if row[0] == 'S':
            s_count += 1
    # Check for 'T' in the last column of any row
    for row in encoded_rows:
        if row[-1] == 'T':
            t_count += 1

    return s_count == 1 and t_count == 1

def count_effective_ineffective_timed_hazards(encoded_rows):
    rows = len(encoded_rows)
    cols = len(encoded_rows[0])
    ineffective_count = 0
    effective_count=0

    for row in range(rows):
        for col in range(cols):
            current_cell = encoded_rows[row][col]
            if current_cell == 'M':
                has_effect = False
                for adj_row in range(row - 1, row + 2):
                    for adj_col in range(col - 1, col + 2):
                        if 0 <= adj_row < rows and 0 <= adj_col < cols:
                            adj_cell = encoded_rows[adj_row][adj_col]
                            if adj_cell == 'P':  # or any other condition you deem as "effective"
                                has_effect = True
                                break
                    if has_effect:
                        effective_count+=1
                        break
                if not has_effect:
                    ineffective_count+=1
    return effective_count,ineffective_count

def easy_level_fitness(individual,difficulty='easy'):
    level = binary_to_encoding(individual,difficulty)
    G = construct_and_draw_graph(level, False)
    # Find nodes with specific colors (S and T)
    start_node = [node for node, attr in G.nodes(data=True) if attr['color'] == 'green'][0]
    end_node = [node for node, attr in G.nodes(data=True) if attr['color'] == 'red'][0]
    
    # Check if there's at least one black edge connected to T
    if all(G[u][v].get('color', 'black') != 'black' for u, v in G.edges(end_node)):
        return 0
    
    # Count blue nodes
    blue_nodes_count = sum(1 for _, attr in G.nodes(data=True) if attr.get('color') == 'blue')
    
    # Calculate percentage of blue nodes
    blue_nodes_ratio = blue_nodes_count / G.number_of_nodes()

    # count the ratio of red edges to all edges
    red_edges_count = sum(1 for _, _, attr in G.edges(data=True) if attr.get('color') == 'red')
    red_edges_ratio = red_edges_count / G.number_of_edges()

    # get all simple paths from S to T
    try:
        paths = nx.all_simple_paths(G, source=start_node, target=end_node)
    except nx.NetworkXNoPath:
        return 0
    
    # we want to cap the number of paths to 4 and penalize for each additional path
    number_of_paths = len(list(paths))
    path_factor = 3/number_of_paths
    
    if blue_nodes_ratio >= 0.7:
        return 1 * path_factor * (1-red_edges_ratio)  # Or some high fitness value
    else:
        return blue_nodes_ratio * path_factor * (1-red_edges_ratio)  # Or some function of blue_nodes_ratio to scale the fitness

def medium_level_fitness(individual):
    level = binary_to_encoding(individual)
    G = construct_and_draw_graph(level, False)
    # Find nodes with specific colors (S and T)
    start_node = [node for node, attr in G.nodes(data=True) if attr['color'] == 'green'][0]
    end_node = [node for node, attr in G.nodes(data=True) if attr['color'] == 'red'][0]
    try:
        shortest_path = nx.shortest_path(G, source=start_node, target=end_node)
    except nx.NetworkXNoPath:
        return 0

    blue_nodes = sum(1 for node in shortest_path if G.nodes[node].get('color') == 'blue')
    hazard_nodes = sum(1 for node in shortest_path if G.nodes[node].get('color') in ['orange', 'pink'])
    
    return hazard_nodes / (blue_nodes + 1)  # +1 to avoid division by zero

def hard_level_fitness(individual):
    level = binary_to_encoding(individual)
    G = construct_and_draw_graph(level, False)
    # Find nodes with specific colors (S and T)
    start_node = [node for node, attr in G.nodes(data=True) if attr['color'] == 'green'][0]
    end_node = [node for node, attr in G.nodes(data=True) if attr['color'] == 'red'][0]
    try:
        shortest_path = nx.shortest_path(G, source=start_node, target=end_node)
    except nx.NetworkXNoPath:
        return 0

    path_length = len(shortest_path)
    blue_nodes = sum(1 for node in shortest_path if G.nodes[node].get('color') == 'blue')
    hazard_nodes = sum(1 for node in shortest_path if G.nodes[node].get('color') in ['orange', 'pink'])
    
    return (hazard_nodes / (blue_nodes + 1)) * path_length

class chromosome:
    def __init__(self, values, age, genome_length=None,fitness=None):
        self.values = values
        self.age = age
        self.genome_length = genome_length
        self.fitness=fitness

    def setgenome_length(self, genome_length):
        self.genome_length = genome_length

    def setage(self, age):
        self.age = age

    def set_fitness(self,fitness):
        self.fitness=fitness

    def __eq__(self, other):
        if isinstance(other, chromosome):
            return self.values == other.values

    def __hash__(self):
        return hash(tuple(self.values))

def generate_chunk_population(pop_size,difficulty, genome_length=12):
    population = []
    for i in range(pop_size):
        values = [random.randint(0, 1) for _ in range(genome_length*4)]
        population.append(chromosome(values, 0, genome_length))
    #preprocess the population
    for individual in population:
        individual = preprocess_individual(individual,difficulty)
    return population

def binary_to_encoding(individual,difficulty=None):
    # Platform: Represented by 'P' = 001.
    # Empty Space: Represented by 'E' = 000.
    # Enemy: Represented by 'N' = 010 for enemies on platforms and 'F' = 011 for flying enemies.
    # Timed Hazards (Radius): Represented by 'M' = 100.
    # Starting Point: Represented by 'S'= 101 .
    # Ending Point: Represented by 'T'= 110.
    # Static Hazard : Represented by 'H'= 111.
    #first split the binary into genome_length chunks
    binary = individual.values
    rows = [binary[i:i + individual.genome_length] for i in range(0, len(binary), individual.genome_length)]
    encoded_rows = []
    if difficulty=='hard':
        for row in rows:
            chunked_row = [row[i:i + 3] for i in range(0, len(row), 3)]
            encoded_row = []
            for chunk in chunked_row:
                chunk_str = ''.join(str(bit) for bit in chunk)
                if chunk_str == '000':
                    encoded_row.append('E')
                elif chunk_str == '001':
                    encoded_row.append('P')
                elif chunk_str == '010':
                    encoded_row.append('N')
                elif chunk_str == '011':
                    encoded_row.append('F')
                elif chunk_str == '100':
                    encoded_row.append('M')
                elif chunk_str == '101':
                    encoded_row.append('S')
                elif chunk_str == '110':
                    encoded_row.append('T')
                elif chunk_str == '111':
                    encoded_row.append('H')
            encoded_rows.append(encoded_row)
    elif difficulty=='medium':
        for row in rows:
            chunked_row = [row[i:i + 3] for i in range(0, len(row), 3)]
            encoded_row = []
            for chunk in chunked_row:
                chunk_str = ''.join(str(bit) for bit in chunk)
                if chunk_str == '000':
                    encoded_row.append('E')
                elif chunk_str == '001':
                    encoded_row.append('P')
                elif chunk_str == '010':
                    encoded_row.append('N')
                elif chunk_str == '011':
                    encoded_row.append('H')
                elif chunk_str == '100':
                    encoded_row.append('P')
                elif chunk_str == '101':
                    encoded_row.append('S')
                elif chunk_str == '110':
                    encoded_row.append('T')
                elif chunk_str == '111':
                    encoded_row.append('E')
            encoded_rows.append(encoded_row)
    elif difficulty=='easy':
        for row in rows:
            chunked_row = [row[i:i + 3] for i in range(0, len(row), 3)]
            encoded_row = []
            for chunk in chunked_row:
                chunk_str = ''.join(str(bit) for bit in chunk)
                if chunk_str == '000':
                    encoded_row.append('E')
                elif chunk_str == '001':
                    encoded_row.append('P')
                elif chunk_str == '010':
                    encoded_row.append('N')
                elif chunk_str == '011':
                    encoded_row.append('E')
                elif chunk_str == '100':
                    encoded_row.append('P')
                elif chunk_str == '101':
                    encoded_row.append('S')
                elif chunk_str == '110':
                    encoded_row.append('T')
                elif chunk_str == '111':
                    encoded_row.append('E')
            encoded_rows.append(encoded_row)
    return encoded_rows

def preprocess_individual(individual,difficulty=None):
    # Convert the binary genome to the encoding
    encoded_rows = binary_to_encoding(individual,difficulty)
    
    # Convert all 'S' and 'T' to 'P'
    for row in encoded_rows:
        for col_idx, cell in enumerate(row):
            if cell == 'S' or cell == 'T':
                row[col_idx] = 'P'
    
    # Gather the indices of 'P' in the first and last columns
    s_indices = [i for i, row in enumerate(encoded_rows) if row[0] == 'P']
    t_indices = [i for i, row in enumerate(encoded_rows) if row[-1] == 'P']

    # Randomly pick one 'P' to convert to 'S' and one to 'T', if available
    if s_indices:
        s_to_convert = random.choice(s_indices)
        encoded_rows[s_to_convert][0] = 'S'
    else:
        encoded_rows[0][0] = 'S'  # Default to first row if no 'P' is found

    if t_indices:
        t_to_convert = random.choice(t_indices)
        encoded_rows[t_to_convert][-1] = 'T'
    else:
        encoded_rows[-1][-1] = 'T'  # Default to last row if no 'P' is found
    
    # Convert the encoding back to binary and update the individual's values
    individual.values = encoding_to_binary(encoded_rows,difficulty)
    return individual

def encoding_to_binary(encoded_rows,difficulty=None):
    binary = []
    if difficulty=='hard':
        for row in encoded_rows:
            for cell in row:
                if cell == 'E':
                    binary.append('000')
                elif cell == 'P':
                    binary.append('001')
                elif cell == 'N':
                    binary.append('010')
                elif cell == 'F':
                    binary.append('011')
                elif cell == 'M':
                    binary.append('100')
                elif cell == 'S':
                    binary.append('101')
                elif cell == 'T':
                    binary.append('110')
                elif cell == 'H':
                    binary.append('111')
    elif difficulty=='medium':
        for row in encoded_rows:
            for cell in row:
                if cell == 'E':
                    binary.append('000')
                elif cell == 'P':
                    binary.append('001')
                elif cell == 'N':
                    binary.append('010')
                elif cell == 'H':
                    binary.append('011')
                elif cell == 'S':
                    binary.append('101')
                elif cell == 'T':
                    binary.append('110')
    elif difficulty=='easy':
        for row in encoded_rows:
            for cell in row:
                if cell == 'E':
                    binary.append('000')
                elif cell == 'P':
                    binary.append('001')
                elif cell == 'N':
                    binary.append('010')
                elif cell == 'S':
                    binary.append('101')
                elif cell == 'T':
                    binary.append('110')
    return ''.join(binary)

def remove_unreachable_nodes(G):
    # Find all start ('S') and target ('T') nodes
    start_nodes = [node for node, data in G.nodes(data=True) if data.get('color') == 'green']
    target_nodes = [node for node, data in G.nodes(data=True) if data.get('color') == 'red']
    
    # Collect all nodes that are in any path from 'S' to 'T'
    nodes_in_paths = set()
    for start in start_nodes:
        for target in target_nodes:
            try:
                for path in nx.all_simple_paths(G, source=start, target=target):
                    nodes_in_paths.update(path)
            except nx.NodeNotFound:
                continue
    # Remove nodes that are not in any path and are blue, pink, or orange
    nodes_to_remove = []
    for node, data in G.nodes(data=True):
        if data.get('color') in ['blue', 'pink', 'orange'] and node not in nodes_in_paths:
            nodes_to_remove.append(node)
    G.remove_nodes_from(nodes_to_remove)
    return G

def color_edges_unreachable_from_S(G):
    start_nodes = [node for node, attr in G.nodes(data=True) if attr.get('color') == 'green']
    
    reachable_edges = set()
    for start_node in start_nodes:
        visited_nodes = set()
        stack = [start_node]
        
        while stack:
            current_node = stack.pop()
            visited_nodes.add(current_node)
            
            for neighbor in G.neighbors(current_node):
                if neighbor not in visited_nodes:
                    stack.append(neighbor)
                    reachable_edges.add((current_node, neighbor))
                    reachable_edges.add((neighbor, current_node))  # Add in reverse as well since the graph is undirected

    for edge in G.edges():
        if edge in reachable_edges:
            G.edges[edge]['color'] = 'black'  # Color for reachable edges
        else:
            G.edges[edge]['color'] = 'red'  # Color for unreachable edges
            
    return G

def construct_and_draw_graph(encoded_rows,draw=False):
    G = nx.Graph()
    rows = len(encoded_rows)
    cols = len(encoded_rows[0])
    label_map = {'P': 'Plat', 'S': 'Start', 'T': 'Tgt', 'N': 'Enemy', 'M': 'RadHz', 'H': 'StcHz'}
    
    for col in range(cols):
        for row in range(rows):
            current_cell = encoded_rows[row][col]
            current_node = (row, col)
            color_map = {'P': 'blue', 'S': 'green', 'T': 'red', 'N': 'orange', 'M': 'purple', 'H': 'yellow'}
            color = color_map.get(current_cell, None)
            
            if color:
                G.add_node(current_node, color=color, label=label_map.get(current_cell, ''))
            
            if current_cell in ['P', 'S', 'T', 'N']:
                for adj_row in range(row - 1, row + 2):
                    if 0 <= adj_row < rows:
                        next_cell = encoded_rows[adj_row][col + 1] if col + 1 < cols else None
                        next_node = (adj_row, col + 1)
                        
                        if next_cell in ['P', 'S', 'T', 'N']:
                            if next_node not in G:
                                color = color_map.get(next_cell, None)
                                if color:
                                    G.add_node(next_node, color=color, label=label_map.get(next_cell, ''))
                            G.add_edge(current_node, next_node)
                            
    for row in range(rows):
        for col in range(cols):
            current_cell = encoded_rows[row][col]
            if current_cell == 'M':
                for adj_row in range(row - 1, row + 2):
                    for adj_col in range(col - 1, col + 2):
                        if 0 <= adj_row < rows and 0 <= adj_col < cols:
                            adj_node = (adj_row, adj_col)
                            if G.nodes.get(adj_node, {}).get('color') == 'blue':
                                G.nodes[adj_node]['color'] = 'pink'

    # G = remove_unreachable_nodes(G)
    G = color_edges_unreachable_from_S(G)
    edge_colors = [G[u][v].get('color', 'black') for u, v in G.edges()]
    pos = {(x, y): (y, -x) for x, y in G.nodes()}
    labels = {node: G.nodes[node]['label'] for node in G.nodes()}
    if draw:
        nx.draw(G, pos, labels=labels, with_labels=True,
                node_color=[nx.get_node_attributes(G, 'color').get(n, 'black') for n in G.nodes()],
                edge_color=edge_colors,
                node_size=300, font_size=8)
        plt.show()
    return G

def calculate_paths(G,shortest_path_length):
    # Find coordinates of 'S' (start) and 'T' (target) based on their colors
    node_colors = nx.get_node_attributes(G, 'color')
    start = [node for node, color in node_colors.items() if color == 'green'][0]
    end = [node for node, color in node_colors.items() if color == 'red'][0]
    
    # Find all simple paths from 'S' to 'T'
    all_paths = list(nx.all_simple_paths(G, source=start, target=end,cutoff=shortest_path_length*1.5))
    num_paths = len(all_paths)
    paths_through_enemy = 0
    
    # Count the number of paths that go through an enemy node (yellow)
    for path in all_paths:
        if any(node_colors[node] == 'orange' or node_colors== 'pink' for node in path):
            paths_through_enemy += 1
            
    return num_paths, paths_through_enemy

def find_shortest_path(G):
    # Find nodes with specific colors (S and T)
    start_node = [node for node, attr in G.nodes(data=True) if attr['color'] == 'green'][0]
    end_node = [node for node, attr in G.nodes(data=True) if attr['color'] == 'red'][0]

    # Use Dijkstra's algorithm to find the shortest path
    try:
        path = nx.shortest_path(G, source=start_node, target=end_node)
        return path
    except nx.NetworkXNoPath:
        return "No path exists between S and T."

# Define the genetic algorithm
def genetic_algorithm(pop_size, fitness_funcs, max_generations, parent_selections, crossover_funcs,
                      mutation_funcs, mutation_controlls, scale, aging,distance_func, similarity_func,
                      speciation=False,
                      sharing=False,
                      crowding=False,
                      mut_p=0.5,difficulty=None):
    #choose the fitness function based on the given difficulty
    if difficulty=='easy':      
        fitness_func=fitness_funcs[0]
    elif difficulty=='medium':
        fitness_func=fitness_funcs[1]
    elif difficulty=='hard':
        fitness_func=fitness_funcs[2]
    # randomly pick a mutation controll method
    mutation_controll = random.choice(mutation_controlls)
    population = []
    T_start = 50
    alpha = 0.5
    threshold = 0
    count = 0
    exploitation_ratio = []
    exploration_ratio = []
    selection_pressure = []
    fitness_variance = []
    genetic_diversity = []
    taspr = []
    generation_fittest = []
    selection_propabilities = []
    gen_best_ind = []
    avg_gen_fitness = []
    population = generate_chunk_population(pop_size,difficulty)
    threshold = 0.3 * len(population[0].values)
    max_distance = len(population[0].values)
    print("The mutation controll method is: ", mutation_controll.__name__)
    if speciation:
        k = find_optimal_k(population, 19, method='silhouette')
        species = speciate(population, k, method='kmeans')
    else:
        species = [population]
    # Evolve the population for a fixed number of generations
    for generation in range(max_generations):
        loop_start = time.perf_counter()
        # randomly choose a parent selection method, a crossover method and a mutation method
        parent_selection = random.choice(parent_selections)
        crossover_func = random.choice(crossover_funcs)
        mutation_func = random.choice(mutation_funcs)
        for i in range(len(species)):
            sp = species[i]
            # Evaluate the fitness of each individual
            fitnesses = [fitness_func(individual) for individual in
                         sp]
            fitnesses = scale(fitnesses)
            if sharing:
                fitnesses = calculate_shared_fitnesses(sp, fitnesses, threshold / 2, distance_func)
            # Select the best individuals for reproduction
            elite_size = max(2, int(len(sp) * 0.1))
            elite_indices = sorted(range(len(sp)), key=lambda i: fitnesses[i], reverse=True)[:elite_size]
            elites = [sp[i] for i in elite_indices]
            elites_fitness = [fitnesses[i] for i in elite_indices]

            # Generate new individuals by applying crossover and mutation operators
            offspring = []
            while len(offspring) < len(sp) - elite_size:
                parents, parents_indices, selection_propabilities = parent_selection(elites, elites_fitness)
                genome_length = parents[0].genome_length
                children_vals = crossover_func(parents[0].values, parents[1].values)

                children = [
                    chromosome(child_vals, 0, genome_length)
                    for child_vals in children_vals
                ]
                #preprocess the children
                for child in children:
                    child = preprocess_individual(child,difficulty)

                mut_propabilities = [
                    mutation_controll(mut_p, fitnesses,
                                      fitness_func(child),
                                      generation_fittest, generation)
                    for child in children
                ]

                mutated_vals = [
                    mutation_func(child.values, mut_prop) for child, mut_prop in zip(children, mut_propabilities)
                ]

                mutated_children = [
                    chromosome(mutated_val, 0, genome_length)
                    for mutated_val in mutated_vals
                ]

                #preprocess the mutated children
                for child in mutated_children:
                    child = preprocess_individual(child,difficulty)


                if not crowding:
                    for child in mutated_children:
                        offspring.append(child)
                else:
                    # Pair each parent with a randomly chosen offspring
                    offspring_indices = np.random.permutation(2)
                    for parent_idx, offspring_idx in zip(parents_indices, offspring_indices):
                        # Calculate the replacement probability
                        similarityval = similarity_func(elites[parent_idx], mutated_children[offspring_idx],
                                                        distance_func,
                                                        max_distance)
                        delta_fitness = fitness_func(mutated_children[offspring_idx]) - fitness_func(
                            elites[parent_idx])
                        temperature = max(T_start * (alpha ** generation), 1)
                        replacement_probability = boltzmann_replacement_probability(delta_fitness,
                                                                                    temperature) * similarityval

                        # Replace the parent with the offspring within the elites list if rand < replacement_probability
                        if np.random.rand() < replacement_probability:
                            elites[parent_idx] = mutated_children[offspring_idx]

                            # Always add the offspring to the offspring list
                        offspring.append(mutated_children[offspring_idx])
            # Update the population with the new set of elites and offspring
            species[i] = elites + offspring
        population = []
        for sp in species:
            population += sp
        population = population[:pop_size]
        loop_end = time.perf_counter()
        if aging:
            for person in population:
                person.age += 1
        fitnesses = [fitness_func(individual) for individual in
                     population]
        best_individual = max(population, key=lambda x: fitness_func(x))
        best_fitness = fitness_func(best_individual)
        mean_fitness = sum(fitnesses) / pop_size
        #        Measurments
        m1 = np.var(fitnesses)
        m2 = exploitation_ratio_calc(population, fitnesses, distance_func)
        m3 = selection_pressure_calc(selection_propabilities)
        m4 = diversification_genetic(population, distance_func, max_distance)
        m5 = taspr_calc(selection_propabilities, int(pop_size * 0.1))
        m6 = exploration_ratio_calc(population, fitnesses, distance_func, threshold)
        print(f"Loop run time: {loop_end - loop_start:.5f} | "
              f"Exploration Factor: {m6:.5f} | "
              f"Exploitation Factor: {m2:.5f} | "
              f"Fitness Variance: {m1:.5f} | "
              f"Genetic Diversity: {m4:.5f} | "
              f"Top-Average Selection Probability Ratio: {m5:.5f} | "
              f"Selection Pressure: {m3:.5f}")
        fitness_variance.append(m1)
        exploitation_ratio.append(m2)
        selection_pressure.append(m3)
        genetic_diversity.append(m4)
        taspr.append(m5)
        exploration_ratio.append(m6)
        generation_fittest.append(best_fitness)
        gen_best_ind.append(best_individual)
        avg_gen_fitness.append(mean_fitness)
        count += 1
    # Find the individual with the highest fitness
    sorted_population = sorted(population, key=lambda x: fitness_func(x), reverse=True)
    n = int(len(sorted_population) * 0.5)
    best_fitness = fitness_func(sorted_population[0])
    best_n_individuals = sorted_population[:n]
    return best_n_individuals, best_fitness, exploitation_ratio, selection_pressure, fitness_variance, genetic_diversity \
        , taspr, exploration_ratio, generation_fittest, gen_best_ind, sorted_population, avg_gen_fitness

def main():
    while True:
        fitness_functions = [easy_level_fitness,medium_level_fitness,hard_level_fitness]
        distance_functions = [hamm_dist]
        similarity_functions = [similarity]
        parent_selections = [SUS, RWS, tournament, randomz]
        crossover_functions = [single_point_crossover, two_point_crossover, uniform_crossover, PMX_crossover,
                            CX_crossover]
        mutation_functions = [swap_mutation, scramble_mutation, inversion_mutation]
        mutation_controll_functions = [constant, non_uniform_mutation_rate, adaptive_mutation_prob_individual,
                                    adaptive_mutation_prob_population, triggered_hyper_mutation]
        scalling_methods = [linear_scaling, min_max_scaling, exponential_scaling, no_scaling]
        # just ask for the population size and the number of generations and the mutation probability , and then randomly choose the rest and run the algorithm
        pop_size = int(input("Enter the population size: "))
        max_generations = int(input("Enter the number of generations: "))
        mut_p = float(input("Enter the mutation probability: "))
        difficulty = input("Enter the difficulty : 1)easy 2)medium 3)hard : ")
        if difficulty=='1':
            difficulty='easy'
        elif difficulty=='2':
            difficulty='medium'
        elif difficulty=='3':
            difficulty='hard'
        print("Start the algorithm ? (y/n)")
        start = input()
        if start == 'y':
            # Run the algorithm
            best_individuals, best_fitness, exploitation_ratio, selection_pressure, fitness_variance, genetic_diversity \
                , taspr, exploration_ratio, generation_fittest, gen_best_ind, sorted_population, avg_gen_fitness = \
                genetic_algorithm(pop_size, fitness_functions, max_generations, parent_selections, crossover_functions,
                                mutation_functions, mutation_controll_functions, scalling_methods[0], False,
                                distance_functions[0], similarity_functions[0], True, True, True,
                                mut_p, difficulty)
            print("Do you want to see the best levels ? (y/n)")
            show = input()
            if show == 'y':
                # draw the graph of the best individuals
                for individual in best_individuals:
                    encoded_rows = binary_to_encoding(individual,difficulty='easy')
                    G = construct_and_draw_graph(encoded_rows, True)
            print("Do you want to see graphs? (y/n)")
            show = input()
            if show == 'y':
                visualize_results(generation_fittest, genetic_diversity, exploration_ratio, exploitation_ratio,
                      fitness_variance, sorted_population, hamm_dist, len(best_individuals[0].values), similarity,avg_gen_fitness)
            
            
            





profiler = cProfile.Profile()
profiler.enable()

main()  # Run your function or module

profiler.disable()
stats = pstats.Stats(profiler).sort_stats('tottime')  # 'tottime' for total time
stats.print_stats()

