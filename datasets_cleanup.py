#scripts to provide information about the datasets
import os
import pickle
import networkx as nx

from SimpleGA import construct_graph,encoding_to_binary_all

def load_or_create_pickle_file(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        return set()
    
def save_to_pickle_file(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    
easy_levels = 'levels_datasets\pickles\easy_levels.pickle'
easy_levels_binary = 'levels_datasets\pickles\easy_levels_binary.pickle'

medium_levels = 'levels_datasets\pickles\medium_levels.pickle'
medium_levels_binary = 'levels_datasets\pickles\medium_levels_binary.pickle'

hard_levels = 'levels_datasets\pickles\hard_levels.pickle'
hard_levels_binary = 'levels_datasets\pickles\hard_levels_binary.pickle'

easy_levels = load_or_create_pickle_file(easy_levels)
easy_levels_binary = load_or_create_pickle_file(easy_levels_binary)

medium_levels = load_or_create_pickle_file(medium_levels)
medium_levels_binary = load_or_create_pickle_file(medium_levels_binary)

hard_levels = load_or_create_pickle_file(hard_levels)
hard_levels_binary = load_or_create_pickle_file(hard_levels_binary)

while True:
    print("Which dataset you want to clean? 1) easy 2) medium 3) hard")
    answer = input()
    if answer == '1':
        levels=easy_levels
        binary_levels=easy_levels_binary
        difficulty = 'easy'
    elif answer == '2':
        levels=medium_levels
        binary_levels=medium_levels_binary
        difficulty = 'medium'
    elif answer == '3':
        levels=hard_levels
        binary_levels=hard_levels_binary
        difficulty = 'hard'

    items_to_remove = set()
    binaries_to_remove = set()

    for level in levels:
        print("Level encoding: ")
        for row in level:
            print(row)
        graph = construct_graph(level)
        
        S = [node for node in graph.nodes if graph.nodes[node]['color'] == 'green']
        T = [node for node in graph.nodes if graph.nodes[node]['color'] == 'red']
        
        try:
            path = nx.shortest_path(graph, S[0], T[0])
        except:
            print("No path from S to T")
            items_to_remove.add(level)
            
            possible_binaries = encoding_to_binary_all(level, difficulty)
            
            for binary in possible_binaries:
                binaries_to_remove.add(binary)

    # Removing items after the loop
    for item in items_to_remove:
        levels.remove(item)
        print("Level deleted")

    for binary in binaries_to_remove:
        if binary in binary_levels:
            binary_levels.remove(binary)
            print("Binary representation deleted")

    # Save the updated datasets
    save_to_pickle_file(levels, f'levels_datasets\pickles\{difficulty}_levels.pickle')
    save_to_pickle_file(binary_levels, f'levels_datasets\pickles\{difficulty}_levels_binary.pickle')