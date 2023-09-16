#scripts to provide information about the datasets
import os
import pickle

from SimpleGA import construct_graph,draw_graph

def load_or_create_pickle_file(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        return set()
    
easy_levels = 'easy_levels.pickle'
easy_levels_binary = 'easy_levels_binary.pickle'

medium_levels = 'medium_levels.pickle'
medium_levels_binary = 'medium_levels_binary.pickle'

hard_levels = 'hard_levels.pickle'
hard_levels_binary = 'hard_levels_binary.pickle'

easy_levels = load_or_create_pickle_file(easy_levels)
easy_levels_binary = load_or_create_pickle_file(easy_levels_binary)

medium_levels = load_or_create_pickle_file(medium_levels)
medium_levels_binary = load_or_create_pickle_file(medium_levels_binary)

hard_levels = load_or_create_pickle_file(hard_levels)
hard_levels_binary = load_or_create_pickle_file(hard_levels_binary)

#print how many levels are in each dataset
print('easy levels: ', len(easy_levels))

print('medium levels: ', len(medium_levels))

print('hard levels: ', len(hard_levels))

#print how many levels are in each dataset into the txt file datasets_info.txt but firstly clear the file
with open('datasets_info.txt', 'w') as f:
    f.write('easy levels: ' + str(len(easy_levels)) + '\n')
    f.write('medium levels: ' + str(len(medium_levels)) + '\n')
    f.write('hard levels: ' + str(len(hard_levels)) + '\n')

print("do you want to see the levels? (y/n)")
answer = input()
if answer == 'y':
    print("which dataset do you want to see? 1) easy 2) medium 3) hard")
    answer = input()
    if answer == '1':
        for level in easy_levels:
            print("Level encoding: ")
            for row in level:
                print(row)
            graph = construct_graph(level)
            draw_graph(graph)
    elif answer == '2':
        for level in medium_levels:
            print("Level encoding: ")
            for row in level:
                print(row)
            graph = construct_graph(level)
            draw_graph(graph)
    elif answer == '3':
        for level in hard_levels:
            print("Level encoding: ")
            for row in level:
                print(row)
            graph = construct_graph(level)
            draw_graph(graph)




    
