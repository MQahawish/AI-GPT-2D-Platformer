from SimpleGA import encoding_to_binary

import os
import pickle

def load_or_create_pickle_file(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        return set()


easy_levels = 'levels_datasets\pickles\easy_levels.pickle'
easy_levels_binary = 'levels_datasets\pickles\easy_levels_binary.pickle'

medium_levels = 'levels_datasets\pickles\medium_levels.pickle'
medium_levels_binary = 'levels_datasets\pickles\medium_levels_binary.pickle'

hard_levels = 'levels_datasets\pickles\hard_levels.pickle'
hard_levels_binary = 'levels_datasets\pickles\hard_levels_binary.pickle'

easy_levels = load_or_create_pickle_file(easy_levels)

medium_levels = load_or_create_pickle_file(medium_levels)

hard_levels = load_or_create_pickle_file(hard_levels)

while True:
    # Sample code to ask user for difficulty level
    difficulty = input("Enter the difficulty level you want to prepare (E/M/H): ")


    if difficulty == 'E':
        levels = easy_levels
        short_difficulty = 'E'
        difficulty = 'easy'
    elif difficulty == 'M':
        levels = medium_levels
        short_difficulty = 'M'
        difficulty = 'medium'
    elif difficulty == 'H':
        levels = hard_levels
        short_difficulty = 'H'
        difficulty = 'hard'

    # Iterate through the loaded set of levels
    for level in levels:
        # Convert 2D level tuple to 1D string
        level_1d = ''.join(''.join(row) for row in level)

        # Convert level to its binary representation
        level_binary = encoding_to_binary(level,difficulty)

        # Get the number of rows and columns
        rows = len(level)
        cols = len(level[0])

        # Create the level descriptor string
        level_descriptor = f"{short_difficulty}{rows}{cols}{level_1d}\n"
        level_descriptor_binary = f"{short_difficulty}{rows}{cols}{level_binary}\n"

        # Append the level descriptor to the text file
        with open(rf'levels_datasets\texts\encoding\{difficulty}_levels.txt', 'a') as file:
            file.write(level_descriptor)
            
        # Append the level descriptor to the binary text file
        with open(rf'levels_datasets\texts\binary\{difficulty}_levels_binary.txt', 'a') as file:
            file.write(level_descriptor_binary)

        print("Level descriptor has been saved.")

    print("Levels and their binary representations have been saved.")





        
