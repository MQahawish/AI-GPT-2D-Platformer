import random
from SimpleGA import load_or_create_pickle_file,save_to_pickle_file

def parse_encoding(encoding):
    # Extract the dimensions
    rows = int(encoding[1])
    cols = int(encoding[2])
    
    # Extract the grid data
    grid_data = encoding[3:]
    
    # Initialize the 2D list
    grid_2d = []
    
    # Fill in the 2D list
    for i in range(0, len(grid_data), cols):
        row_data = grid_data[i:i + cols]
        grid_2d.append(list(row_data))
    #remove the last element of the list
    grid_2d.pop()
    return grid_2d


def generate_permutations(s, limit=None):
    # Locate positions of 'P' and 'N'
    pn_positions = [i for i, c in enumerate(s) if c == 'P' or c == 'N']
    pn_values = [s[i] for i in pn_positions]
    
    # Initialize the output list and a set to check for duplicates
    output = []
    seen = set()
    
    # Randomly shuffle 'P' and 'N' and apply the limit
    attempts = 0
    while len(output) < limit and attempts < limit * 10:
        random.shuffle(pn_values)
        s_list = list(s)
        
        for i, position in enumerate(pn_positions):
            s_list[position] = pn_values[i]
        
        new_str = ''.join(s_list)
        
        if new_str not in seen:
            seen.add(new_str)
            output.append(new_str)
        
        attempts += 1

    return output

easy_levels = 'levels_datasets\pickles\easy_levels.pickle'
medium_levels = 'levels_datasets\pickles\medium_levels.pickle'
hard_levels = 'levels_datasets\pickles\hard_levels.pickle'
easy_levels = load_or_create_pickle_file(easy_levels)
medium_levels = load_or_create_pickle_file(medium_levels)
hard_levels = load_or_create_pickle_file(hard_levels)

print("Which dataset you want to augment? 1) easy 2) medium 3) hard")
answer = input()
if answer == '1':
    levels=easy_levels
    difficulty = 'easy'
    short_difficulty = 'E'
elif answer == '2':
    levels=medium_levels
    difficulty = 'medium'
    short_difficulty = 'M'
elif answer == '3':
    levels=hard_levels
    difficulty = 'hard'
    short_difficulty = 'H'

new_levels = set()

for level in levels:
    print("Level encoding: ")
    # Convert 2D level tuple to 1D string
    level_1d = ''.join(''.join(row) for row in level)
        # Get the number of rows and columns
    rows = len(level)
    cols = len(level[0])
    # Create the level descriptor string
    level_descriptor = f"{short_difficulty}{rows}{cols}{level_1d}\n"
    print(level_descriptor)
    # Parse the encoding
    augmentations = generate_permutations(level_descriptor,limit=10)
    for augmentation in augmentations:
        augmentation_parsed = parse_encoding(augmentation)
        #add the augmentation to the set of levels
        new_levels.add(tuple(tuple(row) for row in augmentation_parsed))
    print("added {} augmentations of this level to the set".format(len(augmentations)))

# Update the original set with the new elements
levels.update(new_levels)

# Save the updated datasets
save_to_pickle_file(levels, f'levels_datasets\pickles\{difficulty}_levels.pickle')

        


