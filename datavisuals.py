import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import random


def visualize_results(generational_fittest, genetic_diversity, exploration_ratio, exploitation_ratio,
                      fintess_variance, sorted_population, dist_func, max_distance, similarity_func,avg_gen_fittest,
                      name="",dim=None):
    generations = [i for i in range(len(generational_fittest))]
    # convergence_rates = [0] + [abs(generational_fittest[i] - generational_fittest[i - 1])
    #                            for i in range(1, len(generational_fittest))]
    x_vals_list = [generations] * 6
    y_vals_list = [generational_fittest, avg_gen_fittest, genetic_diversity, exploration_ratio, exploitation_ratio,
                   fintess_variance]
    x_labels = ["Generation"] * 6
    y_labels = ["Best Fitness", "Average Fitness", "Genetic Diversity", "Exploration", "Exploitation",
                "Fitness Variance"]

    window_size = 5  # Set the sliding window size

    # Create subplots
    n_plots = len(x_vals_list)
    n_cols = 2
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 5 * n_rows))
    fig.subplots_adjust(hspace=0.7, wspace=0.5)
    fig.suptitle(name, fontsize=20)

    for i, ax in enumerate(axes.flat[:n_plots]):
        y_vals_rolling = pd.Series(y_vals_list[i]).rolling(window=window_size).mean()
        ax.plot(x_vals_list[i], y_vals_rolling)
        ax.set_xlabel(x_labels[i])
        ax.set_ylabel(y_labels[i])
        ax.set_title(y_labels[i])

    for i in range(n_plots, n_rows * n_cols):
        axes.flat[i].set_visible(False)

    # Create the heatmap
    similarity_matrix = np.zeros((len(sorted_population), len(sorted_population)))
    max_dist = 0
    for i in range(len(sorted_population)):
        for j in range(len(sorted_population)):
            dist = dist_func(sorted_population[i], sorted_population[j],dim)
            max_dist = max(max_dist, dist)
            similarity_matrix[i, j] = similarity_func(sorted_population[i], sorted_population[j], dist_func,
                                                        max_distance)

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(similarity_matrix, cmap="coolwarm", linewidths=.5, ax=ax)
    ax.set_xlabel("Individuals")
    ax.set_ylabel("Individuals")
    ax.set_title("Similarity Heatmap")
    plt.show()


def visualize_results_multi_island(island_results_list, dist_func, max_distance, similarity_func):
    n_islands = len(island_results_list)

    window_size = 5  # Set the sliding window size

    n_plots = 6
    n_cols = 2 * n_islands
    n_rows = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 5 * n_rows))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    x_labels = ["Generation"] * 6
    y_labels = ["Best Fitness", "Convergence Rate", "Genetic Diversity", "Exploration", "Exploitation",
                "Fitness Variance"]

    for j, island_results in enumerate(island_results_list):
        generational_fittest = island_results[8]
        genetic_diversity = island_results[5]
        exploration_ratio = island_results[7]
        exploitation_ratio = island_results[2]
        fitness_variance = island_results[4]
        sorted_population = island_results[10]

        generations = [i for i in range(len(generational_fittest))]
        epsilon = 1e-8
        convergence_rates = [0] + [
            abs(generational_fittest[i] - generational_fittest[i - 1]) / (generational_fittest[i - 1]+epsilon)
            for i in range(1, len(generational_fittest))]

        # Choose a random color for this island's graphs
        color = tuple(random.uniform(0, 1) for _ in range(3))
        y_vals_list = [generational_fittest, convergence_rates, genetic_diversity, exploration_ratio,
                       exploitation_ratio, fitness_variance]

        for i in range(n_plots):
            ax = axes[i % n_rows, j * 2 + i // n_rows]
            y_vals_rolling = pd.Series(y_vals_list[i]).rolling(window=window_size).mean()
            ax.plot(generations, y_vals_rolling, color=color)
            ax.set_xlabel(x_labels[i])
            ax.set_ylabel(y_labels[i])
            if i == 0:
                ax.set_title(f"Island {j + 1}")

    # Create the heatmaps
    heatmap_axes = plt.subplots(1, n_islands, figsize=(10 * n_islands, 10))[1]
    for j, island_results in enumerate(island_results_list):
        sorted_population = island_results[10]
        similarity_matrix = np.zeros((len(sorted_population), len(sorted_population)))
        max_dist = 0
        for i in range(len(sorted_population)):
            for k in range(len(sorted_population)):
                dist = dist_func(sorted_population[i], sorted_population[k])
                max_dist = max(max_dist, dist)
                similarity_matrix[i, k] = similarity_func(sorted_population[i], sorted_population[k], dist_func,
                                                          max_distance)

        ax = heatmap_axes[j]
        sns.heatmap(similarity_matrix, cmap="coolwarm", linewidths=.5, ax=ax)
        ax.set_xlabel("Individuals")
        ax.set_ylabel("Individuals")
        ax.set_title(f"Island {j + 1} - Similarity Heatmap")

    plt.show()


def scatter_plot(data, x_label='X values', y_label='Y values', title='Scatter Plot of Data', x_range=(-100,100), y_range=(-100,100)):
    # Extract x and y values from the data list
    x_values = [pair[0] for pair in data]
    y_values = [pair[1] for pair in data]

    # Plot the data as a scatter plot
    plt.scatter(x_values, y_values)

    # Add axis labels and a title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    # Set x and y-axis limits if specified
    if x_range:
        plt.xlim(x_range)
    if y_range:
        plt.ylim(y_range)

    # Display the plot
    plt.show()


def plot_histogram(point_pairs):
    # Separate x and y values
    x_values = [pair[0] for pair in point_pairs]
    y_values = [pair[1] for pair in point_pairs]

    # Create the bar plot with a specified width
    plt.bar(x_values, y_values, width=0.8, tick_label=x_values)

    # Set labels and title
    plt.xlabel('Individuals')
    plt.ylabel('Fitness')
    plt.title('Fitness of the best Individuals')

    # Show the plot
    plt.show()


def scatter_plot_multiple(datasets, titles, x_label='X values', y_label='Y values', x_range=(-100, 100), y_range=(-100, 100)):
    fig, axes = plt.subplots(nrows=1, ncols=len(datasets), figsize=(len(datasets)*5, 5))
    for i, data in enumerate(datasets):
        # Extract x and y values from the data list
        x_values = [pair[0] for pair in data]
        y_values = [pair[1] for pair in data]

        # Plot the data as a scatter plot
        axes[i].scatter(x_values, y_values)

        # Add axis labels and a title
        axes[i].set_xlabel(x_label)
        axes[i].set_ylabel(y_label)
        axes[i].set_title(titles[i])

        # Set x and y-axis limits if specified
        if x_range:
            axes[i].set_xlim(x_range)
        if y_range:
            axes[i].set_ylim(y_range)

        # Set aspect ratio to 1 for square subplots
        axes[i].set_aspect('equal')

    plt.tight_layout()
    plt.show()