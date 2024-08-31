import operator
import math
import random
import numpy as np
import pandas as pd

# Define the primitive set
PRIMITIVES = [operator.add, operator.sub, operator.mul, operator.truediv]
UNARY_PRIMITIVES = [math.sin, math.cos]
TERMINALS = ['x', 1.0, 2.0, 3.0, 4.0, 5.0]

# Define a class for the nodes in the expression tree
class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    def evaluate(self, x):
        # this to handle binary functions like add, sub, mul, div
        if self.value in PRIMITIVES:
            if self.value == operator.truediv:
                try:
                    return self.value(self.left.evaluate(x), self.right.evaluate(x))
                except ZeroDivisionError:
                    return 1
            return self.value(self.left.evaluate(x), self.right.evaluate(x))
        
        # this to handle unary functions like sin, cos
        elif self.value in UNARY_PRIMITIVES:
            return self.value(self.left.evaluate(x))
        
        # this to handle the terminal values like x, 1.0, 2.0, 3.0, 4.0, 5.0
        elif self.value == 'x':
            return x
        # this to handle the constant values like 1.0, 2.0, 3.0, 4.0, 5.0
        else:
            return self.value

    def __str__(self):
        if self.value in PRIMITIVES:
            return f"({self.left} {self.value.__name__} {self.right})"
        else:
            return str(self.value)

# Define the target function
def target_function(x):
    if x > 0:
        return 1 / x + math.sin(x)
    else:
        return 2 * x + x**2 + 3.0

# Generate a random expression tree
def generate_random_tree(depth=None):
    depth = depth if depth is not None else random.randint(1, 5)  # Randomize depth
    if depth == 0 or (depth > 1 and random.random() < 0.5):
        return Node(random.choice(TERMINALS))
    else:
        primitive = random.choice(PRIMITIVES)
        return Node(primitive, generate_random_tree(depth - 1), generate_random_tree(depth - 1))


# Evaluate the fitness of an expression tree
def evaluate_fitness(tree, x_values, y_values):
    predictions = [tree.evaluate(x) for x in x_values]
    return np.mean((np.array(predictions) - np.array(y_values))**2)

# Perform tournament selection
def tournament_selection(population, fitnesses, k=3):
    k = min(k, len(population))
    selected = random.sample(list(zip(population, fitnesses)), k)
    selected.sort(key=lambda x: x[1])
    weights = [1 / (i + 1) for i in range(k)]
    selected_index = random.choices(range(k), weights=weights, k=1)[0]
    return selected[selected_index][0]

# Perform crossover
def crossover(tree1, tree2, max_depth=5):
    if tree1 is None or tree2 is None:
        return tree1 if tree2 is None else tree2

    if random.random() < 0.5:
        left = crossover(tree1.left, tree2.left, max_depth) if tree1.left and tree2.left else tree1.left or tree2.left
        new_tree = Node(tree1.value, left, tree1.right)
    else:
        right = crossover(tree1.right, tree2.right, max_depth) if tree1.right and tree2.right else tree1.right or tree2.right
        new_tree = Node(tree1.value, tree1.left, right)

    return ensure_max_depth(new_tree, max_depth)

# Perform mutation with depth check
def mutate(tree, mutation_rate=0.1, max_depth=5):
    if random.random() < mutation_rate:
        return generate_random_tree(max_depth)
    else:
        if tree.left:
            tree.left = mutate(tree.left, mutation_rate, max_depth)
        if tree.right:
            tree.right = mutate(tree.right, mutation_rate, max_depth)
        return ensure_max_depth(tree, max_depth)

# Function to calculate the depth of a tree
def calculate_depth(tree):
    if tree is None:
        return 0
    return 1 + max(calculate_depth(tree.left), calculate_depth(tree.right))

# Function to ensure the tree does not exceed the maximum depth
def ensure_max_depth(tree, max_depth):
    if calculate_depth(tree) > max_depth:
        return generate_random_tree(max_depth)
    return tree

# Function to count the number of nodes in a tree
def count_nodes(tree):
    if tree is None:
        return 0
    return 1 + count_nodes(tree.left) + count_nodes(tree.right)

def print_summary(fitness_values, programe_size, mean_, std_):
    """
        Creates a pandas DataFrame table with the given data.
    """
    # First column
    first_column = ['Run 1','Run 2','Run 3']

    avg_best_values = np.mean(fitness_values)
    avg_programe_size = np.mean(programe_size)
    std_best_values = np.std(fitness_values)
    std_programe_size = np.std(programe_size)

    # Create a dictionary with the two lists as values
    data = {'': first_column, 'Fitness Value': fitness_values, 'Programe Size': programe_size}

    # Create a pandas DataFrame from the dictionary
    data_table = pd.DataFrame(data)

    # Create a new DataFrame with the mean and concatenate it with (data_table)
    mean_row = pd.DataFrame({'': ['Mean'], 'Fitness Value': [avg_best_values], 'Programe Size': [avg_programe_size]})
    data_table = pd.concat([data_table, mean_row], ignore_index=True)

    # Create a new DataFrame with the stander deviation and concatenate it with (data_table)
    std_row = pd.DataFrame({'': ['STD'], 'Fitness Value': [std_best_values], 'Programe Size': [std_programe_size]})
    data_table = pd.concat([data_table, std_row], ignore_index=True)

    return data_table

def main():
    # Algorithm parameters
    population_size = 100
    generations = 50
    seed_ = [20, 30, 40]
    runs = 3
    best_values = []
    GP_size = []

    for run in range(runs):
        print(f'Run {run}:')
        # Initialize the population
        population = [generate_random_tree() for _ in range(population_size)]

        # Define the input and output values 
        x_values = np.linspace(-10, 10, population_size)  # initial x values
        y_values = [target_function(x) for x in x_values] # pass the x values to the target function

        for generation in range(generations):

            # Evaluate the fitness of each individual in the population
            fitnesses = [evaluate_fitness(tree, x_values, y_values) for tree in population]

            # Apply Elitism
            elite_size = 1  # Number of top individuals to carry over
            sorted_population = sorted(zip(population, fitnesses), key=lambda x: x[1])
            elites = [ind for ind, _ in sorted_population[:elite_size]]
            
            # Number of individuals to be created excluding elites
            num_individuals_to_create = population_size - elite_size

            # Perform tournament selection to create the next generation
            # Select the next generation
            new_population = []
            for _ in range(num_individuals_to_create // 2):
                parent1 = tournament_selection(population, fitnesses)
                parent2 = tournament_selection(population, fitnesses)
                child1 = crossover(parent1, parent2)
                child2 = crossover(parent2, parent1)
                child1 = mutate(child1)
                child2 = mutate(child2)
                #new_population.extend([mutate(child1), mutate(child2)])
                new_population.append(child1)
                new_population.append(child2)

            new_population.extend(elites)  # Add the elites to the new population   
            population = new_population
        
        # Print the best solution
        # Select the best individual
        tree_fitnesses = [evaluate_fitness(tree, x_values, y_values) for tree in population]

        best_tree = min(zip(population, tree_fitnesses), key=lambda x: x[1])[0]
        best_values.append(round(evaluate_fitness(best_tree, x_values, y_values),4))
        GP_size.append(round(count_nodes(best_tree),4))
        #best_tree = population[np.argmin(fitnesses)]
        
        print(f"Fitness vlue: {best_values[run]}")
        print(f'Programe size: {GP_size[run]}')
    
    # Claculate the average fitness value
    summary = print_summary(best_values, GP_size, np.mean(best_values), np.std(best_values))
    print(summary)






if __name__ == '__main__':
    main()



