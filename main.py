import operator
import math
import random
import numpy as np

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
def generate_random_tree(depth=None, seed_=42):
    random.seed(seed_)
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
    selected = random.sample(list(zip(population, fitnesses)), k)
    selected.sort(key=lambda x: x[1])
    return selected[0][0]

# Perform crossover
def crossover(tree1, tree2):
    if random.random() < 0.5:
        return tree1  # No crossover, return the first parent
    if random.random() < 0.5:
        return tree2  # No crossover, return the second parent

    # Randomly select a subtree from both parents
    if random.random() < 0.5 and tree1.left and tree2.left:
        tree1.left, tree2.left = tree2.left, tree1.left
    elif tree1.right and tree2.right:
        tree1.right, tree2.right = tree2.right, tree1.right

    return tree1

# Perform mutation
def mutate(tree, mutation_rate=0.1):
    if random.random() < mutation_rate:
        if isinstance(tree, Node) and not isinstance(tree.value, (float, str)):
            # Replace function nodes with a new random function
            return generate_random_tree(depth=1)
        else:
            # Replace terminal nodes with a new random terminal
            return Node(random.choice(TERMINALS))
    else:
        if tree.left:
            tree.left = mutate(tree.left, mutation_rate)
        if tree.right:
            tree.right = mutate(tree.right, mutation_rate)
        return tree

# Function to count the number of nodes in a tree
def count_nodes(tree):
    if tree is None:
        return 0
    return 1 + count_nodes(tree.left) + count_nodes(tree.right)

def main():
    # Algorithm parameters
    population_size = 100
    generations = 10
    seed_ = [20, 30, 40]
    runs = 3
    best_values = []
    GP_size = []

    for run in range(runs):
        print(f'Run {run}:')
        # Initialize the population
        population = [generate_random_tree(3,seed_[run]) for _ in range(population_size)]

        # Define the input and output values 
        x_values = np.linspace(-10, 10, population_size)  # initial x values
        y_values = [target_function(x) for x in x_values] # pass the x values to the target function

        for generation in range(generations):

            # Evaluate the fitness of each individual in the population
            fitnesses = [evaluate_fitness(tree, x_values, y_values) for tree in population]

            #print(f"Generation {generation}: {best_tree} Fitness: {evaluate_fitness(best_tree, x_values, y_values)}")

            # Perform tournament selection to create the next generation
            # Select the next generation
            new_population = []
            for _ in range(population_size // 2):
                parent1 = tournament_selection(population, fitnesses)
                parent2 = tournament_selection(population, fitnesses)
                child1 = crossover(parent1, parent2)
                child2 = crossover(parent2, parent1)
                child1 = mutate(child1)
                child2 = mutate(child2)
                #new_population.extend([mutate(child1), mutate(child2)])
                new_population.append(child1)
                new_population.append(child2)

            population = new_population
        
        # Print the best solution
        # Select the best individual
        tree_fitnesses = [evaluate_fitness(tree, x_values, y_values) for tree in population]

        best_tree = min(zip(population, tree_fitnesses), key=lambda x: x[1])[0]
        best_values.append(evaluate_fitness(best_tree, x_values, y_values))
        GP_size.append(count_nodes(best_tree))
        #best_tree = population[np.argmin(fitnesses)]
        
        print(f"Best Solution: {best_tree}")


if __name__ == '__main__':
    main()



