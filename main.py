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
def generate_random_tree(depth=3, seed_=42):
    random.seed(seed_)
    if depth == 0 or (depth > 1 and random.random() < 0.5):
        return Node(random.choice(TERMINALS))
    else:
        primitive = random.choice(PRIMITIVES)
        return Node(primitive, generate_random_tree(depth - 1), generate_random_tree(depth - 1))

# Evaluate the fitness of an expression tree
def evaluate_fitness(tree, x_values, y_values):
    predictions = [tree.evaluate(x) for x in x_values]
    return np.mean((np.array(predictions) - np.array(y_values))**2)

def main():
    # Algorithm parameters
    population_size = 100
    generations = 50
    seed_ = [20, 30, 40]
    runs = 3
    best_values = []
    GP_size = []

    for run in range(runs):
        print(f'Run {runs[run]}:')
        # Initialize the population
        population = [generate_random_tree(seed_) for _ in range(population_size)]

        # Define the input and output values for the target function
        x_values = np.linspace(-10, 10, 100)
        y_values = [target_function(x) for x in x_values]

        for generation in range(generations):
            # Evaluate the fitness of each individual in the population
            fitnesses = [evaluate_fitness(tree, x_values, y_values) for tree in population]

            # Select the best individual
            best_tree = min(zip(population, fitnesses), key=lambda x: x[1])[0]
            best_values.append(evaluate_fitness(best_tree, x_values, y_values))

            print(f"Generation {generation}: {best_tree} Fitness: {evaluate_fitness(best_tree, x_values, y_values)}")

            # Perform tournament selection to create the next generation
            


if __name__ == '__main__':
    main()



