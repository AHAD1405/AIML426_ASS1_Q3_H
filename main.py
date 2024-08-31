import operator
import math
import random
import numpy as np

# Define the primitive set
PRIMITIVES = [operator.add, operator.sub, operator.mul, operator.truediv, math.sin, math.cos]
TERMINALS = ['x', 1.0, 2.0, 3.0, 4.0, 5.0]

# Define a class for the nodes in the expression tree
class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    def evaluate(self, x):
        if self.value in PRIMITIVES:
            if self.value == operator.truediv:
                try:
                    return self.value(self.left.evaluate(x), self.right.evaluate(x))
                except ZeroDivisionError:
                    return 1
            return self.value(self.left.evaluate(x), self.right.evaluate(x))
        elif self.value == 'x':
            return x
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
def generate_random_tree(depth=3):
    if depth == 0 or (depth > 1 and random.random() < 0.5):
        return Node(random.choice(TERMINALS))
    else:
        primitive = random.choice(PRIMITIVES)
        return Node(primitive, generate_random_tree(depth - 1), generate_random_tree(depth - 1))



def main():
    # Generate initial population
    population_size = 100
    generations = 50
    population = [generate_random_tree() for _ in range(population_size)]

    # Define the input and output values for the target function
    x_values = np.linspace(-10, 10, 100)
    y_values = [target_function(x) for x in x_values]

if __name__ == '__main__':
    main()



