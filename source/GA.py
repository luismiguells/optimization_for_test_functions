#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 22:00:13 2020

@author: luismiguells
"""

"""
Test functions for optimization
Ackley function
-20*np.exp(-0.2*np.sqrt(0.5*(x**2 + y**2))) - np.exp(0.5*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y))) + np.exp(1) + 20

Beale function
(1.5-x+(x*y))**2 + (2.25*-x+(x*y**2))**2 + (2.625-x+(x*y**3))**2

Booth function
(x+(2*y)-7)**2 + ((2*x)+y-5)**2   

Himmelblau's function
(x**2+y-11)**2 + (x+y**2-7)**2

McCormick function
np.sin(x+y) + (x+y)**2 - 1.5*x + 2.5*y + 1

More in: https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""

import random
import numpy as np
import matplotlib.pyplot as plt


def generate_population(size, upper_limit, lower_limit):
    """
    Parameters
    ----------
    size : Size of the population.
    upper_limit : Upper limit where the function exist.
    lower_limit : Lower limit where the function exist.

    Returns
    -------
    population : An array that contains the population
    """
    population = np.array([np.array([random.uniform(upper_limit, lower_limit),
                                     random.uniform(upper_limit, lower_limit)]) for _ in range(size)])
    
    return population

def objective_function(individual):
    """
    Parameters
    ----------
    individual : An individual that contains X and Y values.

    Returns
    -------
    The result of the individual evaluated in the objective function
    """
    x = individual[0]
    y = individual[1]
    
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2 # Change for the desire function


def choice_by_roulette(sorted_population, objective_sum):
    """
    Parameters
    ----------
    sorted_population : Sorted population from the best to the worst.
    objective_sum : Sum of all the results evaluated of the individuals.

    Returns
    -------
    individual : The best individual.
    """
    # Choose a random number between 0 and 1
    number = random.uniform(0, 1) 

    sub_sum = 0
    
    # Calculate the probability of each individual and add it until the 
    # the cumulative probability is greater than the number chosen at random
    for individual in sorted_population:
        probability = 1.0 - (objective_function(individual) /objective_sum)
        sub_sum += probability

        if number <= sub_sum:
            return individual

def sort_population_by_function(population):
    """
    Parameters
    ----------
    population : The entire population

    Returns
    -------
    The population sorted from the best to the worst.
    """
    return sorted(population, key=objective_function)

def crossover(individual_a, individual_b):
    """
    Parameters
    ----------
    individual_a : The best individual from the first selection
    individual_b : The best individual from the second selection

    Returns
    -------
    A new individual.

    """
    xa = individual_a[0]
    ya = individual_a[1]

    xb = individual_b[0]
    yb = individual_b[1]

    return np.array([(xa+xb)/2, (ya+yb)/2])


def mutate(individual, upper_limit, lower_limit):
    """
    Parameters
    ----------
    individual : The individual 
    a : Upper limit where the function exist.
    b : Lower limit where the function exist.

    Returns
    -------
    A new mutated individual
    """
    
    # Add a random number to individual
    next_x = individual[0] + random.uniform(-0.05, 0.05)
    next_y = individual[1] + random.uniform(-0.05, 0.05)

    # Guarantee the individual keep inside limits
    next_x = min(max(next_x, upper_limit), lower_limit)
    next_y = min(max(next_y, upper_limit), lower_limit)

    return np.array([next_x, next_y])


def make_next_generation(previous_population, population_size, upper_limit, lower_limit):
    """
    Parameters
    ----------
    previous_population : Previous population generated.
    population_size : Size of the population.
    upper_limit : Upper limit where the function exist.
    lower_limit : Lower limit where the function exist.

    Returns
    -------
    next_generation : A new generation.
    """
    
    next_generation = np.zeros((population_size, 2))
    sorted_by_fitness_population = sort_population_by_function(previous_population)
    
    # Sum of the results of the individuals
    objective_sum = sum(objective_function(individual) for individual in population)

    for i in range(population_size):
        first_individual = choice_by_roulette(sorted_by_fitness_population, objective_sum)
        second_individual = choice_by_roulette(sorted_by_fitness_population, objective_sum)

        individual = crossover(first_individual, second_individual)
        individual = mutate(individual, upper_limit, lower_limit)
        next_generation[i] = individual

    return next_generation



# Initialization
generations = 1000
population_size = 40
upper_limit = -5
lower_limit = 5
gen_ite = 1
Y = 0 # Target value
epsilon = 1e-6


# Initialize the population
population = generate_population(population_size, upper_limit, lower_limit)

# Begin the algorithm
while gen_ite < generations:
    #print("Generation: ", gen_ite)

    population = make_next_generation(population, population_size, upper_limit, lower_limit)
    best_individual = sort_population_by_function(population)[0]
    
    error = abs(objective_function(best_individual) - Y)
    
    if(error < epsilon):
        break

    gen_ite += 1 

print("The minimum value is in (X, Y):", best_individual[0], best_individual[1], "in", gen_ite, "generations")


# Set the values for the plot the figure
ax = plt.axes(projection='3d')
x = np.linspace(upper_limit, lower_limit, 100)
y = np.linspace(upper_limit, lower_limit, 100)
X, Y = np.meshgrid(x, y)

points = np.stack([X, Y])
Z = objective_function(points)

# Plot the figure
ax.plot_wireframe(X, Y, Z, linewidth=0.2)

# Plot the minimum value
ax.scatter(best_individual[0], best_individual[1], objective_function(best_individual), 'o', color='red')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()
