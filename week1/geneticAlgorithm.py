# -*- coding: utf-8 -*-
"""
    https://colab.research.google.com/drive/1Id7mExyxe6YCkvXQpglhP-NFX8MTroV-
"""

import numpy as np
from numpy.core.numeric import indices
from copy import deepcopy

x = np.array([1,2,3,4,5])

def sphere_function(x):
  return sum(x**2)

sphere_function(x)

class Problem:
  def __init__(self):
    self.number_of_genes = 8
    self.min_gene_value = -10
    self.max_gene_value = 10
    self.cost_function = sphere_function

p = Problem()

p.number_of_genes

y = np.array([-1,3,-5,8,3,9,2,0])

p.cost_function(y)

sphere_function(y)

class Individual:
  chromosone = None
  def __init__(self, prob):
    prob.number_of_genes
    self.chromosone = np.random.uniform(prob.min_gene_value, prob.max_gene_value, prob.number_of_genes)
    self.cost = prob.cost_function(self.chromosone)


np.random.uniform(-10,10,5)

i1 = Individual(p)

i1.cost

i2 = Individual(p)

i2.cost

class Parameters:
  def __init__(self):
    self.number_in_population = 1000
    self.number_of_generations = 500
    self.child_rate = 0.5

para = Parameters()

def choose_diff_indices(max_value):
  index1 = np.random.randint(0, max_value)
  index2 = np.random.randint(0, max_value)
  if index1 == index2:
    return choose_diff_indices(max_value)
  else:
    return index1, index2

choose_diff_indices(4)

def run_genetic(prob, params):
  # read the problem
  cost_function = prob.cost_function

  # read parameters
  number_in_population = params.number_in_population
  number_of_children = params.child_rate * number_in_population

  # initialise the population
  best_solution = Individual(prob)
  best_solution.cost = 999999

  population = []
  for i in range(number_in_population):
    new_individual = Individual(prob)
    population.append(new_individual)
    if new_individual.cost < best_solution.cost:
      best_solution = deepcopy(new_individual)


  # generate children
  children = []
  while len(children) < number_of_children:

     # choose parents
    parent1_index, parent2_index = choose_diff_indices(len(population))
    parent1 = population[parent1_index]
    parent2 = population[parent2_index]
     # cost of chidren

  return population, best_solution

pop, best_sol = run_genetic(p, para)

pop[445].chromosone
pop[445].cost

best_sol.cost



