# -*- coding: utf-8 -*-
"""GeneticAlgo2023.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YLu2o85wraL9_dsLJRzrqB-MOzGxEyRv
"""

import numpy as np
from copy import deepcopy

x= np.array([1,2,3,4,5])

x

def sphere_function(x):
  return sum(x**2)

sphere_function(x)

class problem:
  def __init__(self):
    self.number_of_genes = 8
    self.min_gene_value = -10
    self.max_gene_value = 10
    self.cost_function = sphere_function

p = problem()

x=np.array([-1,3,-5,8,3,9,2,0])

p.cost_function(x)

sphere_function(x)

class individual:
  chromosone = None
  def __init__(self, prob):
    #Create a random individual.
    self.chromosone = np.random.uniform(prob.min_gene_value,prob.max_gene_value,prob.number_of_genes)
    self.cost = prob.cost_function(self.chromosone)

np.random.uniform(-10,10,8)

class parameters:
  def __init__(self):
    self.number_in_population = 1000
    self.number_of_generations = 500
    self.child_rate = 0.5

def run_genetic(prob, params):
  #  read the problem
  cost_function = prob.cost_function

  #   read parameters
  number_in_population = params.number_in_population
  number_of_children = params.child_rate * number_in_population

  #  Initialise the population
  best_solution = individual(prob)
  best_solution.cost = 999999

  population = []
  for i in range(number_in_population):
    new_individual = individual(prob)
    population.append(new_individual)
    if new_individual.cost < best_solution.cost:
      best_solution = deepcopy(new_individual)  # copy new_individual


# loop  

  #generate children
  children = []
  while len(children) < number_of_children:


    # choose parents
    parent1_index, parent2_index = choose_different_indices(len(population))
    parent1 = population[parent1_index]
    parent2 = population[parent2_index]
    



    #  cost the children


  # add children to popuation

  # sort and cull population



  return population, best_solution

para = parameters()

para

pop, best_sol = run_genetic(p,para)

pop[43]

pop[43]

i2.cost

i1.chromosone

def choose_different_indices(max_value):
  index1 = np.random.randint(0,max_value)
  index2 = np.random.randint(0,max_value)
  if index1 == index2:
    return choose_different_indices(max_value)
  return index1, index2

pop[1000]

for i in range(200):
  i1, i2 = choose_different_indices(20)
  print("First" + str(i1) + "Second" + str(i2))

