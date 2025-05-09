import random
from deap import base, creator, tools, algorithms

# Evaluation function (to minimize)
def eval_func(individual): 
    return sum(x ** 2 for x in individual),  # Minimize sum of squares

# Avoid creator errors if re-running
try:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
except RuntimeError:
    pass

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -5.0, 5.0)  # Values between -5 and 5
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=3)  # 3D individual
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", eval_func)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Initialize population
population = toolbox.population(n=50)
generations = 20

for gen in range(generations): 
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
    fits = toolbox.map(toolbox.evaluate, offspring)
    
    for fit, ind in zip(fits, offspring): 
        ind.fitness.values = fit
    
    population = toolbox.select(offspring, k=len(population))
    
    best_ind = tools.selBest(population, k=1)[0]
    best_fitness = best_ind.fitness.values[0]
    print(f"Generation {gen+1}: Best Fitness = {best_fitness:.5f}, Individual = {best_ind}")

# Final result
print("\nFinal Best Individual:", best_ind)
print("Final Best Fitness:", best_fitness)
