import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Coordinates for plotting (arbitrary layout of cities)
coordinates = np.array([
    [0, 0],
    [2, 1],
    [3, 6],
    [5, 2],
    [6, 5],
    [8, 1],
    [7, 7]
])

# Distance matrix between cities
cities = np.array([
    [0, 2, 9, 10, 7, 14, 11],
    [1, 0, 6, 4, 12, 8, 10],
    [15, 7, 0, 8, 6, 9, 13],
    [6, 3, 12, 0, 9, 11, 5],
    [7, 12, 6, 9, 0, 4, 8],
    [14, 8, 9, 11, 4, 0, 6],
    [11, 10, 13, 5, 8, 6, 0]
])

# ACO parameters
num_ants = 10
num_iterations = 100
decay = 0.1
alpha = 1
beta = 2
num_cities = cities.shape[0]
pheromone = np.ones((num_cities, num_cities)) / num_cities
best_cost = float('inf')
best_path = None

def route_distance(route):
    return sum(cities[route[i - 1], route[i]] for i in range(len(route)))

def select_next_city(probabilities):
    return np.random.choice(len(probabilities), p=probabilities)

# ACO algorithm
for _ in range(num_iterations):
    all_routes = []
    all_distances = []

    for _ in range(num_ants):
        visited = [np.random.randint(num_cities)]
        while len(visited) < num_cities:
            current = visited[-1]
            unvisited = list(set(range(num_cities)) - set(visited))
            pher = np.array([pheromone[current][j] for j in unvisited])
            dist = np.array([cities[current][j] for j in unvisited])
            prob = (pher ** alpha) * ((1 / dist) ** beta)
            prob /= prob.sum()
            next_city = unvisited[select_next_city(prob)]
            visited.append(next_city)

        distance = route_distance(visited)
        all_routes.append(visited)
        all_distances.append(distance)
        if distance < best_cost:
            best_cost = distance
            best_path = visited

    pheromone *= (1 - decay)
    for route, dist in zip(all_routes, all_distances):
        for i in range(num_cities):
            a, b = route[i - 1], route[i]
            pheromone[a][b] += 1 / dist

# Plotting best path
G = nx.DiGraph()
for i in range(len(best_path)):
    G.add_edge(best_path[i - 1], best_path[i])

pos = {i: coordinates[i] for i in range(num_cities)}
plt.figure(figsize=(10, 7))
nx.draw_networkx(G, pos, node_color='skyblue', with_labels=True, edge_color='r', arrows=True)

# Annotate coordinates
for i, (x, y) in enumerate(coordinates):
    plt.text(x + 0.2, y + 0.2, f"({x},{y})", fontsize=9, color='black')

# Show axis with ticks
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title(f"Best Path: {best_path + [best_path[0]]}\nTotal Cost: {best_cost}")
plt.grid(True)
plt.axhline(0, color='black',linewidth=1)
plt.axvline(0, color='black',linewidth=1)
plt.xticks(np.arange(min(coordinates[:,0])-1, max(coordinates[:,0])+1, 1))
plt.yticks(np.arange(min(coordinates[:,1])-1, max(coordinates[:,1])+1, 1))
plt.show()
