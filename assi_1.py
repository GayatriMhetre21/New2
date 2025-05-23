import numpy as np

def union(A, B): 
    return np.maximum(A, B)

def intersection(A, B): 
    return np.minimum(A, B)

def complement(A): 
    return 1 - A

def difference(A, B): 
    return np.maximum(A - B, 0)

def cartesian_product(A, B): 
    return np.outer(A, B)

def max_min_composition(R1, R2): 
    # Perform max-min composition
    return np.array([[np.max(np.minimum(R1[i, :], R2[:, j])) for j in range(R2.shape[1])] for i in range(R1.shape[0])])

# Sample fuzzy sets
A = np.array([0.2, 0.4, 0.7, 0.8]) 
B = np.array([0.1, 0.8, 0.2, 0.3]) 

# Cartesian products
R1 = cartesian_product(A, B) 
R2 = cartesian_product(B, A) 

# Max-min composition
result = max_min_composition(R1, R2)

# Outputs
print("Union of A and B:", union(A, B)) 
print("Intersection of A and B:", intersection(A, B)) 
print("Complement of A:", complement(A)) 
print("Difference of A and B:", difference(A, B)) 
print("Cartesian product of A and B:\n", R1) 
print("Cartesian product of B and A:\n", R2) 
print("Max-min composition of R1 and R2:\n", result)
