# problem statement:
# a company makes two products, A and B.
# it costs $2 to make one unit of A and $3 for a unit of B.
# they need to produce at least 5 units total per day.
# labor rules say that units of A plus twice the units of B must be at least 6.
# find the best production amounts to minimize total cost.

# lpp formulation:
# min z = 2x1 + 3x2
# subject to: 
#   x1 + x2 >= 5
#   x1 + 2x2 >= 6
#   x1, x2 >= 0

# converting to standard form (using slack s1, s2 and artificial a1, a2):
# min z = 2x1 + 3x2 + 0s1 + 0s2 + M*a1 + M*a2
# subject to:
#   x1 + x2 - s1 + a1 = 5
#   x1 + 2x2 - s2 + a2 = 6
#   x1, x2, s1, s2, a1, a2 >= 0

import numpy as np

def solve_big_m():
    M = 10000
    c = np.array([2, 3, 0, 0, M, M, 0], dtype=float)
    
    A = np.array([
        [1, 1, -1,  0, 1, 0, 5],
        [1, 2,  0, -1, 0, 1, 6]
    ], dtype=float)
    
    basis = [4, 5]
    
    max_iters = 100
    iters = 0
    
    while iters < max_iters:
        iters += 1
        cb = c[basis]
        zj_cj = np.dot(cb, A[:, :-1]) - c[:-1]
        
        if np.all(zj_cj <= 1e-7):
            break
            
        enter_idx = np.argmax(zj_cj)
        
        ratios = []
        for i in range(len(basis)):
            if A[i, enter_idx] > 1e-7:
                ratios.append(A[i, -1] / A[i, enter_idx])
            else:
                ratios.append(float('inf'))
                
        if all(r == float('inf') for r in ratios):
            print("solution is unbounded.")
            return
            
        leave_idx = np.argmin(ratios)
        basis[leave_idx] = enter_idx
        
        pivot = A[leave_idx, enter_idx]
        A[leave_idx] = A[leave_idx] / pivot
        
        for i in range(len(basis)):
            if i != leave_idx:
                A[i] = A[i] - A[i, enter_idx] * A[leave_idx]
                
    x = np.zeros(6)
    for i, b in enumerate(basis):
        x[b] = A[i, -1]
        
    z = np.dot(c[:-1], x)
    
    print(f"optimal objective value (z): {round(z, 2)}")
    print(f"product A (x1) = {round(x[0], 2)}")
    print(f"product B (x2) = {round(x[1], 2)}")

if __name__ == "__main__":
    solve_big_m()