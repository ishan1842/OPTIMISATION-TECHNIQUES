# transportation problem:
# a firm has 3 factories (f1, f2, f3) supplying 4 warehouses (w1, w2, w3, w4).
# daily factory supply capacities: 7, 9, 18.
# daily warehouse demands: 5, 8, 7, 14.
# find the initial basic feasible solution and total cost using vam.

# formulation:
# min z = 19x11 + 30x12 + 50x13 + 10x14 
#       + 70x21 + 30x22 + 40x23 + 60x24 
#       + 40x31 +  8x32 + 70x33 + 20x34
# subject to constraints:
#   sum of x1j = 7,  sum of x2j = 9,  sum of x3j = 18  (supply limits)
#   sum of xi1 = 5,  sum of xi2 = 8,  sum of xi3 = 7, sum of xi4 = 14 (demand needs)
#   xij >= 0 for all i, j

import numpy as np

def calculate_penalty(array_1d):
    valid_elements = [x for x in array_1d if x != float('inf')]
    if len(valid_elements) >= 2:
        sorted_elements = sorted(valid_elements)
        return sorted_elements[1] - sorted_elements[0]
    elif len(valid_elements) == 1:
        return valid_elements[0]
    return -1

def solve_vam():
    costs = np.array([
        [19, 30, 50, 10],
        [70, 30, 40, 60],
        [40,  8, 70, 20]
    ], dtype=float)
    
    supply = np.array([7, 9, 18])
    demand = np.array([5, 8, 7, 14])
    
    allocations = np.zeros_like(costs)
    active_costs = costs.copy()
    
    while np.sum(supply) > 0 and np.sum(demand) > 0:
        row_penalties = []
        for i in range(active_costs.shape[0]):
            if supply[i] > 0:
                row_penalties.append((i, calculate_penalty(active_costs[i, :])))
                
        col_penalties = []
        for j in range(active_costs.shape[1]):
            if demand[j] > 0:
                col_penalties.append((j, calculate_penalty(active_costs[:, j])))
                
        max_penalty = -1
        is_row = True
        target_idx = -1
        
        for i, p in row_penalties:
            if p > max_penalty:
                max_penalty = p
                is_row = True
                target_idx = i
                
        for j, p in col_penalties:
            if p > max_penalty:
                max_penalty = p
                is_row = False
                target_idx = j
                
        if is_row:
            valid_cols = [(j, active_costs[target_idx, j]) for j in range(active_costs.shape[1]) if demand[j] > 0]
            min_j = min(valid_cols, key=lambda x: x[1])[0]
            allocation_amount = min(supply[target_idx], demand[min_j])
            allocations[target_idx, min_j] = allocation_amount
            supply[target_idx] -= allocation_amount
            demand[min_j] -= allocation_amount
            
            if supply[target_idx] == 0:
                active_costs[target_idx, :] = float('inf')
            if demand[min_j] == 0:
                active_costs[:, min_j] = float('inf')
        else:
            valid_rows = [(i, active_costs[i, target_idx]) for i in range(active_costs.shape[0]) if supply[i] > 0]
            min_i = min(valid_rows, key=lambda x: x[1])[0]
            allocation_amount = min(supply[min_i], demand[target_idx])
            allocations[min_i, target_idx] = allocation_amount
            supply[min_i] -= allocation_amount
            demand[target_idx] -= allocation_amount
            
            if supply[min_i] == 0:
                active_costs[min_i, :] = float('inf')
            if demand[target_idx] == 0:
                active_costs[:, target_idx] = float('inf')
                
    total_cost = np.sum(allocations * costs)
    
    print("vam allocation matrix:")
    print(allocations)
    print(f"initial feasible transportation cost: {total_cost}")

if __name__ == "__main__":
    solve_vam()