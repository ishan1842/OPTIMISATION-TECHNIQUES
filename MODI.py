# transportation problem setup:
# 3 factories (f1, f2, f3) supply 4 warehouses (w1, w2, w3, w4).
# supply available: 7, 9, 18.
# demand required: 5, 8, 7, 14.
# use an initial bfs then apply modi to optimize the shipping routes.

# mathematical formulation:
# min z = 19x11 + 30x12 + 50x13 + 10x14 
#       + 70x21 + 30x22 + 40x23 + 60x24 
#       + 40x31 +  8x32 + 70x33 + 20x34
# sub to:
#   sum of x1j = 7,  sum of x2j = 9,  sum of x3j = 18
#   sum of xi1 = 5,  sum of xi2 = 8,  sum of xi3 = 7, sum of xi4 = 14
#   xij >= 0

import numpy as np

def locate_cycle(basic_cells, start_cell):
    path_nodes = [start_cell]
    
    def traverse(current_cell, check_horizontal):
        if len(path_nodes) >= 4 and path_nodes[-1] == start_cell and len(path_nodes) % 2 == 1:
            return True
            
        for cell in basic_cells + [start_cell]:
            if cell not in path_nodes or (cell == start_cell and len(path_nodes) >= 3):
                if check_horizontal and cell[0] == current_cell[0] and cell[1] != current_cell[1]:
                    path_nodes.append(cell)
                    if traverse(cell, not check_horizontal):
                        return True
                    path_nodes.pop()
                elif not check_horizontal and cell[1] == current_cell[1] and cell[0] != current_cell[0]:
                    path_nodes.append(cell)
                    if traverse(cell, not check_horizontal):
                        return True
                    path_nodes.pop()
        return False

    traverse(start_cell, True)
    if len(path_nodes) <= 1:
        traverse(start_cell, False)
        
    return path_nodes[:-1] if len(path_nodes) > 1 else []

def solve_modi():
    costs = np.array([
        [19, 30, 50, 10],
        [70, 30, 40, 60],
        [40,  8, 70, 20]
    ], dtype=float)
    
    s = [7, 9, 18]
    d = [5, 8, 7, 14]
    
    alloc = np.zeros_like(costs)
    i, j = 0, 0
    basic_cells = []
    
    # get initial bfs using northwest corner method
    while i < len(s) and j < len(d):
        val = min(s[i], d[j])
        alloc[i, j] = val
        basic_cells.append((i, j))
        s[i] -= val
        d[j] -= val
        
        if s[i] == 0 and d[j] == 0 and (i < len(s)-1 or j < len(d)-1):
            i += 1
            basic_cells.append((i, j))
            d[j] = 0
        elif s[i] == 0:
            i += 1
        else:
            j += 1

    iteration = 1
    while True:
        u = {0: 0}
        v = {}
        
        # calculate u and v
        while len(u) + len(v) < costs.shape[0] + costs.shape[1]:
            for r, c in basic_cells:
                if r in u and c not in v:
                    v[c] = costs[r, c] - u[r]
                elif c in v and r not in u:
                    u[r] = costs[r, c] - v[c]
                    
        # evaluate empty cells
        evaluations = []
        for r in range(costs.shape[0]):
            for c in range(costs.shape[1]):
                if (r, c) not in basic_cells:
                    delta = costs[r, c] - (u[r] + v[c])
                    evaluations.append((delta, (r, c)))
                    
        evaluations.sort(key=lambda x: x[0])
        best_delta, entering_cell = evaluations[0]
        
        # if all deltas are positive, we found the optimal
        if best_delta >= 0:
            break
            
        loop_path = locate_cycle(basic_cells, entering_cell)
        minus_nodes = loop_path[1::2]
        
        min_transfer = min([alloc[r, c] for r, c in minus_nodes])
        
        for idx, (r, c) in enumerate(loop_path):
            if idx % 2 == 0:
                alloc[r, c] += min_transfer
            else:
                alloc[r, c] -= min_transfer
                
        basic_cells.append(entering_cell)
        for r, c in minus_nodes:
            if alloc[r, c] == 0:
                basic_cells.remove((r, c))
                break
                
        iteration += 1
                
    final_cost = np.sum(alloc * costs)
    
    print("modi optimal allocation matrix:")
    print(alloc)
    print(f"minimum transportation cost: {final_cost}")

if __name__ == "__main__":
    solve_modi()