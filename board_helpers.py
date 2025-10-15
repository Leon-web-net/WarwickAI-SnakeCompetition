from collections import deque

# --- geometry / metrics ---
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# --- board helpers ---
def out_of_bounds(p, W, H):
    x, y = p
    return x < 0 or x >= W or y < 0 or y >= H

def build_hazard_sets(my_body, enemies, walls):
    self_body = set(my_body)
    enemy_cells = {c for e in enemies for c in e.body}
    wall_cells = set(walls)
    return self_body, enemy_cells, wall_cells

def unsafe(p, W, H, self_body, enemy_cells, wall_cells):
    return out_of_bounds(p, W, H) or p in wall_cells or p in self_body or p in enemy_cells

def passable(p, W, H, self_body, enemy_cells, wall_cells):
    return not unsafe(p, W, H, self_body, enemy_cells, wall_cells)

def neighbours(p):
    x, y = p
    return [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]