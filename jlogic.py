import json, os, random
from collections import deque
from snake.logic import GameState, Turn, DIRECTIONS

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def bfs_path(start, goals, width, height, walls, bodies):
    """Find shortest path to any goal avoiding walls and bodies."""
    queue = deque([(start, [])])
    visited = {start}
    while queue:
        pos, path = queue.popleft()
        if pos in goals:
            return path
        for dx, dy in DIRECTIONS:
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < width and 0 <= ny < height:
                npos = (nx, ny)
                if npos not in visited and npos not in walls and npos not in bodies:
                    visited.add(npos)
                    queue.append((npos, path + [npos]))
    return None

def open_space_score(start, width, height, walls, bodies, limit=50):
    """Count how many free tiles are reachable from start (limited flood fill)."""
    queue = deque([start])
    visited = {start}
    while queue and len(visited) < limit:
        x, y = queue.popleft()
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                npos = (nx, ny)
                if npos not in visited and npos not in walls and npos not in bodies:
                    visited.add(npos)
                    queue.append(npos)
    return len(visited)

def can_reach_tail(head, tail, width, height, walls, bodies):
    """Check if we can reach our own tail from the head position."""
    if tail in bodies:
        return False
    return bfs_path(head, {tail}, width, height, walls, bodies - {tail}) is not None

def is_dead_end(pos, width, height, walls, bodies, min_space=10):
    """Check if a position leads to a dead end with insufficient space."""
    reachable = open_space_score(pos, width, height, walls, bodies, limit=min_space + 5)
    return reachable < min_space

def predict_enemy_heads(enemies):
    """Predict where enemy heads might be on next turn."""
    danger_zones = set()
    for enemy in enemies:
        for turn in Turn:
            next_pos = enemy.get_next_head(turn)
            danger_zones.add(next_pos)
    return danger_zones

def jAI(state: GameState) -> Turn:
    snake = state.snake
    enemies = [e for e in state.enemies if e.isAlive]
    walls = state.walls
    food = state.food
    width, height = state.width, state.height

    body_set = set(snake.body)
    all_enemy_bodies = set().union(*(e.body_set for e in enemies)) if enemies else set()
    
    # Predict dangerous zones from enemy movements
    enemy_danger_zones = predict_enemy_heads(enemies) if enemies else set()

    possible_moves = {t: snake.get_next_head(t) for t in Turn}

    safe_moves = {}
    for t, next_head in possible_moves.items():
        # Check bounds
        if not (0 <= next_head[0] < width and 0 <= next_head[1] < height):
            continue
        
        # Check for collisions with walls and bodies
        # Exclude our tail since it will move (unless we're about to eat)
        body_check = body_set - {snake.body[-1]} if next_head not in food else body_set
        if next_head in walls or next_head in body_check or next_head in all_enemy_bodies:
            continue
        
        # Avoid predicted enemy head positions (potential head-on collisions)
        if next_head in enemy_danger_zones:
            # Allow if we're longer than all nearby enemies
            risky = False
            for enemy in enemies:
                if manhattan(next_head, enemy.head) <= 2:
                    if len(snake.body) <= len(enemy.body):
                        risky = True
                        break
            if risky:
                continue
        
        safe_moves[t] = next_head

    if not safe_moves:
        return Turn.STRAIGHT  # No options, last resort

    # Score each safe move
    move_scores = {}
    for turn, pos in safe_moves.items():
        score = 0
        
        # Check if this move leads to sufficient open space
        obstacles = body_set | all_enemy_bodies
        open_space = open_space_score(pos, width, height, walls, obstacles, limit=50)
        score += open_space * 2  # Weight open space heavily
        
        # Penalize dead ends more harshly
        min_required_space = max(len(snake.body), 15)
        if open_space < min_required_space:
            score -= 100
        
        # Check if we can still reach our tail (important for survival)
        tail = snake.body[-1]
        if can_reach_tail(pos, tail, width, height, walls, obstacles):
            score += 30
        
        # Food seeking logic - but be smart about it
        if food:
            path_to_food = bfs_path(pos, food, width, height, walls, obstacles)
            if path_to_food:
                food_dist = len(path_to_food)
                
                # Only prioritize food if we're not too long or if we have safe space
                if len(snake.body) < 15 or open_space > 25:
                    score += max(0, 50 - food_dist)  # Closer food = higher score
                    
                    # Bonus for food that doesn't trap us
                    food_pos = path_to_food[-1]
                    space_after_food = open_space_score(food_pos, width, height, walls, 
                                                       obstacles | {pos}, limit=25)
                    if space_after_food > min_required_space:
                        score += 20
                else:
                    # When long, be more cautious about food
                    score += max(0, 20 - food_dist)
        
        # Avoid edges when possible (gives more maneuvering room)
        x, y = pos
        if x == 0 or x == width - 1 or y == 0 or y == height - 1:
            score -= 5
        
        # Avoid getting too close to enemies when we're smaller
        for enemy in enemies:
            dist_to_enemy = manhattan(pos, enemy.head)
            if len(snake.body) < len(enemy.body):
                if dist_to_enemy <= 2:
                    score -= 30
                elif dist_to_enemy <= 4:
                    score -= 10
        
        move_scores[turn] = score

    # Choose the move with the highest score
    best_turn = max(move_scores.items(), key=lambda x: x[1])[0]
    return best_turn

# import json, os, random
# from collections import deque
# from snake.logic import GameState, Turn, DIRECTIONS

# def manhattan(a, b):
#     return abs(a[0] - b[0]) + abs(a[1] - b[1])

# def bfs_path(start, goals, width, height, walls, bodies):
#     """Find shortest path to any goal avoiding walls and bodies."""
#     queue = deque([(start, [])])
#     visited = {start}
#     while queue:
#         pos, path = queue.popleft()
#         if pos in goals:
#             return path
#         for dx, dy in DIRECTIONS:
#             nx, ny = pos[0] + dx, pos[1] + dy
#             if 0 <= nx < width and 0 <= ny < height:
#                 npos = (nx, ny)
#                 if npos not in visited and npos not in walls and npos not in bodies:
#                     visited.add(npos)
#                     queue.append((npos, path + [npos]))
#     return None

# def open_space_score(start, width, height, walls, bodies, limit=30):
#     """Count how many free tiles are reachable from start (limited flood fill)."""
#     queue = deque([start])
#     visited = {start}
#     while queue and len(visited) < limit:
#         x, y = queue.popleft()
#         for dx, dy in DIRECTIONS:
#             nx, ny = x + dx, y + dy
#             if 0 <= nx < width and 0 <= ny < height:
#                 npos = (nx, ny)
#                 if npos not in visited and npos not in walls and npos not in bodies:
#                     visited.add(npos)
#                     queue.append(npos)
#     return len(visited)

# def jAI(state: GameState) -> Turn:
#     snake = state.snake
#     enemies = [e for e in state.enemies if e.isAlive]
#     walls = state.walls
#     food = state.food
#     width, height = state.width, state.height

#     body_set = set(snake.body)
#     all_enemy_bodies = set().union(*(e.body_set for e in enemies))

#     possible_moves = {t: snake.get_next_head(t) for t in Turn}

#     safe_moves = {}
#     for t, next_head in possible_moves.items():
#         # Check bounds
#         if not (0 <= next_head[0] < width and 0 <= next_head[1] < height):
#             continue
#         # Check for collisions
#         if next_head in walls or (next_head in body_set - {snake.body[-1]}) or (next_head in all_enemy_bodies):
#             continue
#         # Avoid head-on collisions
#         if any(next_head == e.get_next_head(Turn.STRAIGHT) for e in enemies):
#             continue
#         safe_moves[t] = next_head

#     if not safe_moves:
#         return Turn.STRAIGHT  # No options, go straight and hope

#     # BFS to nearest food
#     if food:
#         path = bfs_path(snake.head, food, width, height, walls, body_set | all_enemy_bodies)
#         if path and len(path) > 0:
#             next_tile = path[0]
#             for turn, head_pos in safe_moves.items():
#                 if head_pos == next_tile:
#                     return turn

#     # Fallback: pick move with most open space
#     best_turn, best_score = None, -1
#     for turn, pos in safe_moves.items():
#         score = open_space_score(pos, width, height, walls, body_set | all_enemy_bodies)
#         if score > best_score:
#             best_score = score
#             best_turn = turn

#     return best_turn if best_turn else Turn.STRAIGHT

# import random
# import json
# import os
# from snake.logic import GameState, Turn

# def load_head_positions():
#     """Load head positions from file"""
#     try:
#         if os.path.exists('head_positions.json'):
#             with open('head_positions.json', 'r') as f:
#                 return json.load(f)
#         return []
#     except:
#         return []

# def save_head_positions(positions):
#     """Save head positions to file"""
#     try:
#         with open('head_positions.json', 'w') as f:
#             json.dump(positions, f)
#     except:
#         pass

# def manhattan(a, b):
#     """Simple Manhattan distance"""
#     return abs(a[0] - b[0]) + abs(a[1] - b[1])

# def jAI(state: GameState) -> Turn:
#     snake = state.snake
#     walls = state.walls
#     enemies = state.enemies
#     width, height = state.width, state.height
#     food = state.food
#     safe_moves = []

#     # Load previous head positions
#     head_positions = load_head_positions()
    
#     # Add current head position at the front
#     head_positions.insert(0, list(snake.head))  # Convert tuple to list for JSON
    
#     # Keep only as many positions as body length
#     if len(head_positions) > len(snake.body):
#         head_positions = head_positions[:len(snake.body)]

#     # Check all possible turns
#     for turn in Turn:
#         next_head = snake.get_next_head(turn)

#         # 1️⃣ Check if within bounds
#         if not (0 <= next_head[0] < width and 0 <= next_head[1] < height):
#             continue

#         # 2️⃣ Check for wall collision
#         if next_head in walls:
#             continue

#         # 3️⃣ Check for self collision (exclude tail, since it moves)
#         if next_head in list(snake.body)[:-1]:
#             continue

#         # 4️⃣ Check for enemy collision
#         collides_enemy = any(next_head in e.body_set for e in enemies if e.isAlive)
#         if collides_enemy:
#             continue

#         # 5️⃣ Check if next position is in recent head positions (avoid revisiting)
#         if list(next_head) in head_positions:
#             continue

#         # If passed all checks, it's safe
#         safe_moves.append(turn)

#     # Save updated head positions for next call
#     save_head_positions(head_positions)

#     # If no safe moves, go straight as a last resort
#     if not safe_moves:
#         return Turn.STRAIGHT

#     # === NEW: Choose move that gets closest to nearest food ===
#     if food:
#         # Find closest food
#         closest_food = min(food, key=lambda f: manhattan(snake.head, f))
#         # Pick safe move that reduces distance to closest food
#         best_move = safe_moves[0]
#         min_dist = manhattan(snake.get_next_head(best_move), closest_food)
#         for move in safe_moves[1:]:
#             dist = manhattan(snake.get_next_head(move), closest_food)
#             if dist < min_dist:
#                 min_dist = dist
#                 best_move = move
#         return best_move

#     # If no food, pick any safe move (first one)
#     return safe_moves[0]