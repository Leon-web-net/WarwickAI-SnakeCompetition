import random
import json
import os
from snake.logic import GameState, Turn

def load_head_positions():
    """Load head positions from file"""
    try:
        if os.path.exists('head_positions.json'):
            with open('head_positions.json', 'r') as f:
                return json.load(f)
        return []
    except:
        return []

def save_head_positions(positions):
    """Save head positions to file"""
    try:
        with open('head_positions.json', 'w') as f:
            json.dump(positions, f)
    except:
        pass

def manhattan(a, b):
    """Simple Manhattan distance"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def jAI(state: GameState) -> Turn:
    snake = state.snake
    walls = state.walls
    enemies = state.enemies
    width, height = state.width, state.height
    food = state.food
    safe_moves = []

    # Load previous head positions
    head_positions = load_head_positions()
    
    # Add current head position at the front
    head_positions.insert(0, list(snake.head))  # Convert tuple to list for JSON
    
    # Keep only as many positions as body length
    if len(head_positions) > len(snake.body):
        head_positions = head_positions[:len(snake.body)]

    # Check all possible turns
    for turn in Turn:
        next_head = snake.get_next_head(turn)

        # 1️⃣ Check if within bounds
        if not (0 <= next_head[0] < width and 0 <= next_head[1] < height):
            continue

        # 2️⃣ Check for wall collision
        if next_head in walls:
            continue

        # 3️⃣ Check for self collision (exclude tail, since it moves)
        if next_head in list(snake.body)[:-1]:
            continue

        # 4️⃣ Check for enemy collision
        collides_enemy = any(next_head in e.body_set for e in enemies if e.isAlive)
        if collides_enemy:
            continue

        # 5️⃣ Check if next position is in recent head positions (avoid revisiting)
        if list(next_head) in head_positions:
            continue

        # If passed all checks, it's safe
        safe_moves.append(turn)

    # Save updated head positions for next call
    save_head_positions(head_positions)

    # If no safe moves, go straight as a last resort
    if not safe_moves:
        return Turn.STRAIGHT

    # === NEW: Choose move that gets closest to nearest food ===
    if food:
        # Find closest food
        closest_food = min(food, key=lambda f: manhattan(snake.head, f))
        # Pick safe move that reduces distance to closest food
        best_move = safe_moves[0]
        min_dist = manhattan(snake.get_next_head(best_move), closest_food)
        for move in safe_moves[1:]:
            dist = manhattan(snake.get_next_head(move), closest_food)
            if dist < min_dist:
                min_dist = dist
                best_move = move
        return best_move

    # If no food, pick any safe move (first one)
    return safe_moves[0]