import random
from collections import deque
from snake.logic import GameState, Turn, Snake, Direction
from board_helpers import (
    manhattan, build_hazard_sets, unsafe, neighbours, passable,
)

from jlogic import jAI

"""
Your mission, should you choose to accept it, is to write the most cracked snake AI possible.

All the info you'll need to do this is in the GameState and Snake classes in snake/logic.py

Below is all of the data you'll need, and some small examples that you can uncomment and use if you want :)

"""

def bfs_path(start,goals,W,H,self_body,enemy_cells,wall_cells):
    goals = set(goals)
    q = deque([start]); prev = {start: None}
    while q:
        u = q.popleft()
        if u in goals:
            path = []
            while u is not None:
                path.append(u); u = prev[u]
            return list(reversed(path))
        
        for v in neighbours(u):
            if v not in prev and passable(v,W,H,self_body,enemy_cells, wall_cells):
                prev[v] = u; q.append(v)
    
    return None

def flood_area(start,W,H,self_body, enemy_cells, wall_cells):
    if not passable(start,W,H,self_body,enemy_cells,wall_cells): return 0
    q = deque([start]); seen = {start}
    while q:
        u = q.popleft()
        for v in neighbours(u):
            if v not in seen and passable(v,W,H, self_body, enemy_cells,wall_cells):
                seen.add(v); q.append(v)
    
    return len(seen)    


def myAI(state: GameState) -> Turn:

    # ======================================
    # =         Some Useful data.          =
    # ======================================

    grid_width: int = state.width
    grid_height: int = state.height
    food: set = state.food
    walls: set = state.walls
    score: int = state.score
    my_snake: Snake = state.snake
    my_snake_direction: Direction = Direction(state.snake.direction)
    my_snake_body: list = list(state.snake.body)
    enemy_snakes = state.enemies

    # you may also find the get_next_head() method of the Snake class useful!
    # this tells you what position the snake's head will end up in for each of the moves
    # you can then check for collisions, food etc
    straight = my_snake.get_next_head(Turn.STRAIGHT)
    left = my_snake.get_next_head(Turn.LEFT)
    right = my_snake.get_next_head(Turn.RIGHT)

    # ======================================
    # =         Your Code Goes Here        =
    # ======================================

    self_body, enemy_cells, wall_cells = build_hazard_sets(
        my_snake_body,enemy_snakes,walls)
    
    next_pos = {
        Turn.STRAIGHT: straight,
        Turn.LEFT: left,
        Turn.RIGHT: right,
    }

    safe_moves = [t for t,p in next_pos.items() if not unsafe(
        p,grid_width, grid_height, self_body, enemy_cells,wall_cells
    )]
    head = my_snake_body[0]
    target = min(food, key=lambda f: manhattan(f,head)) if food else None

    # 1) Try BFS path to food on current board
    if safe_moves and target:
        path = bfs_path(head,[target], grid_width,grid_height,
                        self_body,enemy_cells,wall_cells)
        if path and len(path) >=2:
            step = path[1] # next cell along the path

            for mv,pos in next_pos.items():
                if pos == step and mv in safe_moves:
                    return mv
    
    # 2) fallback : maximise space after the move
    def score(move):
        p = next_pos[move]
        space = flood_area(p,grid_width,grid_height, self_body, enemy_cells, wall_cells)
        dist = manhattan(p, target) if target else 0
        tie =  0 if move is Turn.STRAIGHT else 1
        return (space, -dist, -(1-tie))
    
    if safe_moves:
        return max(safe_moves, key=score)

    return Turn.STRAIGHT


    # ======================================
    # =       Try out some examples!       =
    # ======================================

    # from examples.dumbAI import dumbAI
    # return dumbAI(state)

    #from examples.smartAI import smartAI
    #return smartAI(state)
