import random
from collections import deque
from snake.logic import GameState, Turn, Snake, Direction

"""
Your mission, should you choose to accept it, is to write the most cracked snake AI possible.

All the info you'll need to do this is in the GameState and Snake classes in snake/logic.py

Below is all of the data you'll need, and some small examples that you can uncomment and use if you want :)

"""


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
    def out_of_bounds(p):
        """
        Return True if position p lies outside the grid.
        Args:
            p: (x,y) integer coordinates.
        """
        x, y = p
        return x < 0 or x >= grid_width or y < 0 or y >= grid_height

    def unsafe(p):
        """
        Returns True if moving the head to position p would cause a collision
        or leave the grid.
        Collision checked against walls, own body, and all enemy bodies.
        """

        if out_of_bounds(p):
            return True

        if p in walls:
            return True
        
        if p in set(my_snake_body):
            return True
        
        if any(p in set(e.body) for e in enemy_snakes):
            return True
        
        return False

    def manhattan(a,b):
        return abs(a[0]-b[0]) +abs(a[1]-b[1])


    head = my_snake_body[0]
    candidates = [Turn.STRAIGHT, Turn.LEFT, Turn.RIGHT]
    next_pos   = {Turn.STRAIGHT: straight, Turn.LEFT: left, Turn.RIGHT: right}

    # pick the first safe move; fallback to any move if none safe
    safe_moves = [t for t in candidates if not unsafe(next_pos[t])]
    next_move = safe_moves[0] if safe_moves else Turn.STRAIGHT

    
    def food_dist(move):
        """
        Returns the manhattan distance from the resulting head position
        to the chosen target food. If no food exists, returns 0.
        """
        p = next_pos[move]
        return manhattan(p,target) if target is not None else 0

    target = min(food, key=lambda f: manhattan(f, head)) if food else None

    next_move = Turn.STRAIGHT

    if safe_moves:
        best = min(safe_moves, key=lambda t: (food_dist(t),0 if t is Turn.STRAIGHT else 1))
        next_move = best
        # next_move = 
    else:
        next_move = Turn.STRAIGHT    

    return next_move
    # push
    # ======================================
    # =       Try out some examples!       =
    # ======================================

    # from examples.dumbAI import dumbAI
    # return dumbAI(state)

    #from examples.smartAI import smartAI
    #return smartAI(state)
