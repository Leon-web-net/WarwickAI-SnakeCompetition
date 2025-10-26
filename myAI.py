import os, json, random
from collections import defaultdict, deque
from snake.logic import GameState, Turn, Direction
from board_helpers import neighbours, passable, manhattan

# --- Globals for persistence across ticks and games ---
ACTIONS = [Turn.LEFT, Turn.STRAIGHT, Turn.RIGHT]
Q = defaultdict(lambda: [0.0, 0.0, 0.0])   # maps state_tuple -> list of Qs for 3 actions
prev_state = None
prev_action = None
prev_food_dist = None
prev_score = None

# Hyperparameters
ALPHA = 0.2
GAMMA = 0.98
EPS_START = 0.10     # start with small exploration to be stable
EPS_MIN = 0.02
EPS_DECAY = 0.9995   # decay per step

epsilon = EPS_START
model_path = "q_table.json"

def save_q():
    with open(model_path, "w") as f:
        json.dump({str(k): v for k, v in Q.items()}, f)

def load_q():
    if os.path.exists(model_path):
        with open(model_path) as f:
            raw = json.load(f)
        Q.clear()
        for k, v in raw.items():
            Q[tuple(eval(k))] = v

load_q()

# ----- Feature engineering -----
def danger(pos, state: GameState):
    """Cell unsafe next turn: wall, body, enemy, hazard."""
    x, y = pos
    # Outside grid
    if not (0 <= x < state.width and 0 <= y < state.height):
        return 1
    # Walls
    if pos in state.walls:
        return 1
    # Any snake bodies (yours or enemies)
    if pos in state.snake.body:
        return 1
    for e in state.enemies:
        if pos in e.body:
            return 1
    return 0

def rel_food_flags(state: GameState):
    """Food relative to current heading in {left, straight, right}."""
    head = state.snake.head
    if not state.food:
        return (0, 0, 0)
    # nearest food
    target = min(state.food, key=lambda f: manhattan(head, f))
    dir_now = Direction(state.snake.direction)
    # Map left/straight/right heads
    left = state.snake.get_next_head(Turn.LEFT)
    straight = state.snake.get_next_head(Turn.STRAIGHT)
    right = state.snake.get_next_head(Turn.RIGHT)
    def closer(next_pos): return 1 if manhattan(next_pos, target) < manhattan(head, target) else 0
    return (closer(left), closer(straight), closer(right))

def heading_id(state: GameState):
    # Encode absolute heading as 0..3 for coarse symmetry breaking
    # In this engine, snake.direction is an int index 0..3
    return state.snake.direction

def encode_state(state: GameState):
    """Small, discrete state tuple."""
    head = state.snake.head
    left = state.snake.get_next_head(Turn.LEFT)
    straight = state.snake.get_next_head(Turn.STRAIGHT)
    right = state.snake.get_next_head(Turn.RIGHT)

    dL = danger(left, state)
    dS = danger(straight, state)
    dR = danger(right, state)
    fL, fS, fR = rel_food_flags(state)
    h = heading_id(state)

    return (dL, dS, dR, fL, fS, fR, h)

def food_distance(state: GameState):
    if not state.food:
        return None
    head = state.snake.head
    return min(manhattan(head, f) for f in state.food)

# ----- Action selection -----
def select_action(s):
    global epsilon
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    qs = Q[s]
    # argmax with random tie-break
    maxq = max(qs)
    idxs = [i for i, q in enumerate(qs) if q == maxq]
    return ACTIONS[random.choice(idxs)]

# ----- Reward shaping -----
def compute_reward(state: GameState, done: bool):
    # Base step penalty
    r = -0.1
    # Food eaten is detectable via score increase or length growth; here use score
    # Guard for engines without last_events; also detect via score increase
    events = getattr(state, "last_events", None)
    if events and ("ate_food" in events):
        r += 10.0
    else:
        # Fallback: reward if score increased since last step
        global prev_score
        if prev_score is not None and state.score > prev_score:
            r += 10.0
    # Death signal (engine must expose; if not, infer from done flag or head collides)
    if done:
        r -= 100.0
    # Dense shaping: closer to food this step
    global prev_food_dist
    cur = food_distance(state)
    if prev_food_dist is not None and cur is not None:
        r += 1.0 if cur < prev_food_dist else -1.0
    return r

# --- Main policy function called by the game each tick ---
def myAI(state: GameState) -> Turn:
    """
    ε-greedy Q-learning policy. Needs the game engine to call with a 'done' or end-of-episode callback.
    If you lack a done flag, approximate: when snake dies, your AI won't be called again;
    do a terminal update on the next game's first call using a stored 'terminal_pending' flag.
    """
    global prev_state, prev_action, epsilon, prev_food_dist, prev_score

    # Encode current state
    s = encode_state(state)

    # Choose action
    a = select_action(s)

    # === Learning update for transition (prev_state, prev_action) -> s ===
    if prev_state is not None and prev_action is not None:
        # You must supply 'done' and 'last_events' from engine.
        done = getattr(state, "terminal", False)
        r = compute_reward(state, done)

        # Q-update
        a_idx_prev = ACTIONS.index(prev_action)
        a_idx_next = max(range(3), key=lambda i: Q[s][i])
        td_target = r + (0.0 if done else GAMMA * Q[s][a_idx_next])
        Q[prev_state][a_idx_prev] += ALPHA * (td_target - Q[prev_state][a_idx_prev])

        # Save sometimes
        if random.random() < 0.001:
            save_q()

        # Decay exploration
        epsilon = max(EPS_MIN, epsilon * EPS_DECAY)

    # Book-keeping for next step
    prev_state = s
    prev_action = a
    prev_food_dist = food_distance(state)
    prev_score = state.score

    return a
