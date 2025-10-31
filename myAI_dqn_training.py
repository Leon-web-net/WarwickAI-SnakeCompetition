import os, atexit, time
from collections import deque
import numpy as np
import torch

from snake.logic import GameState, Turn
from rl_dqn.models import init_model
from rl_dqn.agent import DQNAgent
from rl_dqn.utils import state_to_ndarray, save_model, load_model_if_exists, save_checkpoint, load_checkpoint_if_exists
from config import Config

print("[DQN] myAI imported from:", __file__)
# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Enable cuDNN autotuner for fixed input sizes on GPU
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
print(f"DEVICE: {DEVICE}")
EVAL = os.getenv("SNAKE_EVAL", "0") == "1"
# inside _ensure_agent after loading weights:


# --- Globals for the game callback ---
_cfg = Config()

# Apply epsilon overrides from environment (set by launcher)
def _env_float(name, default):
    try:
        return float(os.getenv(name)) if os.getenv(name) is not None else default
    except Exception:
        return default

_cfg.eps_start = _env_float("DQN_EPS_START", _cfg.eps_start)
_cfg.eps_end = _env_float("DQN_EPS_END", _cfg.eps_end)
_cfg.eps_decay_steps = int(_env_float("DQN_EPS_DECAY_STEPS", float(_cfg.eps_decay_steps)))

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_model_path   = os.path.join(PROJECT_ROOT, _cfg.model_dir, _cfg.model_name)
print("[DQN] Checkpoint path:", _model_path)
os.makedirs(os.path.dirname(_model_path), exist_ok=True)

_stack_k = getattr(Config, 'stack_k', 2) if hasattr(Config, '__annotations__') else 2
_in_channels = None  # will infer from first observation and stack_k
# We infer C,H,W on first call
_policy_net = None
_target_net = None
_agent: DQNAgent = None

# Book-keeping to assemble transitions from frame-to-frame
_prev_obs = None        # np.ndarray (C,H,W)
_prev_action = None     # int
_prev_score = None
_prev_food_dist = None
_step_counter = 0
_episode_counter = 0
_episode_return = 0.0
_episode_len = 0
_returns_100 = deque(maxlen=100)
_lengths_100 = deque(maxlen=100)
_last_loss = None
_loss_ema = None
_last_eps = 1.0
_last_log_time = time.time()
_last_log_step = 0
_log_every_episodes = int(os.getenv("DQN_LOG_EVERY", "100"))
_csv_log_enabled = os.getenv("DQN_CSV_LOG", "0") == "1"
_log_csv_path = os.path.join(PROJECT_ROOT, _cfg.model_dir, "train_log.csv")
_obs_history = deque(maxlen=max(0, _cfg.stack_k - 1))

def _build_stacked(obs_now: np.ndarray) -> np.ndarray:
    # Build [obs_now] + history frames; pad with zeros if insufficient history
    frames = [obs_now]
    for past in _obs_history:
        frames.append(past)
    # Pad with zeros for initial steps
    while len(frames) < _cfg.stack_k:
        frames.append(np.zeros_like(obs_now))
    return np.concatenate(frames[:_cfg.stack_k], axis=0)

def _food_distance(state: GameState):
    if not state.food:
        return None
    hx, hy = state.snake.head
    return min(abs(hx - fx) + abs(hy - fy) for fx, fy in state.food)

def _safe_action_mask(state: GameState):
    """Compute which of [LEFT, STRAIGHT, RIGHT] are immediately safe.
    Mirrors core checks in SnakeGame._move_snake for the next head cell.
    """
    snake = state.snake
    h, w = state.height, state.width

    # Precompute occupied cells of other snakes
    other_bodies = set()
    for e in state.enemies:
        if getattr(e, "isAlive", True):
            other_bodies |= set(e.body)

    # Player body without tail (tail may move)
    body_wo_tail = set(list(snake.body)[:-1])

    def is_safe(turn: Turn) -> bool:
        nhx, nhy = snake.get_next_head(turn)
        if nhx < 0 or nhx >= w or nhy < 0 or nhy >= h:
            return False
        if (nhx, nhy) in state.walls:
            return False
        if (nhx, nhy) in body_wo_tail:
            return False
        if (nhx, nhy) in other_bodies:
            return False
        return True

    mask = [is_safe(Turn.LEFT), is_safe(Turn.STRAIGHT), is_safe(Turn.RIGHT)]
    return mask

def _reward(state: GameState, done: bool) -> float:
    # Softer step cost to not overwhelm sparse rewards
    r = -0.02
    # food via score delta
    global _prev_score
    if _prev_score is not None and state.score > _prev_score:
        r += 10.0
    # shaping: gentle distance-to-food incentive
    global _prev_food_dist
    cur = _food_distance(state)
    if _prev_food_dist is not None and cur is not None:
        if cur < _prev_food_dist:
            r += 0.05
        elif cur > _prev_food_dist:
            r -= 0.05
    if done:
        r -= 100.0
    return r

def _ensure_agent(state: GameState):
    global _policy_net, _target_net, _agent
    if _policy_net is not None:
        return
    # Probe observation to determine channels and initialize model
    obs_probe = state_to_ndarray(state)
    base_c, H, W = obs_probe.shape
    in_c = base_c * _cfg.stack_k
    global _in_channels
    _in_channels = in_c
    _policy_net, _target_net = init_model(DEVICE, in_c, H, W, _cfg.n_actions)
    _agent = DQNAgent(_policy_net, _target_net, _cfg, DEVICE)
    # Try to load full training checkpoint first; fallback to weights-only
    loaded, extra = load_checkpoint_if_exists(_model_path, _policy_net, _target_net, _agent)
    if not loaded:
        if load_model_if_exists(_model_path, _policy_net):
            _target_net.load_state_dict(_policy_net.state_dict())
    if EVAL:
        _agent.cfg.eps_start = 0.0
        _agent.cfg.eps_end   = 0.0
    # Optional: reset epsilon schedule on resume
    if os.getenv("DQN_RESET_EPS", "0") == "1":
        _agent.step = 0
        _agent.update_steps = 0
    # Restore outer counters if available
    try:
        global _step_counter, _episode_counter
        _step_counter = int(extra.get("myai_step_counter", _step_counter))
        _episode_counter = int(extra.get("myai_episode_counter", _episode_counter))
    except Exception:
        pass


def _episode_reset_detect(state: GameState) -> bool:
    # Detect an episode boundary by a score reset from >0 to 0.
    # This avoids incorrectly flagging early steps in a fresh episode as terminal.
    return (
        _prev_obs is not None
        and _prev_score is not None
        and _prev_score > 0
        and state.score == 0
    )

def _save_ckpt(tag=""):
    extra = {
        "myai_step_counter": _step_counter,
        "myai_episode_counter": _episode_counter,
    }
    try:
        save_checkpoint(_model_path, _policy_net, _target_net, _agent, extra)
        print(f"[DQN] Saved {tag} -> {_model_path}")
    except Exception:
        # Fallback to weights-only
        save_model(_model_path, _policy_net)
        print(f"[DQN] Saved weights-only {tag} -> {_model_path}")

atexit.register(lambda: _policy_net is not None and _save_ckpt("atexit"))

def myAI(state: GameState) -> Turn:
    global _prev_obs, _prev_action, _prev_score, _prev_food_dist, _step_counter, _episode_counter
    global _episode_return, _episode_len, _returns_100, _lengths_100, _last_loss, _loss_ema, _last_eps
    global _last_log_time, _last_log_step

    _ensure_agent(state)
    obs_now = state_to_ndarray(state)
    obs_stacked = _build_stacked(obs_now)
    obs_tensor = torch.from_numpy(obs_stacked).unsqueeze(0)
    # Action masking to avoid obvious suicides
    mask_list = _safe_action_mask(state)
    mask_tensor = torch.tensor(mask_list, dtype=torch.bool)
    action_idx, eps = _agent.act(obs_tensor, valid_mask=mask_tensor)
    _last_eps = eps

    # Detect episode end for the PREVIOUS transition
    done_flag = _episode_reset_detect(state)

    if not EVAL and _prev_obs is not None and _prev_action is not None:
        r = _reward(state, done_flag)
        _agent.push(_prev_obs, _prev_action, r, obs_stacked, float(done_flag))
        loss = _agent.optimize()  # may be a no-op if buffer < min_buffer
        if loss is not None:
            _last_loss = loss
            _loss_ema = loss if _loss_ema is None else (0.98 * _loss_ema + 0.02 * loss)
        _episode_return += r
        _episode_len += 1
        _step_counter += 1
        # if _step_counter % 10000 == 0:
        #     _save_ckpt(f"step{_step_counter}")
        if done_flag:
            _episode_counter += 1
            # Lightweight progress reporting
            _returns_100.append(_episode_return)
            _lengths_100.append(_episode_len)
            avg_ret = sum(_returns_100) / len(_returns_100)
            avg_len = sum(_lengths_100) / len(_lengths_100)
            now = time.time()
            dt = now - _last_log_time
            steps_delta = _step_counter - _last_log_step
            sps = (steps_delta / dt) if dt > 0 else 0.0
            if _episode_counter % _log_every_episodes == 0:
                loss_str = f"{_loss_ema:.4f}" if _loss_ema is not None else "n/a"
                print(
                    f"[DQN] ep={_episode_counter} step={_step_counter} eps={_last_eps:.3f} "
                    f"loss_ema={loss_str} ret100={avg_ret:.2f} len100={avg_len:.1f} "
                    f"buf={len(_agent.buffer)} sps={sps:.0f}"
                )
                _last_log_time = now
                _last_log_step = _step_counter
                if _csv_log_enabled:
                    try:
                        with open(_log_csv_path, "a", encoding="utf-8") as f:
                            f.write(
                                f"{_step_counter},{_episode_counter},{_last_eps:.6f},{_last_loss if _last_loss is not None else ''},{_loss_ema if _loss_ema is not None else ''},{_episode_return:.6f},{avg_ret:.6f},{_episode_len},{avg_len:.2f},{len(_agent.buffer)},{sps:.2f},{int(now)}\n"
                            )
                    except Exception:
                        pass
            # periodic checkpoint by episodes
            if _episode_counter % 500 == 0:
                _save_ckpt(f"ep{_episode_counter}_step{_step_counter}")
            # reset per-episode accumulators
            _episode_return = 0.0
            _episode_len = 0

    _prev_obs = obs_stacked
    _prev_action = action_idx
    _prev_score = state.score
    _prev_food_dist = _food_distance(state)
    # Update history with the latest base observation
    _obs_history.appendleft(obs_now)

    return [Turn.LEFT, Turn.STRAIGHT, Turn.RIGHT][action_idx]

if __name__ == "__main__":
    import argparse, subprocess, sys, os

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=8000)
    parser.add_argument("--difficulty", type=str, default="hard")
    # Epsilon overrides and schedule reset
    parser.add_argument("--eps-start", type=float, default=0.3)
    parser.add_argument("--eps-end", type=float, default=0.1)
    
    parser.add_argument("--eps-decay-steps", type=int)
    parser.add_argument("--reset-eps-schedule", action="store_true", default=100_000)
    args = parser.parse_args()

    # Use current interpreter; run engine module directly
    cmd = [sys.executable, "-m", "snake.snake", "test", str(args.episodes), args.difficulty]

    # Ensure training mode in child
    env = os.environ.copy()
    env.pop("SNAKE_EVAL", None)  # force learning
    env["SNAKE_TRAIN"] = "1"  # delegate myAI to training implementation in child
    # Optional: make saves noisy
    env["DQN_VERBOSE"] = "1"
    # Pass epsilon overrides to child via env
    if args.eps_start is not None:
        env["DQN_EPS_START"] = str(args.eps_start)
    if args.eps_end is not None:
        env["DQN_EPS_END"] = str(args.eps_end)
    if args.eps_decay_steps is not None:
        env["DQN_EPS_DECAY_STEPS"] = str(args.eps_decay_steps)
    if args.reset_eps_schedule:
        env["DQN_RESET_EPS"] = "1"

    print("Running:", " ".join(cmd))
    rc = subprocess.call(cmd, env=env)
    sys.exit(rc)
