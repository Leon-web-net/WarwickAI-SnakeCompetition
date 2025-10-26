import os
import math
import time
import torch
import numpy as np

def epsilon_by_step(step, eps_start, eps_end, eps_decay_steps):
    return eps_end + (eps_start - eps_end) * math.exp(-step / eps_decay_steps)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_model(path: str, net: torch.nn.Module):
    ensure_dir(os.path.dirname(path))
    torch.save(net.state_dict(), path)

def _load_state_flexible(net: torch.nn.Module, state: dict) -> tuple[int, int, int]:
    """Load only matching-shaped parameters from state into net.

    Returns (loaded, missing, skipped) counts.
    """
    cur = net.state_dict()
    loadable = {}
    loaded = skipped = 0
    for k, v in state.items():
        if k in cur and cur[k].shape == v.shape:
            loadable[k] = v
            loaded += 1
        else:
            skipped += 1
    net.load_state_dict(loadable, strict=False)
    missing = sum(1 for k in cur.keys() if k not in loadable)
    return loaded, missing, skipped

def load_model_if_exists(path: str, net: torch.nn.Module):
    if os.path.exists(path):
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict):
            try:
                # Try strict first for exact match
                net.load_state_dict(state)
                return True
            except Exception:
                # Fallback: partial/flexible load for changed shapes (e.g., channel count)
                _load_state_flexible(net, state)
                return True
    return False

# --- Full training checkpoint helpers (backward compatible) ---

def save_checkpoint(path: str, policy_net: torch.nn.Module, target_net: torch.nn.Module, agent, extra: dict | None = None):
    """Save a full training checkpoint. Backward compatible with weight-only loads.

    Contents:
      - policy, target: state_dicts
      - optimizer: agent.optimizer state
      - scaler: AMP scaler state (if present)
      - agent_step, update_steps
      - extra: optional dict for outer counters (e.g., step_counter)
      - saved_at, version
    """
    ensure_dir(os.path.dirname(path))
    ckpt = {
        "version": 1,
        "saved_at": int(time.time()),
        "policy": policy_net.state_dict(),
        "target": target_net.state_dict(),
        "optimizer": agent.optimizer.state_dict() if hasattr(agent, "optimizer") else None,
        "scaler": agent.scaler.state_dict() if getattr(agent, "scaler", None) is not None else None,
        "agent_step": getattr(agent, "step", 0),
        "update_steps": getattr(agent, "update_steps", 0),
        "extra": extra or {},
    }
    torch.save(ckpt, path)

def load_checkpoint_if_exists(path: str, policy_net: torch.nn.Module, target_net: torch.nn.Module, agent):
    """Load a full checkpoint if available; otherwise try weight-only.

    Returns: (loaded: bool, extra: dict)
    """
    if not os.path.exists(path):
        return False, {}
    obj = torch.load(path, map_location="cpu")
    # Weight-only backward compatibility
    if isinstance(obj, dict) and all(k in obj for k in ("policy", "target")):
        try:
            policy_net.load_state_dict(obj["policy"])
            target_net.load_state_dict(obj["target"])
        except Exception:
            # Flexible load if shapes changed (e.g., added channels)
            _load_state_flexible(policy_net, obj["policy"])
            _load_state_flexible(target_net, obj["target"])
        # Optimizer/scaler may fail to load across shape changes; best-effort
        try:
            if obj.get("optimizer") is not None and hasattr(agent, "optimizer"):
                agent.optimizer.load_state_dict(obj["optimizer"])
        except Exception:
            pass
        try:
            if obj.get("scaler") is not None and getattr(agent, "scaler", None) is not None:
                agent.scaler.load_state_dict(obj["scaler"])
        except Exception:
            pass
        agent.step = int(obj.get("agent_step", getattr(agent, "step", 0)))
        agent.update_steps = int(obj.get("update_steps", getattr(agent, "update_steps", 0)))
        return True, obj.get("extra", {})
    # If it's a raw state_dict, assume policy weights were saved
    if isinstance(obj, dict):
        try:
            policy_net.load_state_dict(obj)
            target_net.load_state_dict(policy_net.state_dict())
            return True, {}
        except Exception:
            # Partial load if shapes changed
            _load_state_flexible(policy_net, obj)
            target_net.load_state_dict(policy_net.state_dict())
            return True, {}
    return False, {}

# --- State encoding: 3-channel (snake, food, walls). Add more channels if needed.
import numpy as np

def state_to_ndarray(state) -> np.ndarray:
    h, w = state.height, state.width
    planes = []

    # 1) my body
    p = np.zeros((h, w), np.float32)
    for x, y in state.snake.body: p[y, x] = 1.0
    planes.append(p)

    # 2) my head
    p = np.zeros((h, w), np.float32)
    hx, hy = state.snake.head
    p[hy, hx] = 1.0
    planes.append(p)

    # 3) my tail (last body segment)
    p = np.zeros((h, w), np.float32)
    tx, ty = state.snake.body[-1]
    p[ty, tx] = 1.0
    planes.append(p)

    # 4) enemy bodies
    p = np.zeros((h, w), np.float32)
    for e in state.enemies:
        for x, y in e.body: p[y, x] = 1.0
    planes.append(p)

    # 5) enemy heads
    p = np.zeros((h, w), np.float32)
    for e in state.enemies:
        ex, ey = e.head
        p[ey, ex] = 1.0
    planes.append(p)

    # 6) food
    p = np.zeros((h, w), np.float32)
    for fx, fy in state.food: p[fy, fx] = 1.0
    planes.append(p)

    # 7) walls
    p = np.zeros((h, w), np.float32)
    for wx, wy in state.walls: p[wy, wx] = 1.0
    planes.append(p)

    # 8-9) food vector planes (dx, dy) from head to nearest food, normalized
    # If no food, fill zeros
    hx, hy = state.snake.head
    if state.food:
        # Find nearest food by Manhattan distance
        fx, fy = min(state.food, key=lambda f: abs(hx - f[0]) + abs(hy - f[1]))
        # Normalize by board size; handle division-by-zero for degenerate sizes
        denom_x = max(1, w)
        denom_y = max(1, h)
        dx = np.float32((fx - hx) / denom_x)
        dy = np.float32((fy - hy) / denom_y)
    else:
        dx = np.float32(0.0)
        dy = np.float32(0.0)
    planes.append(np.full((h, w), dx, dtype=np.float32))
    planes.append(np.full((h, w), dy, dtype=np.float32))

    arr = np.stack(planes, axis=0)  # (C,H,W)
    return arr

