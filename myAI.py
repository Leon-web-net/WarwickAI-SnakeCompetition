import os
import subprocess
import sys
from collections import deque
from typing import Optional

import numpy as np
import torch

from snake.logic import GameState, Turn
from rl_dqn.models import init_model
from rl_dqn.utils import state_to_ndarray, load_model_if_exists
from config import Config


# Evaluation-only DQN agent (no training or optimizing)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True

print(f"Device {DEVICE}")

_cfg = Config()
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_model_path = os.path.join(PROJECT_ROOT, _cfg.model_dir, _cfg.model_name)
print(f"Model path: {_model_path}")
_policy_net = None
_target_net = None
_in_channels: Optional[int] = None
_expected_in_c: Optional[int] = None  # in-channels expected by loaded weights

# Lazy training delegation flag/state
_SNAKE_TRAIN = (os.environ.get("SNAKE_TRAIN") == "1")
_delegated_myAI = None

# For frame stacking on eval to match training input shape
_obs_history = deque(maxlen=max(0, _cfg.stack_k - 1))


def _build_stacked(obs_now: np.ndarray) -> np.ndarray:
    frames = [obs_now]
    for past in _obs_history:
        frames.append(past)
    while len(frames) < _cfg.stack_k:
        frames.append(np.zeros_like(obs_now))
    return np.concatenate(frames[: _cfg.stack_k], axis=0)


def _safe_action_mask(state: GameState):
    snake = state.snake
    h, w = state.height, state.width

    other_bodies = set()
    for e in state.enemies:
        if getattr(e, "isAlive", True):
            other_bodies |= set(e.body)

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

    return [is_safe(Turn.LEFT), is_safe(Turn.STRAIGHT), is_safe(Turn.RIGHT)]


def _resolve_model_path() -> str:
    # Allow override via env var
    p = os.environ.get("SNAKE_MODEL_PATH")
    if p and os.path.exists(p):
        return p
    # Default from config
    p = _model_path
    if os.path.exists(p):
        return p
    # Fallback: pick most recent .pt in checkpoints directory
    ckpt_dir = os.path.join(PROJECT_ROOT, _cfg.model_dir)
    if os.path.isdir(ckpt_dir):
        pts = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
        if pts:
            pts.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            return pts[0]
    return p  # may not exist


def _ensure_model(state: GameState):
    global _policy_net, _target_net, _in_channels
    if _policy_net is not None:
        return

    obs_probe = state_to_ndarray(state)
    base_c, H, W = obs_probe.shape
    in_c_default = base_c * _cfg.stack_k
    _in_channels = in_c_default

    # Inspect checkpoint to infer expected in_channels if possible
    model_path = _resolve_model_path()
    in_c_from_ckpt = None
    if os.path.exists(model_path):
        try:
            obj = torch.load(model_path, map_location="cpu")
            state_dict = None
            if isinstance(obj, dict) and "policy" in obj and isinstance(obj["policy"], dict):
                state_dict = obj["policy"]
            elif isinstance(obj, dict):
                state_dict = obj
            if isinstance(state_dict, dict) and "conv1.weight" in state_dict:
                shape = state_dict["conv1.weight"].shape  # [32, C, 3, 3]
                if len(shape) == 4:
                    in_c_from_ckpt = int(shape[1])
        except Exception:
            pass

    chosen_in_c = in_c_from_ckpt or in_c_default
    _expected_in_c = chosen_in_c

    _policy_net, _target_net = init_model(DEVICE, chosen_in_c, H, W, _cfg.n_actions)
    # Load weights from checkpoint (full or weights-only)
    loaded = False
    if os.path.exists(model_path):
        try:
            obj = torch.load(model_path, map_location="cpu")
            if isinstance(obj, dict) and "policy" in obj and isinstance(obj["policy"], dict):
                _policy_net.load_state_dict(obj["policy"], strict=False)
                loaded = True
            elif isinstance(obj, dict):
                _policy_net.load_state_dict(obj, strict=False)
                loaded = True
        except Exception:
            loaded = False
    if not loaded:
        # As a final attempt, use utility for raw weights
        try:
            if load_model_if_exists(model_path, _policy_net):
                loaded = True
        except Exception:
            loaded = False
    # Basic diagnostics to help verify we're loading what we expect
    try:
        print(f"[myAI] Device={DEVICE} | obsC={in_c_default} | ckptC={(in_c_from_ckpt or 'n/a')} | usingC={chosen_in_c}")
        print(f"[myAI] Checkpoint path: {model_path} | loaded={loaded}")
    except Exception:
        pass
    _target_net.load_state_dict(_policy_net.state_dict())

    _policy_net.eval()
    _target_net.eval()


@torch.no_grad()
def _greedy_action(obs_tensor: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> int:
    q = _policy_net(obs_tensor.to(DEVICE))  # type: ignore[arg-type]
    if valid_mask is not None and valid_mask.dtype == torch.bool and valid_mask.numel() == _cfg.n_actions and valid_mask.any().item():
        q = q.clone()
        invalid = (~valid_mask.view(1, -1)).to(q.device)
        q[invalid] = -1e9
    return int(torch.argmax(q, dim=1).item())


def myAI(state: GameState) -> Turn:
    global _delegated_myAI
    # Lazily delegate to training implementation to avoid import-time side effects
    if _SNAKE_TRAIN:
        if _delegated_myAI is None:
            try:
                from myAI_dqn_training import myAI as _training_myAI  # type: ignore
                _delegated_myAI = _training_myAI
                print("[myAI] Delegating to training myAI (SNAKE_TRAIN=1)")
            except Exception as _e:
                print(f"[myAI] Failed to delegate to training myAI: {_e}")
                # Fall through to eval mode if delegation fails
        if _delegated_myAI is not None:
            return _delegated_myAI(state)
    _ensure_model(state)
    obs_now = state_to_ndarray(state)
    obs_stacked = _build_stacked(obs_now)
    # Align channels to match the loaded weights if needed
    if _expected_in_c is not None:
        c = obs_stacked.shape[0]
        if c > _expected_in_c:
            obs_stacked = obs_stacked[:_expected_in_c, :, :]
        elif c < _expected_in_c:
            pad = np.zeros((_expected_in_c - c, *obs_stacked.shape[1:]), dtype=obs_stacked.dtype)
            obs_stacked = np.concatenate([obs_stacked, pad], axis=0)
    obs_tensor = torch.from_numpy(obs_stacked).unsqueeze(0)

    mask_list = _safe_action_mask(state)
    mask_tensor = torch.tensor(mask_list, dtype=torch.bool)

    action_idx = _greedy_action(obs_tensor, mask_tensor)

    # Update history for next frame
    _obs_history.appendleft(obs_now)

    return [Turn.LEFT, Turn.STRAIGHT, Turn.RIGHT][action_idx]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run one evaluation game with DQN")
    parser.add_argument("--difficulty", type=str, default="medium", help="Difficulty from snake/difficulties.yaml")
    parser.add_argument("--viz", action="store_true", help="Run with renderer instead of headless test")
    args = parser.parse_args()

    # Force evaluation mode in the child process
    env = os.environ.copy()
    env["SNAKE_EVAL"] = "1"

    if args.viz:
        cmd = [sys.executable, "-m", "snake.snake", "run", args.difficulty]
    else:
        # Headless single game
        cmd = [sys.executable, "-m", "snake.snake", "test", "100", args.difficulty]

    sys.exit(subprocess.call(cmd, env=env))
