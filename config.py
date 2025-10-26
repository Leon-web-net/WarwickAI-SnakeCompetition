from dataclasses import dataclass

@dataclass
class Config:
    # Model / training
    lr: float = 1e-4
    gamma: float = 0.98
    batch_size: int = 64
    buffer_size: int = 100_000
    min_buffer: int = 5_000
    target_update_freq: int = 1_000
    optimize_every: int = 8
    max_grad_norm: float = 10.0

    # Exploration
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 250_000  # exponential schedule

    # IO
    model_dir: str = "checkpoints"
    model_name: str = "dqn_snake.pt"

    # Env
    n_actions: int = 3  # LEFT, STRAIGHT, RIGHT

    # Advanced training toggles
    use_double_dqn: bool = True
    use_amp: bool = True
    updates_per_optimize: int = 2  # more GPU work per env step
    target_update_by_updates: bool = True  # update target by optimizer steps
    # Observation
    stack_k: int = 2  # frame stacking: number of frames to stack
