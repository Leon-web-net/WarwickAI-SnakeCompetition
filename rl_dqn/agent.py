from typing import Optional
import torch
import torch.nn.functional as F
from .replay import ReplayBuffer
from .utils import epsilon_by_step

class DQNAgent:
    def __init__(self, net, target_net, cfg, device):
        self.net = net
        self.target_net = target_net
        self.device = device
        self.gamma = cfg.gamma
        self.batch_size = cfg.batch_size
        self.optimize_every = cfg.optimize_every
        self.target_update_freq = cfg.target_update_freq
        self.max_grad_norm = cfg.max_grad_norm
        self.step = 0
        self.update_steps = 0  # count successful optimizer updates
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_size)
        self.cfg = cfg
        # Mixed precision scaler for faster training on GPU
        amp_enabled = getattr(cfg, "use_amp", True) and (device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled) if hasattr(torch.cuda, "amp") else None
        # Multiple updates per optimization trigger
        self.updates_per_optimize = getattr(cfg, "updates_per_optimize", 1)
        # Whether to apply target update by optimizer update steps rather than env steps
        self.target_update_by_updates = getattr(cfg, "target_update_by_updates", True)
        # Double DQN toggle (default True)
        self.use_double_dqn = getattr(cfg, "use_double_dqn", True)

    def get_state(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "step": self.step,
            "update_steps": self.update_steps,
        }

    def load_state(self, state: dict):
        if not state:
            return
        if state.get("optimizer") is not None:
            self.optimizer.load_state_dict(state["optimizer"])
        if state.get("scaler") is not None and self.scaler is not None:
            self.scaler.load_state_dict(state["scaler"])
        self.step = int(state.get("step", self.step))
        self.update_steps = int(state.get("update_steps", self.update_steps))

    def act(self, state_tensor, valid_mask: Optional[torch.Tensor] = None):
        """Select an action with optional action masking.

        valid_mask: 1D bool tensor of shape [n_actions] on CPU. If provided and has
        any True, random and greedy selection will be restricted to valid actions.
        """
        eps = epsilon_by_step(self.step, self.cfg.eps_start, self.cfg.eps_end, self.cfg.eps_decay_steps)
        self.step += 1
        mask_ok = (
            valid_mask is not None
            and valid_mask.dtype == torch.bool
            and valid_mask.numel() == self.cfg.n_actions
            and valid_mask.any().item()
        )
        if torch.rand(1).item() < eps:
            if mask_ok:
                idxs = torch.nonzero(valid_mask, as_tuple=False).view(-1)
                choice = idxs[torch.randint(0, idxs.numel(), (1,))].item()
                return int(choice), eps
            return int(torch.randint(0, self.cfg.n_actions, (1,)).item()), eps

        with torch.no_grad():
            q = self.net(state_tensor.to(self.device))
            if mask_ok:
                q = q.clone()
                invalid = ~valid_mask.view(1, -1).to(q.device)
                q[invalid] = -1e9
            return int(torch.argmax(q, dim=1).item()), eps

    def push(self, s, a, r, s2, done):
        self.buffer.push(s, a, r, s2, done)

    def optimize(self):
        if self.step % self.optimize_every != 0: 
            return None
        if len(self.buffer) < self.cfg.min_buffer:
            return None

        last_loss = None
        for _ in range(self.updates_per_optimize):
            s, a, r, s2, d = self.buffer.sample(self.batch_size, self.device)

            use_amp = self.scaler is not None and self.scaler.is_enabled()
            self.optimizer.zero_grad(set_to_none=True)
            if use_amp:
                # Use new torch.amp.autocast API to avoid deprecation warnings
                with torch.amp.autocast("cuda"):
                    q = self.net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        if self.use_double_dqn:
                            next_actions = self.net(s2).argmax(1)
                            next_q = self.target_net(s2).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                        else:
                            next_q = self.target_net(s2).max(1)[0]
                        target = r + (1.0 - d) * self.gamma * next_q
                    loss = F.smooth_l1_loss(q, target)
                self.scaler.scale(loss).backward()
                # Clip after unscale
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                q = self.net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    if self.use_double_dqn:
                        next_actions = self.net(s2).argmax(1)
                        next_q = self.target_net(s2).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    else:
                        next_q = self.target_net(s2).max(1)[0]
                    target = r + (1.0 - d) * self.gamma * next_q
                loss = F.smooth_l1_loss(q, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

            self.update_steps += 1
            last_loss = float(loss.item())

            # Target network update
            if self.target_update_by_updates:
                if self.update_steps % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.net.state_dict())
            else:
                if self.step % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.net.state_dict())

        return last_loss
