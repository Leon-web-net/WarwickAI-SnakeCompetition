import random
from collections import deque
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity:int):
        self.buf = deque(maxlen=capacity)
    
    def push(self,s,a,r,s2,done):
        self.buf.append((s,a,r,s2,done))
    
    def __len__(self): return len(self.buf)

    def sample(self,batch_size:int, device):
        batch = random.sample(self.buf,batch_size)
        s,a,r,s2,d = map(np.array, zip(*batch))
        # Use pinned memory for faster host->device copies; non_blocking transfers
        pin = (device.type == "cuda")
        s = torch.from_numpy(np.stack(s)).float().pin_memory() if pin else torch.from_numpy(np.stack(s)).float()
        s2 = torch.from_numpy(np.stack(s2)).float().pin_memory() if pin else torch.from_numpy(np.stack(s2)).float()
        a = torch.from_numpy(a.astype(np.int64)).pin_memory() if pin else torch.from_numpy(a.astype(np.int64))
        r = torch.from_numpy(r.astype(np.float32)).pin_memory() if pin else torch.from_numpy(r.astype(np.float32))
        d = torch.from_numpy(d.astype(np.float32)).pin_memory() if pin else torch.from_numpy(d.astype(np.float32))
        s = s.to(device, non_blocking=pin)
        s2 = s2.to(device, non_blocking=pin)
        a = a.to(device, non_blocking=pin)
        r = r.to(device, non_blocking=pin)
        d = d.to(device, non_blocking=pin)
        return s, a, r, s2, d
        
