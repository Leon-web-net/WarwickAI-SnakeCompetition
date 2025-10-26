import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNCNN(nn.Module):
    def __init__(self, in_channels: int, h: int, w: int, n_actions: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        # Global average pooling drastically reduces parameters and compute
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # Dueling head: value + advantage streams
        self.fc = nn.Linear(64, 256)
        self.val = nn.Linear(256, 1)
        self.adv = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc(x))
        val = self.val(x)
        adv = self.adv(x)
        q = val + (adv - adv.mean(dim=1, keepdim=True))
        return q

def init_model(device, in_channels, h, w, n_actions):
    net = DQNCNN(in_channels, h, w, n_actions).to(device)
    target = DQNCNN(in_channels, h, w, n_actions).to(device)
    target.load_state_dict(net.state_dict())
    target.eval()
    return net, target
