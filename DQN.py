import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(1,32),
            nn.ReLU(),
            nn.Linear(32,3) # 左か右か
        )
    def forward(self,x):
        logits = self.f(x)
        return logits