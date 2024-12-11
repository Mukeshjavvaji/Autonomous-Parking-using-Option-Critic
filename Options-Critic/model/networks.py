import torch.nn as nn
import torch

torch.autograd.set_detect_anomaly(True)

class PolicyOverOptions(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.Softmax(dim=0)
        )

    def forward(self, input):
        return self.layers(input)
    

class SubPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(59, 64),
            nn.ReLU(),
            nn.Linear(64, 16), 
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Tanh()
        )

    def forward(self, input):
        return self.layers(input)
    

class TerminationPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(7, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.layers(input)
    

class QOmega(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(7, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, input):
        return self.layers(input)
    
class QU(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(61, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, input):
        return self.layers(input)


