import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=512, num_experts=3):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_experts)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        hidden = self.fc2(x)           # Hidden representation
        logits = self.fc3(hidden)              # Raw logits output
        return logits                     # Return logits directly