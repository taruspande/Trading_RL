import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size * 5, hidden_size)  # Notice the change here
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        state = state.view(state.size(0), -1)  # Flattening the last two dimensions
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
