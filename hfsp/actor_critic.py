import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, num_input, num_output, node_num=100):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_input, node_num)
        # self.fc2 = nn.Linear(node_num, node_num)
        self.action_head = nn.Linear(node_num, num_output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self, num_input, num_output=1, node_num=100):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_input, node_num)
        # self.fc2 = nn.Linear(node_num, node_num)
        self.state_value = nn.Linear(node_num, num_output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        value = self.state_value(x)
        return value
