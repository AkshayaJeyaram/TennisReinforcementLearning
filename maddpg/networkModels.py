"""
Reference:
This file was modified from model.py from Udacity's GitHub Repository ddpg-pendulum
https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

## Actor Neural Network

class ActorNetwork(nn.Module):
    """
    The actor network outputs the action of a single agent by using the state (observed by the same agent) as the input
    In this project, each agent (player) observes the state of dimension = 24 and has the action (continuous value) of dimension = 2
    The values of the action are continuous, ranging from -1 to +1
    """
    
    def __init__(self, input_size = 24, output_size = 2, hidden_sizes = [400, 300]):
        super(ActorNetwork, self).__init__()

        self.hidden_layers = nn.ModuleList([])
        self.batch_norms = nn.ModuleList([])

        self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.batch_norms.append(nn.BatchNorm1d(hidden_sizes[0]))

        for h1, h2 in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            self.hidden_layers.append(nn.Linear(h1, h2))
            self.batch_norms.append(nn.BatchNorm1d(h2))

        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

        self.reset_parameters() # reset the initial weights and biases of all layers
    
    def reset_parameters(self):
        for layer in self.hidden_layers:
            f = layer.weight.data.size()[0]
            layer.weight.data.uniform_(-1.0/np.sqrt(f), 1.0/np.sqrt(f))
            layer.bias.data.fill_(0.1)
        self.output_layer.weight.data.uniform_(-3e-3, 3e-3)
        self.output_layer.bias.data.fill_(0.1)

    def forward(self, x):
        for layer, bn in zip(self.hidden_layers, self.batch_norms):
            x = F.relu(bn(layer(x)))
        return F.tanh(self.output_layer(x))


## Critic Neural Network

class CriticNetwork(nn.Module):
    """
    The critic network
    
    Given the states and actions of "both agents", the critic network outputs the action-value Q(s1, s2, a1, a2)
    """
    
    def __init__(self, state_size = 48, action_size = 4, hidden_sizes = [400, 300]):
        super(CriticNetwork, self).__init__()

        self.first_layer = nn.Linear(state_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
        self.second_layer = nn.Linear(hidden_sizes[0] + action_size, hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.output_layer = nn.Linear(hidden_sizes[1], 1)

        self.reset_parameters()

    def reset_parameters(self):
        f1 = self.first_layer.weight.data.size()[0]
        f2 = self.second_layer.weight.data.size()[0]

        self.first_layer.weight.data.uniform_(-1.0/np.sqrt(f1), 1.0/np.sqrt(f1))
        self.second_layer.weight.data.uniform_(-1.0/np.sqrt(f2), 1.0/np.sqrt(f2))
        self.output_layer.weight.data.uniform_(-3e-3, 3e-3)

        self.first_layer.bias.data.fill_(0.1)
        self.second_layer.bias.data.fill_(0.1)
        self.output_layer.bias.data.fill_(0.1)

    def forward(self, state, action):
        xs = F.relu(self.bn1(self.first_layer(state)))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.bn2(self.second_layer(x)))
        x = self.output_layer(x)
        return x