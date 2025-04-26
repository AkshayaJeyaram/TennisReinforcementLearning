############################################################
########### Code by George Kontis, April 2025 ##############
############################################################

import torch
import torch.nn as nn # for building neural networks in the actor-critic code
import torch.nn.functional as F # API for activations and loss functions
from torch.distributions import Normal

NEURONS_COUNT = 128 # for the Actor_Critic networks below
LEARNING_RATE = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor_Critic(nn.Module):
    """
    This Actor-Critic class is a 3 layer fully connected neural network used in the PPO class below.

    I understood the intuition for this code from:
        - Week 9 of the CM50270 module
        - Reinforecement Learning, Sutton & Barto, 2018, 2nd ed, Section 11.1
        - CodeEmporium's video "Proximal Policy Optimization | ChatGPT uses this" https://www.youtube.com/watch?v=MVXdncpCbYE&t=15s
        - shimao's response https://stats.stackexchange.com/questions/380123/reinforcement-learning-what-is-the-logic-behind-actor-critic-methods-why-use

    The actor learns the probability of the agent choosing a given action in a given state (policy function)
    The critic evaluates the agent's action choices (value function)
    Both of these are actualised in the PPO class below
    """
    
    def __init__(self, input_dim, output_dim, hidden_dim, output_gate = None):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)
        self.output_gate = output_gate

    def forward(self, x):
        """
        I got the idea of using linear layers and separate networks for the policy and value function (see PPOAgent) from the original PPO paper: 
        Schulman, J., Wolski, F., Dhariwal, P., Radford, A. and Klimov, O. (2017) 'Proximal Policy Optimization Algorithms', arXiv:1707.06347. Available at: https://arxiv.org/abs/1707.06347.
        
        However, unlike in the paper, I experimented with ReLu instead of tanh as I found ReLu to be more stable in training for this application. Also, unlike in the paper, I made use of
        entropy bonus (see MultiAgentPPO in ma.py file)
        """
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        if self.output_gate:
            x = self.output_gate(x)
        return x

class PPO:
    """
    Here I define the PPO Agent for the tennis environment, to be used in the MultiAgentPPO class in the ma.py file
    Actions are sampled from a multivariate Normal distribution (mu, std) following the official pseudocode and documentation https://spinningup.openai.com/en/latest/algorithms/ppo.html
    The update step is in the MultiAgentPPO class in the ma.py file
    """
    
    def __init__(self, state_size, action_size):
        """
        state_size: 24 for each agent
        action_size: 2 for each agent (horizontal movement and jumping vertically)

        Like in the paper, I use separate networks for the actor and critic
        """
        
        self.actor = Actor_Critic(state_size, action_size, NEURONS_COUNT, torch.tanh).to(DEVICE) # separate networks for the actor and the critic
        self.critic = Actor_Critic(state_size, 1, NEURONS_COUNT).to(DEVICE)
        self.std = nn.Parameter(torch.ones(1, action_size).to(DEVICE))
        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + 
                                          list(self.critic.parameters()) + 
                                          [self.std], lr = LEARNING_RATE)

    def get_action(self, state):
        """
        Samples a continuous action (given the tennis env) from the current policy and
        returns both the action and its log probability. Action dimension is 2, controlling horizontal movement and jumping

        Batch size is 1 as I am sampling one action for a state at a time

        Dimension definitions for variables (like actions) in the OpenAI documentation https://spinningup.openai.com/en/latest/algorithms/ppo.html

        This took a lot of trial and error so I left the dimension comments for sanity checking
        """

        state = torch.tensor(state, dtype = torch.float32, device = DEVICE) # Tensors on GPU

        # Ensuring shape is [batch_size = 1, state_dim = 2]
        if state.dim() == 1:
            state = state.unsqueeze(0)  # [1, 2]
        elif state.dim() > 2:
            raise ValueError(f"Unexpected state shape: {state.shape}")

        with torch.no_grad(): # for inference only
            mu = self.actor(state)  # [1, 2]

            std = self.std
            # Ensuring std shape matches mu
            if std.dim() == 1:
                std = std.unsqueeze(0)  # [1, 2]
            std = std.expand_as(mu) # [1, 2]

            dist = Normal(mu, std)
            action = dist.sample() # [1, 2]
            log_prob = dist.log_prob(action).sum(dim = -1) # [1]

        return action.squeeze(0).cpu().numpy(), log_prob.item()

    def evaluate(self, state, action):
        """
        This evaluates the current policy and value function for given states and actions for training using the actor and critic networks

        Returns:
            log_prob: log probability of the provided action under the current policy
            entropy: the policy's entropy (encourages exploration, see solution.py)
            value: the critic's value estimate for the state
        """

        mu = self.actor(state)
        std = self.std.expand_as(mu)
        dist = Normal(mu, std)
        log_prob = dist.log_prob(action).sum(dim = -1, keepdim = True)
        entropy = dist.entropy().sum(dim = -1)
        value = self.critic(state)
        return log_prob, entropy, value