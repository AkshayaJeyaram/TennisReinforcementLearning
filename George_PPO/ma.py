############################################################
########### Code by George Kontis, April 2025 ##############
############################################################

import torch
import torch.nn.functional as F # API for activations and loss functions

from model import PPOAgent

EPS_CLIP = 0.2 # This controls the extent to which updates can change the probability ratio in the PPO-clip objective https://spinningup.openai.com/en/latest/algorithms/ppo.html
K_EPOCHS = 6 # Number of times states, actions, rewards are used per epoch as defined below

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiAgentPPO:
    """
    Manages the training of multiple PPO agents, using the PPOAgent class defined in model.py.
    """
    
    def __init__(self, num_agents, state_size, action_size):
        self.agents = [PPOAgent(state_size, action_size) for _ in range(num_agents)]

    def act(self, states):
        """
        This collects actions and probabilities from each agent in the environment
        """

        actions, log_probs = [], []
        for agent, state in zip(self.agents, states):
            action, log_prob = agent.get_action(state)
            actions.append(action)
            log_probs.append(log_prob)
        return actions, log_probs

    def update(self, trajectories, entropy_coef):
        """
        This performs PPO updates for each agent using collected trajectories, where for each agent, it runs K_EPOCHS of gradient descent on the clipped PPO
        objective with value and entropy bonuses to balance exploration and exploitation. I employ entropy decay in the solution.py file

        Inputs:
            trajectories: list of states, actions, old_log_probs, returns, advantages
            entropy_coef: annealed entropy bonus to use this update

        I implemented this following the optimisation pseudocode from the official documentation https://spinningup.openai.com/en/latest/algorithms/ppo.html for L
        and step 6 of the PPO pseudocode.
        """
        
        for i, agent in enumerate(self.agents):

            # unpacking and converting into tensors
            states, actions, old_log_probs, returns, advantages = trajectories[i]
            states = torch.FloatTensor(states).to(DEVICE)
            actions = torch.FloatTensor(actions).to(DEVICE)
            old_log_probs = torch.FloatTensor(old_log_probs).to(DEVICE).unsqueeze(1)
            returns = torch.FloatTensor(returns).to(DEVICE)
            advantages = torch.FloatTensor(advantages).to(DEVICE).unsqueeze(1)

            for epoch in range(K_EPOCHS):
                new_log_probs, entropy, values = agent.evaluate(states, actions)
                ratios = torch.exp(new_log_probs - old_log_probs)

                # PPO surrogate losses
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values.squeeze(-1), returns)

                entropy_loss = - entropy_coef * entropy.mean()
                loss = actor_loss + 0.5 * critic_loss + entropy_loss

                # backpropagating the agent's actor, critic and log-std
                agent.optimizer.zero_grad()
                loss.backward()
                agent.optimizer.step()
