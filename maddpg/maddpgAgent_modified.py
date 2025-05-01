"""
Reference:
This file was modified from

1. ddpg_agent.py from Udacity's GitHub Repository ddpg-pendulum
https://github.com/udacity/deep-reinforcement-optimizeing/blob/master/ddpg-pendulum/ddpg_agent.py

2. maddpg.py from Udacity's drlnd MADDPG-Lab

"""

import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from networkModels import ActorNetwork, CriticNetwork


## MADDPG_Agent

class MADDPG_Agent:
    
    """
    An MADDPG_Agent includes:
        2 select_actionor (select_actionor_local, select_actionor_target) networks
        2 critic (critic_local, critic_target) networks
    """
    
    def __init__(self, agent_index, config):
        
        """
        agent_index: index of the agent (0 or 1)
        
        config: the dictionary containing the keys:
            select_actionor_input_size: input size of the select_actionor (24, dimension of the state of a single agent)
            select_actionor_output_size: output size of the select_actionor (2, dimension of the select_actionion of a single agent)
            select_actionor_hidden_sizes: input and output sizes of the hidden FC layer of the select_actionor
            
            critic_state_size: sum of the dimensions of the state of both participants (48)
            critic_select_actionion_size: sum of the dimensions of the select_actionion of both participants (4)
            critic_hidden_sizes: input and output sizes of the hidden FC layer of the critic
            
            select_actionor_lr: optimizeing rate of the select_actionor
            critic_lr: optimizeing rate of the critic
            critic_L2_decay: L2 weight decay of the critic
            
            gamma: the discounting rate
            
            tau: soft-update fselect_actionor
        """
        
        self.agent_index = agent_index
        
        select_actionor_input_size = config['select_actionor_input_size']
        select_actionor_output_size = config['select_actionor_output_size']
        select_actionor_hidden_sizes = config['select_actionor_hidden_sizes']
        
        critic_state_size = config['critic_state_size']
        critic_select_actionion_size = config['critic_select_action_size']
        critic_hidden_sizes = config['critic_hidden_sizes']
        
        select_actionor_lr = config['select_actionor_lr']
        critic_lr = config['critic_lr']
        critic_L2_decay = config['critic_L2_decay']
        
        self.gamma = config['gamma']
        
        self.tau = config['tau']
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        
        ## Actor networks (local & target)
        self.select_actionor_local = ActorNetwork(select_actionor_input_size, select_actionor_output_size, select_actionor_hidden_sizes).to(self.device)
        self.select_actionor_target = ActorNetwork(select_actionor_input_size, select_actionor_output_size, select_actionor_hidden_sizes).to(self.device)
        
        # set select_actionor_local and select_actionor_target with same weights & biases
        for local_param, target_param in zip(self.select_actionor_local.parameters(), self.select_actionor_target.parameters()):
            target_param.data.copy_(local_param.data)
        
        ## Critic networks (local & target)
        self.critic_local = CriticNetwork(critic_state_size, critic_select_actionion_size, critic_hidden_sizes).to(self.device)
        self.critic_target = CriticNetwork(critic_state_size, critic_select_actionion_size, critic_hidden_sizes).to(self.device)
        
        # set critic_local and critic_target with same weights & biases
        for local_param, target_param in zip(self.critic_local.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(local_param.data)
        
        # optimizers
        self.select_actionor_optimizer = optim.Adam(self.select_actionor_local.parameters(), lr = select_actionor_lr)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr = critic_lr, weight_decay = critic_L2_decay)
        
        
    def select_action(self, state_t, select_actionor_name = 'target', noise_bool = False, noise_np = None):
        """
        Use the select_actionor network to determine the select_actionion
            
        inputs:
            state_t: state tensor of shape (m, 24) observed by the agent
            select_actionor_name: the select_actionor network to use ("local" or "target")
            noise_bool: whether or not to add the noise
            noise_np - the noise to be added (if noise_bool == True), an ndarray of shape (m, 2)
        output:
            the select_actionion (tensor) of shape (m, 2)
        """
        if select_actionor_name == 'local':
            select_actionor_network = self.select_actionor_local
        elif select_actionor_name == 'target':
            select_actionor_network = self.select_actionor_target
        
        select_actionor_network.eval()
        with torch.no_grad():
            select_actionion = select_actionor_network(state_t).float().detach().to(self.device) # select_actionion is a tensor
            
            if noise_bool: # to add noise
                select_actionion = select_actionion.cpu().data.numpy() # convert select_actionion to ndarray
                select_actionion = np.clip(select_actionion + noise_np, -1, 1) # add noise and clip between [-1, +1]
                select_actionion = torch.from_numpy(select_actionion).float().detach().to(self.device) # convert select_actionion to tensor
        select_actionor_network.train()
        
        return select_actionion       
    
    
    def soft_update(self, local_nn, target_nn):
        """
        Soft-update the weight of the select_actionor (or critic) target network
        """
        for local_param, target_param in zip(local_nn.parameters(), target_nn.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
            
    
    def optimize(self, replays, other_agent):
        """        
        Used the sampled replays to train the select_actionor and the critic
        
        replays: replay tuples in the format of (states, select_actionions, rewards, next_states, dones)
            
            states.shape = (m, 48)
            select_actionions.shape = (m, 4)
            rewards.shape = (m, 2)
            next_states.shape = (m, 48)
            dones.shape = (m, 2)
        
        other_select_actionor: the other agent
        """
                        
        ## assign s, a, r, s', d from replays
        states, select_actionions, rewards, next_states, dones = replays

        # size of the batch
        m = select_actionions.shape[0]
        
        ## convert from ndarrays to tensors
        
        if self.agent_index == 0:
            # states, select_actionions, next_select_actionions of the agent
            states_self = torch.from_numpy(states[:, :24]).float().to(self.device)             # [m, 24]
            select_actionion_self = torch.from_numpy(select_actionions[:, :2]).float().to(self.device)             # [m, 2]
            next_states_self = torch.from_numpy(next_states[:, :24]).float().to(self.device)   # [m, 24]
            # states, select_actionions, next_select_actionions of the other agent
            states_other = torch.from_numpy(states[:, 24:]).float().to(self.device)            # [m, 24]
            select_actionion_other = torch.from_numpy(select_actionions[:, 2:]).float().to(self.device)            # [m, 2]
            next_states_other = torch.from_numpy(next_states[:, 24:]).float().to(self.device)  # [m, 24]
            # rewards and dones
            rewards = torch.from_numpy(rewards[:, 0].reshape((-1, 1))).float().to(self.device)                  # [m, 1]
            dones = torch.from_numpy(dones[:, 0].reshape((-1, 1)).astype(np.uint8)).float().to(self.device)     # [m, 1]
            
        elif self.agent_index == 1:
            # states, select_actionions, next_select_actionions of the agent
            states_self = torch.from_numpy(states[:, 24:]).float().to(self.device)             # [m, 24]
            select_actionion_self = torch.from_numpy(select_actionions[:, 2:]).float().to(self.device)             # [m, 2]
            next_states_self = torch.from_numpy(next_states[:, 24:]).float().to(self.device)   # [m, 24]
            # states, select_actionions, next_select_actionions of the other agent
            states_other = torch.from_numpy(states[:, :24]).float().to(self.device)            # [m, 24]
            select_actionion_other = torch.from_numpy(select_actionions[:, :2]).float().to(self.device)            # [m, 2]
            next_states_other = torch.from_numpy(next_states[:, :24]).float().to(self.device)  # [m, 24]
            # rewards and dones
            rewards = torch.from_numpy(rewards[:, 1].reshape((-1, 1))).float().to(self.device)                  # [m, 1]
            dones = torch.from_numpy(dones[:, 1].reshape((-1, 1)).astype(np.uint8)).float().to(self.device)     # [m, 1]
        
        # s, a, s' for both participants
        states = torch.from_numpy(states).float().to(self.device)                 # [m, 48]
        select_actionions = torch.from_numpy(select_actionions).float().to(self.device)               # [m, 4]
        next_states = torch.from_numpy(next_states).float().to(self.device)       # [m, 48]
                
        
        """ Train critic_local """
        # next_select_actionions of the agent
        next_select_actionions_self = self.select_action(state_t = next_states_self, select_actionor_name = 'target', noise_bool = False)
        
        # next_select_actionions of the other
        next_select_actionions_other = other_agent.select_action(state_t = next_states_other, select_actionor_name = 'target', noise_bool = False)
                
        # combine next select_actionions from both participants
        if self.agent_index == 0:
            next_select_actionions = torch.cat([next_select_actionions_self, next_select_actionions_other], dim = 1).float().detach().to(self.device) # (m, 4)
        elif self.agent_index == 1:
            next_select_actionions = torch.cat([next_select_actionions_other, next_select_actionions_self], dim = 1).float().detach().to(self.device) # (m, 4)
         
        
        # q_next: use critic_target to obatin the select_actionion-value of (next_states, next_select_actionions)
        self.critic_target.eval()
        with torch.no_grad():
            q_next = self.critic_target(next_states, next_select_actionions).detach().to(self.device) # [m, 1]
        self.critic_target.train()
        
        
        # q_target: the TD target of the critic, i.e. q_target = r + gamma*q_next
        q_target = rewards + self.gamma * q_next * (1-dones) # [m, 1]
        
        # q_local: the current select_actionion-value of (states, select_actionions)
        q_local = self.critic_local(states, select_actionions) # [m, 1]
        
        # critic_loss
        self.critic_optimizer.zero_grad()
        critic_loss = F.smooth_l1_loss(q_local, q_target.detach())
        critic_loss.backward()
        
        """ Train select_actionor_local """
        # select_actionion_local
        if self.agent_index == 0:
            select_actionion_local = torch.cat([self.select_actionor_local(states_self),
                                      other_agent.select_action(states_other, select_actionor_name = 'local', noise_bool = False)],
                                      dim = 1)
        elif self.agent_index == 1:
            select_actionion_local = torch.cat([other_agent.select_action(states_other, select_actionor_name = 'local', noise_bool = False),
                                      self.select_actionor_local(states_self)],
                                      dim = 1)
                       
        # select_actionor_loss
        self.select_actionor_optimizer.zero_grad()
        select_actionor_loss = - self.critic_local(states, select_actionion_local).mean()
        select_actionor_loss.backward()
        
        ## soft-update select_actionor_target and critic_target
        self.soft_update(self.select_actionor_local, self.select_actionor_target)
        self.soft_update(self.critic_local, self.critic_target)