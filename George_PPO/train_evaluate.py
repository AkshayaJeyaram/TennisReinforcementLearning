############################################################
########### Code by George Kontis, April 2025 ##############
############################################################

import numpy as np
import torch
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment
from mappo import MultiAgentPPO

STATE_SIZE = 24
ACTION_SIZE = 2
GAMMA = 0.99
GAE_LAMBDA = 0.95
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def GAE(rewards, masks, values, gamma = GAMMA, lam = GAE_LAMBDA):
    """
    I compute the Generalised Advantage Estimation (GAE) of each agent.

    References that helped me understand GAE:
        The original paper Schulman, J., Moritz, P., Levine, S., Jordan, M. and Abbeel, P. (2015) High-Dimensional Continuous Control Using Generalized Advantage Estimation. Available at: https://arxiv.org/abs/1506.02438.
        Raj's notes on "High-Dimensional Continuous Control Using Generalized Advantage Estimation" because the original paper is very mathsy and complex
        Reinforecement Learning, Sutton & Barto, 2018, 2nd ed, Section 7.3 - λ-backwards return
        Supplementary material section of Peng, X.B., Abbeel, P., Levine, S. and van de Panne, M. (2018) DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills. arXiv preprint arXiv:1804.02717. Available at: https://arxiv.org/abs/1804.02717
        https://www.reddit.com/r/reinforcementlearning/comments/kjxrpk/generalised_advantage_estimator_gae_how_does_it/
        
    """

    T = len(rewards)
    gae = 0
    returns = np.zeros(T, dtype = np.float32)

    for t in reversed(range(T)): # collecting advantages backwards, kind of like the backward view of TD(λ)
        delta = rewards[t] + gamma * values[t+1] * masks[t] - values[t] # computing the one step TD error
        gae = delta + gamma * lam * masks[t] * gae # collecting the advantages
        returns[t] = gae + values[t]

    return returns

class RolloutBuffer:
    """
    Stores and clears PPO trajectories
    Sebastian's video https://www.youtube.com/watch?v=PetlIokI9Ao helped me understand the role of buffers
    """

    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, state, action, log_prob, reward, done, value):
        """
        Collects timesteps of experience (actions, states etc), used to update the agents in the train function below
        """

        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(1.0 - float(done))  # mask = 1 - done
        self.values.append(value)


class UnityTennisWrapper:
    """
    This class represents the tennis environment. Intuition for this code was influenced by 
    Joshua Evans' RacetrackEnv class for CW1 in the CM50270 module.
    """

    def __init__(self, env, brain_name):
        self.env = env
        self.brain_name = brain_name # https://github.com/miyamotok0105/unity-ml-agents/blob/master/docs/Learning-Environment-Design-Brains.md
        self.brain = env.brains[brain_name]
        self.num_agents = None
        self.action_size = None
        self.state_size = None

    def reset(self, train_mode=True):
        env_info = self.env.reset(train_mode = train_mode)[self.brain_name]
        self.num_agents = len(env_info.agents)
        self.action_size = self.brain.vector_action_space_size
        self.state_size = len(env_info.vector_observations[0])
        return env_info.vector_observations

    def step(self, actions):
        env_info = self.env.step(actions)[self.brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        return next_states, rewards, dones, None  

env = UnityEnvironment(file_name = "/Users/georg/deep-reinforcement-learning/Tennis_Windows_x86_64/Tennis.exe")

# Extracting the "brain"
brain_name = env.brain_names[0]
brain = env.brains[brain_name]     
wrapped_env = UnityTennisWrapper(env, brain_name)
states = wrapped_env.reset(train_mode = True)

def train(env, episodes = 700, rollout_length = 2048, initial_entropy_coef = 0.02):
    """
    I wrote this function to neatly train the agents on the Unity Tennis Environment

    Initially i defined each episode as one match, but found that the training was slow to converge and often plateaued as there wasn't always enough 
    experience in a match to help meaningfully update the policy. I then decided to have a consistent amount of timesteps represent an episode through rollouts to
    stabilise training (this is better than defining say 10 matches as an episode as by that logic episodes would last much longer near the end of training as the 
    matches themselves would be longer given the agents would learn to hit the ball).
    
    I also am making use of entropy decay. In the first episodes, I prioritise exploration, but as episode count increases, i lower the entropy coefficient
    to allow for exploitation bias

    I am scaling the rewards of +0.1 when the agent hits the ball to the other side and -0.01 if the ball hits the ground on its side of the pitch by a factor of 10
    as i found this helps converge training more stably
    """

    num_agents = env.num_agents
    ppo = MultiAgentPPO(num_agents, STATE_SIZE, ACTION_SIZE)
    buffers = [RolloutBuffer() for _ in range(num_agents)]
    all_scores = []

    states = env.reset(train_mode = True)
    match_counter = 0
    match_counter_plot = []
    match_avg_plot = []

    for upd in range(1, episodes + 1):
        ent_coef = initial_entropy_coef * max(0.0, 1 - upd/episodes)

        # reseting per-rollout stats
        match_rewards = np.zeros(num_agents)
        volley_counts = np.zeros(num_agents, dtype=int)

        # collecting exactly rollout_length timesteps
        steps = 0
        while steps < rollout_length:
            actions, logps = ppo.act(states)
            next_states, rewards, dones, _ = env.step(actions)

            rewards_arr = np.array(rewards, dtype = np.float32)

            # accumulating rewards for the episode print statement
            match_rewards += rewards_arr
            volley_counts += (rewards_arr > 0).astype(int)

            # storing into PPO buffers
            for i in range(num_agents):
                with torch.no_grad():
                    val = ppo.agents[i].critic(torch.FloatTensor(states[i]).unsqueeze(0).to(DEVICE)).squeeze(-1).item() # this is to match the shapes as defined in the critic network

                buffers[i].states.append(states[i])
                buffers[i].actions.append(actions[i])
                buffers[i].log_probs.append(logps[i])
                buffers[i].rewards.append(rewards_arr[i] * 10.0)
                buffers[i].dones.append(1.0 - float(dones[i]))
                buffers[i].values.append(val)

            steps += 1

            # immediate reset on terminal, but NO logging here
            if any(dones):
                states = env.reset(train_mode = True)
            else:
                states = next_states

        # each episode is defined according to the rollout_length parameter. I experimented with values of 256, 1024 and 2048
        # I found 2048 to allow for faster episodal convergence as each batch would contain adequate data to allow for effective updates in the policy and value functions

        match_counter += 1
        match_counter_plot.append(match_counter)
        match_max = match_rewards.max()
        match_avg = match_rewards.mean()
        match_avg_plot.append(match_avg)
        best_agent = match_rewards.argmax()
        best_volleys = volley_counts[best_agent]
        all_scores.append(match_max)

        print(f"Episode {match_counter:4d} | "f"Volleys: {best_volleys:3d} | "f"Max Reward: {match_max:.3f} | "f"Avg Reward: {match_avg:.3f}")
        
        # building trajectories and updating the PPO agent
        trajectories = []
        for i, buf in enumerate(buffers):
            with torch.no_grad():
                next_val = ppo.agents[i].critic(torch.FloatTensor(states[i]).unsqueeze(0).to(DEVICE)).squeeze(-1).item()

            values = buf.values + [next_val]
            returns = GAE(buf.rewards, buf.dones, values)
            advantages = np.array(returns) - np.array(buf.values)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            trajectories.append((buf.states, buf.actions, buf.log_probs, returns, advantages))

        ppo.update(trajectories, ent_coef)

        # clearing buffers for the next update
        for buf in buffers:
            buf.clear()

    # plotting rewards for comparison in the report
    plt.plot(match_counter_plot, match_avg_plot)
    plt.ylabel("Episode Reward")
    plt.xlabel("Episode count")
    plt.title("Episodal rewards in PPO implementation")
    plt.show()

    return ppo, all_scores

ppo, scores = train(wrapped_env)

## used for the untrained ppo video showcase
#num_agents = wrapped_env.reset(train_mode=False).shape[0]
#untrained_ppo = MultiAgentPPO(num_agents, STATE_SIZE, ACTION_SIZE)

def evaluate(mappo,env, n_episodes = 5, alpha = 0.85, max_steps = 1000):
    """
    Because I am using a stochastic policy, there is sampling happening at every timestep. This leads to "jittery" movement in the agents during 
    evaluation. To eliminate this, I got the idea of implementing action smoothening from the paper:
    Mysore, S., Mabsout, B., Mancuso, R. and Saenko, K., 2021. Regularizing action policies for smooth control with reinforcement learning. 
    In: IEEE International Conference on Robotics and Automation (ICRA), Xi'an, China,2021, pp. 1810-1816. IEEE. doi:10.1109/ICRA48506.2021.9561138

    I am specifically implementing exponential smoothening using the equation found on https://en.wikipedia.org/wiki/Exponential_smoothing

    "Values of α close to 1 have less of a smoothing effect and give greater weight to recent changes in the data, while values of α 
    closer to 0 have a greater smoothing effect and are less responsive to recent changes." I have left alpha = 0.85 because I only wanted a small
    smoothening effect
    """

    for agent in mappo.agents:
        agent.actor.eval()

    for ep in range(1, n_episodes + 1): # because count begins from 0
        states = env.reset(train_mode = False)
        scores = np.zeros(env.num_agents)
        done, step = False, 0

        # initialising smoothed actions to zeros
        prev_actions = [np.zeros(env.action_size) for _ in range(env.num_agents)]

        while not done and step < max_steps: # max steps in case the agents get really good and the episode takes really long to terminate
            actions, _ = mappo.act(states)  # stochastically sampling

            # applying exponential smoothing
            smoothed = []
            for a_prev, a_new in zip(prev_actions, actions):
                a_s = alpha * a_new + (1 - alpha) * a_prev # eq from wikipedia link above
                smoothed.append(a_s)
            prev_actions = smoothed

            next_states, rewards, dones, _ = env.step(smoothed)
            scores += rewards
            states = next_states
            done = np.any(dones)
            step += 1

        print(f"Evaluation Ep {ep} | Scores: {scores} | Max: {scores.max():.3f}")
        
ppo_eval = evaluate(ppo, wrapped_env, n_episodes = 100)