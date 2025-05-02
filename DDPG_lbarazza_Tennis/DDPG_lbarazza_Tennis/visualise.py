from pathlib import Path
from collections import deque
from unityagents import UnityEnvironment
import numpy as np
from ddpg_agent import DDPGAgent

alpha = 0.85   # between 0 (max smoothing) and 1 (no smoothing)

env = UnityEnvironment(file_name="/Users/georg/deep-reinforcement-learning/p3_collab-compet/Tennis_Windows_x86_64/Tennis.exe")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
state = env_info.vector_observations
state_size = state.shape[1]

agent1 = DDPGAgent(nS=state_size, nA=action_size, lr_actor=0.0005, lr_critic=0.0005,
                   gamma=0.99, batch_size=60, tau=0.001,
                   memory_length=int(1e6), no_op=int(1e3), net_update_rate=1,
                   std_initial=0.15, std_final=0.025, std_decay_frames=200000)
agent2 = DDPGAgent(nS=state_size, nA=action_size, lr_actor=0.0005, lr_critic=0.0005,
                   gamma=0.99, batch_size=60, tau=0.001,
                   memory_length=int(1e6), no_op=int(1e3), net_update_rate=1,
                   std_initial=0.15, std_final=0.025, std_decay_frames=200000)

run_name = 'sample_test03'
for idx, agent in enumerate([agent1, agent2], start=1):
    ckpt = Path(f'checkpoints/{run_name}_{idx}.tar')
    if ckpt.is_file():
        agent.load(str(ckpt))

prev_actions = [np.zeros(action_size) for _ in range(num_agents)]

rets = deque(maxlen=100)
for episode in range(1, 31):
    ret = 0
    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations
    done = False

    prev_actions = [np.zeros(action_size) for _ in range(num_agents)]

    while not done:
        raw_a1 = agent1.choose_action(states[0])
        raw_a2 = agent2.choose_action(states[1])
        raw_actions = [raw_a1, raw_a2]

        smoothed_actions = []
        for i, (a_prev, a_new) in enumerate(zip(prev_actions, raw_actions)):
            a_s = alpha * a_new + (1 - alpha) * a_prev
            smoothed_actions.append(a_s)
        prev_actions = smoothed_actions

        actions_stack = np.vstack(smoothed_actions)
        env_info = env.step(actions_stack)[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done

        ret += rewards[0]
        states = next_states
        done = np.any(dones)

    rets.append(ret)
    avg = np.mean(rets)
    print(f'Episode {episode:2d}\tScore: {ret:.2f}\tAverage (100): {avg:.2f}')

env.close()
